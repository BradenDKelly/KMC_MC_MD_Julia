# Concentrated mixtures (x = 0.5; user 2026-10-18: "test stock kmc and our latest version on systems that aren't dilute,
# 1:1 ethanol in water and 1:1 ethanol in methanol; compare ΔG solvation against GROMACS for the same systems"):
#   mixEW = 200 ethanol + 200 SPC/E, mixEM = 200 ethanol + 200 methanol (OPLS-AA flexible, RF ε_rf = ∞, site cut-off 10 Å,
#   100 kT overlap cap), NVT 300 K in the box of the GROMACS NPT run; start = the GROMACS NVT frame
#   (gromacs_fe/mixtures/<mix>/nvt.gro; molecule order ETHX, SOLX|MOHX, 199 ETHF, 199 SOL|MOHF as in gromacs_fe/<mix>.top).
# Modes: stock   — stock eKMC (relocations with ideal-gas conformers), μ_ex of every species from the removal estimator
#                  (Tan 2017; time-weighted ⟨exp(βu_i)⟩ per species);
#        ladder1 — latest version, coupling ladder on molecule 1 (an ethanol), λ = 0 hub capped at λ ≤ 0.03, local + internal
#                  L-moves of it, flexible solvent L-moves of all others (rigid water: T/R slots), Wang–Landau then frozen η,
#                  μ_ex by MBAR (settings of the dilute _jm3 production);
#        ladder2 — the same on molecule 2 (the water / methanol); single-site/torsion local moves only if that molecule is flexible.
# μ reported GROMACS-comparable: μ + E_const (Coulomb molecule constant) + ΔU_tail (LJ tail of one molecule), as for the dilute
# solutes. Chains: SPCE_CHECKPOINT, SPCE_WALL_SECONDS, SPCE_CKPT_EVERY as dev/validate_spce_solute.jl.
# Usage: julia -t 4 --project=. dev/mixture_ekmc.jl <mixEW|mixEM> <stock|ladder1|ladder2> <events> <seed> <outdir>
using MolSim, Random, Printf, Dates, Serialization
# MIX_DEVICE=cuda: device pair rows + solvent-candidate bank (DESIGN §15.5) and, with PME, the mesh gathers (§16) on the GPU;
# needs an environment with CUDA (e.g. envs/molsim_gpu_*). Device state is never serialized: detached around checkpoints.
const MIX_DEVICE = Symbol(get(ENV, "MIX_DEVICE", "none"))
if MIX_DEVICE === :cuda
    using CUDA
end
t_script = time()
const MCm = MolSim.MC
mix = ARGS[1]; mode = ARGS[2]; events = parse(Int, ARGS[3]); seed = parse(Int, ARGS[4]); outdir = ARGS[5]
mix in ("mixEW", "mixEM") || error("mix must be mixEW or mixEM"); mode in ("stock", "ladder1", "ladder2") || error("bad mode")
mkpath(outdir)
ckpt = get(ENV, "SPCE_CHECKPOINT", ""); wall = parse(Float64, get(ENV, "SPCE_WALL_SECONDS", "Inf"))
ckpt_every = parse(Float64, get(ENV, "SPCE_CKPT_EVERY", "1800"))
gro = get(ENV, "MIX_GRO", "/shared/kmc_polyatomic/gromacs_fe/mixtures/$mix/nvt.gro")
wl_1t = get(ENV, "MIX_LADDER_WL1T", "0") == "1"   # 1/t Wang–Landau (Belardinelli & Pereyra 2007): η keeps adapting through production
pool_M = parse(Int, get(ENV, "MIX_POOL_M", "20000"))         # conformers per flexible species (20000 as the dilute runs; smaller only for smoke tests)
coulomb_req = Symbol(get(ENV, "MIX_COULOMB", "rf"))          # rf (ε_rf = ∞) | ewald (α from ewald-rtol 1e-5 at 10 Å) | pme (SPME mesh path)
coulomb_req in (:rf, :ewald, :pme, :wolf) || error("MIX_COULOMB must be rf, ewald, pme or wolf")   # wolf: α = 0.2/Å (as the dilute runs)
coulomb = coulomb_req === :pme ? :ewald : coulomb_req
out = joinpath(outdir, "mix_$(mix)_$(mode)_seed$(seed)$(coulomb_req === :rf ? "" : "_$(coulomb_req)")$(wl_1t ? "_wl1t" : "").txt")

function setup(io)
    solutes = mix == "mixEW" ? [:ethanol_flex] : [:ethanol_flex, :methanol_itp]
    m = MCm.spce_mixture_model(solutes; rc_angstrom=10.0, cutoff=:site, coulomb=coulomb, epsilon_rf=Inf,
                               α_per_angstrom=coulomb === :ewald ? MCm.ewald_alpha_for(10.0, 1e-5) : 0.20)
    T = 300.0 / m.ϵ_kelvin; kJ = m.ϵ_kelvin * 1.3806488e-23 * 6.02214129e23 / 1000
    atm = m.ϵ_kelvin * 1.3806488e-23 / (m.σ_angstrom * 1e-10)^3 / 101325.0
    ff = first(MCm.kT_overlap_cap(m.ff, m.templates, T; U_kT=100.0))
    # template index per molecule, in the order of the .gro / GROMACS topology
    te = 2; to = mix == "mixEW" ? 1 : 3
    tmpl = vcat(te, to, fill(te, 199), fill(to, 199))
    lines = readlines(gro); nat = parse(Int, strip(lines[2])); box = parse.(Float64, split(lines[end])[1:3])
    (box[1] ≈ box[2] ≈ box[3]) || error("cubic box expected")
    nat == sum(m.templates[t].n_sites for t in tmpl) || error("$gro: $nat atoms, model $(sum(m.templates[t].n_sites for t in tmpl))")
    L = box[1] * 10 / m.σ_angstrom
    X = zeros(3, nat)
    for k in 1:nat
        l = lines[2 + k]; X[:, k] .= parse.(Float64, (l[21:28], l[29:36], l[37:44])) .* (10 / m.σ_angstrom)
    end
    sys = MCm.RigidSystem(m.templates, tmpl, L, 1 / T, Xoshiro(seed))
    s = 0
    for i in eachindex(tmpl)
        t = m.templates[tmpl[i]]; Y = X[:, s+1:s+t.n_sites]; s += t.n_sites
        for a in 2:t.n_sites, d in 1:3; Y[d, a] = Y[d, 1] + MCm._min_image(Y[d, a] - Y[d, 1], L); end
        if MCm.isflexible(t)
            MCm.set_conformation!(sys, i, Y); MCm.set_molecule!(sys, i, (Y * t.site_mass) ./ sum(t.site_mass), sys.quat[:, i])
        else
            MCm.molecule_from_sites!(sys, i, Y)
        end
    end
    # reciprocal-space state from the loaded configuration; every move after this keeps it current
    if coulomb_req === :pme                                  # GROMACS defaults: fourier-spacing 0.12 nm, pme-order 4
        MCm.attach_pme!(sys, ff; rtol=1e-5, spacing=1.2 / m.σ_angstrom, order=4)
    elseif coulomb === :ewald
        MCm.attach_ewald!(sys, ff; rtol=1e-5, backend=Symbol(get(ENV, "MIX_EWALD_BACKEND", "cpu")))
    end
    ncap, _ = MCm.capped_overlaps(sys, ff, 1)
    @printf(io, "# %s %s seed %d  %s  %d molecules (%s), L = %.4f Å (GROMACS NVT box), T = 300 K, %s, site cut-off 10 Å, cap kT100\n",
            mix, mode, seed, Dates.now(), length(tmpl), join(["$(count(==(k), tmpl)) $(m.templates[k].name)" for k in unique(tmpl)], " + "), L * m.σ_angstrom, coulomb === :rf ? "RF ε_rf = ∞" : coulomb === :wolf ? "Wolf α = 0.2 1/Å (damped shifted)" : sys.recip isa MCm.PMERecip ? "SPME mesh $(sys.recip.K)³ order $(sys.recip.p), α = $(round(MCm.ewald_alpha_for(10.0, 1e-5); digits=5)) 1/Å (NVT; no pressure)" : "Ewald α = $(round(MCm.ewald_alpha_for(10.0, 1e-5); digits=5)) 1/Å, $(MCm.n_kvectors(sys.recip)) k-vectors (NVT; no pressure)")
    # ideal-gas conformer pools of the flexible species (vacuum rejection-free eKMC with torsion slots, as the dilute runs)
    reservoirs = Any[nothing for _ in m.templates]
    for k in unique(tmpl)
        MCm.isflexible(m.templates[k]) || continue
        t0 = time()
        reservoirs[k] = MCm.build_reservoir_ekmc(m.templates, k, ff, 1 / T, Xoshiro(seed + 7919 + k); M=pool_M, burnin_sweeps=2000,
                                                 stride_sweeps=20, δ=0.03, torsion=true)
        @printf(io, "# conformer pool %s: %d conformers, <U_intra>_ideal = %.3f kJ/mol, %.0f s\n", m.templates[k].name,
                length(reservoirs[k].confs), sum(reservoirs[k].U_intra) / length(reservoirs[k].U_intra) * kJ, time() - t0)
        flush(io)
    end
    # per-species corrections: LJ tail of one molecule, Coulomb molecule constant
    spec = unique(tmpl); ΔU_tail = Dict{Int,Float64}(); E_const = Dict{Int,Float64}()
    for k in spec
        rest = copy(tmpl); deleteat!(rest, findfirst(==(k), rest))
        ΔU_tail[k] = MCm.lrc_energy(sys, ff) - MCm.lrc_energy(MCm.RigidSystem(m.templates, rest, L, 1 / T, Xoshiro(0)), ff)
        E_const[k] = MCm.coulomb_molecule_constant(m.templates[k], ff)
        @printf(io, "# species %d (%s): ΔU_tail = %.4f, E_const = %.4f kJ/mol\n", k, m.templates[k].name, ΔU_tail[k] * kJ, E_const[k] * kJ)
    end
    δ = 0.1 / m.σ_angstrom * 3.0; δθ = 0.3
    if mode == "stock"
        r = MCm.init_rigid_ekmc_run(sys, ff; events=events, sample_every=100, block_events=max(25, events ÷ 25), species_mu=true,
                                    reservoirs=reservoirs, debug_check_every=events ÷ 10)
    else
        # ν from a stock-eKMC pilot (SPCE_NU_SOURCE=ekmc of the dilute runs): per-molecule time-weighted ⟨R/N⟩
        tp = time(); npil = parse(Int, get(ENV, "MIX_PILOT", string(4000 * length(tmpl))))   # stock pilot events (default as the dilute runs)
        rp = MCm.init_rigid_ekmc_run(sys, ff; events=npil, sample_every=100, block_events=max(25, npil ÷ 25), species_mu=true,
                                     reservoirs=reservoirs)
        MCm.advance_rigid_ekmc!(rp, sys, ff)
        rpr = MCm.rigid_ekmc_results(rp)
        Rsp = [MCm.ratio_mean(rpr.RN_species[k]) for k in eachindex(m.templates)]
        nk = [count(==(k), tmpl) for k in eachindex(m.templates)]
        R_mc = sum(Rsp[k] * nk[k] for k in eachindex(nk) if nk[k] > 0) / length(tmpl)
        fr = mode == "ladder1" ? 1 : 2
        βμ_guess = log(Rsp[tmpl[fr]])
        @printf(io, "# stock eKMC pilot %d events (%.0f s): <R/N> = %.4e; per species %s; ladder on molecule %d (%s), η ramp from βμ = %.3f\n",
                npil, time() - tp, R_mc, join([@sprintf("%.3e", Rsp[k]) for k in spec], " "), fr, m.templates[tmpl[fr]].name, βμ_guess)
        flush(io)
        r = MCm.init_rigid_ekmc_run(sys, ff; events=events, sample_every=100, block_events=max(25, events ÷ 25), species_mu=true,
                                    reservoirs=reservoirs, debug_check_every=events ÷ 10, mc_δ=δ, mc_δθ=δθ,
                                    ladder_solute=fr, ladder_nu=30.0 * R_mc, ladder_wl_events=round(Int, 0.35 * events),
                                    ladder_eta_ramp=βμ_guess, ladder_jump=true, ladder_jump_max=3, ladder_local=true, ladder_wl_1t=wl_1t,
                                    local_internal=MCm.isflexible(m.templates[tmpl[fr]]), local_solvent=true, solvent_flexible=true, solvent_nu=R_mc)
        println(io, "# ladder: ν = 30·<R/N> = $(30.0 * R_mc), $(wl_1t ? "1/t Wang–Landau (adapting through production), production after" : "WL over") $(round(Int, 0.35 * events)) events, hub ≤ λ_3; solvent L-moves $(length(r.slmoves.mols)) movers × $(r.slmoves.kinds)")
    end
    flush(io)
    frk = mode == "stock" ? 0 : tmpl[mode == "ladder1" ? 1 : 2]
    ctx = (; out, T, kJ, atm, ΔU_tail, E_const, spec, names=Dict(k => m.templates[k].name for k in spec), t_ek=0.0, mode, frk)
    return sys, r, ff, ctx
end

function report(io, r, ctx)
    (; T, kJ, atm, ΔU_tail, E_const, spec, names, t_ek, mode, frk) = ctx
    ek = MCm.rigid_ekmc_results(r)
    @printf(io, "RESULT %s %s events=%d time=%.0fs  U/N=%.4f ± %.4f kJ/mol  P=%.1f ± %.1f atm  <U_intra>=%.3f ± %.3f kJ/mol  relocations=%d\n",
            mix, mode, r.ev, t_ek, MCm.mean(ek.U) * kJ, MCm.stderr(ek.U) * kJ, MCm.mean(ek.P) * atm, MCm.stderr(ek.P) * atm,
            MCm.mean(ek.U_intra) * kJ, MCm.stderr(ek.U_intra) * kJ, ek.relocations)
    for k in spec
        rk = ek.RN_species[k]
        μ = T * log(MCm.ratio_mean(rk)); eμ = T * MCm.ratio_jackknife_stderr(rk; f=log)
        @printf(io, "RESULT stock-removal mu_ex %s = %.3f ± %.3f kJ/mol (GROMACS-comparable: + E_const + tail; jackknife over %d blocks)\n",
                names[k], (μ + E_const[k] + ΔU_tail[k]) * kJ, eμ * kJ, length(rk.block_sum_x))
    end
    if ek.ladder !== nothing
        L = ek.ladder; k = frk
        sm = MCm.ladder_summary(L); mb, emb = MCm.ladder_mu_mbar(L, T); oc, eoc = MCm.ladder_mu_occupancy(L, T)
        corr = (E_const[k] + ΔU_tail[k]) * kJ
        @printf(io, "RESULT ladder mu_ex %s = MBAR %.3f ± %.3f, occupancy %.3f ± %.3f kJ/mol (GROMACS-comparable); WL stages %d frozen at %d; rung t = %s\n",
                names[k], mb * kJ + corr, emb * kJ, oc * kJ + corr, eoc * kJ, sm.wl_stages, sm.ev_frozen, join([@sprintf("%.3f", x) for x in sm.frac_rung], " "))
    end
    flush(io)
end

function save_checkpoint(path, state)
    tmp = path * ".tmp"; open(io -> serialize(io, state), tmp, "w"); mv(tmp, path; force=true)
end
if !isempty(ckpt) && isfile(ckpt * ".final")
    println("run already complete"); exit(0)
end
if !isempty(ckpt) && isfile(ckpt)
    sys, r, ff, ctx = deserialize(ckpt); io = open(ctx.out, "a")
    println(io, "# resumed from checkpoint at event $(r.ev) of $(r.events)  $(Dates.now())  host=$(gethostname())"); flush(io)
else
    io = open(out, "w"); sys, r, ff, ctx = setup(io)
    isempty(ckpt) || save_checkpoint(ckpt, (sys, r, ff, ctx))
end
function dev_attach!(sys, r, ff)
    MIX_DEVICE === :none && return nothing
    S = r.slmoves
    np = S === nothing ? 1 : length(S.kinds) * length(S.mols) + 1
    MCm.attach_device_pairs!(sys, ff; backend=MIX_DEVICE, precision=:double, max_poses=max(np, 64))
    S === nothing || MCm.slm_attach_device!(S, sys, ff)
    sys.recip isa MCm.PMERecip && MCm.pme_attach_device!(sys.recip, sys; backend=MIX_DEVICE)
    return nothing
end
function dev_detach!(sys, r)
    sys.dev = nothing
    S = r.slmoves
    S === nothing || (S.dbase = 0; S.crP === nothing || (S.crP.dev = nothing))
    sys.recip isa MCm.PMERecip && (sys.recip.dev = nothing; sys.recip.backend = :cpu)
    return nothing
end
dev_attach!(sys, r, ff)
MIX_DEVICE === :none || println(io, "# device $(MIX_DEVICE): pair rows, solvent-candidate bank$(sys.recip isa MCm.PMERecip ? ", PME mesh gathers" : "") ($(Dates.now()), host $(gethostname()))")
while r.ev < r.events
    left = wall - (time() - t_script); left > 0 || break
    t0 = time()
    MCm.advance_rigid_ekmc!(r, sys, ff; wall_seconds=isempty(ckpt) ? left : min(left, ckpt_every))
    global ctx = merge(ctx, (; t_ek=ctx.t_ek + time() - t0))
    if r.ev < r.events && !isempty(ckpt)
        dev_detach!(sys, r); save_checkpoint(ckpt, (sys, r, ff, ctx)); dev_attach!(sys, r, ff)
        println(io, "# checkpoint at event $(r.ev) of $(r.events)  $(Dates.now())  eKMC wall time so far $(round(ctx.t_ek; digits=0)) s"); flush(io)
    end
end
if r.ev < r.events
    close(io); println("checkpoint written at event $(r.ev) of $(r.events)"); exit(0)
end
report(io, r, ctx); close(io)
isempty(ckpt) || mv(ckpt, ckpt * ".final"; force=true)

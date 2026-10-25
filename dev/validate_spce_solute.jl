"""
One solute (OPLS-AA all-atom methane or methanol, rigid; `methanol_flex` = fully flexible OPLS-AA methanol with a
conformer reservoir, DESIGN §7; TraPPE-UA `methane_ua`/`methanol_ua` for reference)
at infinite dilution in SPC/E water: Metropolis MC (Widom insertion and particle-removal estimators of the
solute μ_ex) vs hybrid eKMC (per-species ⟨exp(βu)⟩ with delete-one-block jackknife errors). JOURNAL
2026-10-12 entry 14.

State: T = 300 K; N water molecules in the box of pure water at 1000 kg/m³ plus one solute.
Model: `spce_solute_model` — SPC/E with molecule-cutoff Wolf (α = SPCE_ALPHA Å⁻¹, r_c = 10 Å, cap 1000 ϵ,
LJ tail) and the rigid solute (OPLS-AA: geometric mixing; TraPPE-UA: Lorentz–Berthelot). The LJ tail of
the solute, ΔU_tail = U_tail(N water + solute) − U_tail(N water), is added to every solute μ_ex
(Frenkel & Smit §3.2.2). Widom here inserts a second solute into the system holding the real one: at
N = 500 the composition effect is O(1/N) (it is +0.2 kT for N = 32 LJ, test suite), negligible here.

Reference: the GROMACS alchemical decoupling (BAR over 21 λ windows, PME, same box, same rigid solute,
gromacs_fe/) of the identical model, −ΔG_decouple = μ_ex. Literature context for united-atom methane in
SPC/E at 300 K: μ_ex = 9.71/9.56/9.45 ± 0.1 kJ/mol for N = 256/500/864 (Paschek, J. Chem. Phys. 120 (2004)
6674, Table IV, OPLS-UA CH4, Widom) and ΔG_hyd = 9.39 kJ/mol for TraPPE-UA CH4 (Ramezani & Sharma, J. Phys.
Chem. B 130 (2026) 4636, Table 1); experiment: methane 8.37 kJ/mol (same table), methanol −5.1 kcal/mol =
−21.3 kJ/mol (Ben-Naim & Marcus, J. Chem. Phys. 81 (1984) 2016).

Usage: julia --project=. dev/validate_spce_solute.jl solute N mc_sweeps ekmc_events seed outdir
Env: SPCE_ALPHA (0.20), SPCE_CUTOFF (molecule), SPCE_COULOMB (wolf|rf|ewald|pme), SPCE_LAMBDA (100; λ = fac·⟨R⟩_MC, 0 = pure
eKMC), SPCE_SOLUTE_SHARE (0.5; share of relocation bursts given to the solute by rate scaling), SPCE_TAG,
SPCE_CHECKPOINT (path; checkpoint/restart of the eKMC stage for chained jobs), SPCE_WALL_SECONDS (eKMC wall budget
of one invocation), SPCE_CKPT_EVERY (1800 s between checkpoints), SPCE_BAR (0; 1 = Bennett/landing-Widom on the solute), SPCE_LADDER (0; 1 = coupling ladder on the solute,
SPCE_LADDER_LAMBDA rungs, SPCE_LADDER_NU ν/⟨R⟩_MC, SPCE_LADDER_WL fraction of events for Wang–Landau, SPCE_LADDER_RAMP μ guess in kJ/mol, SPCE_LADDER_ADAPT re-flatten η per block, SPCE_LADDER_JUMP λ = 0 hub jumps, SPCE_MC_LOCAL Å radius of local MC-kernel sweeps with SPCE_LAMBDA > 0), SPCE_TRACE (1; 0 drops the per-event solute trace), SPCE_LOCAL (0; 1 = rejection-free local moves of the fractional molecule,
DESIGN §11.2), SPCE_JUMP_MAX (0; hub limited to rungs 3…m), SPCE_RESERVOIR (mc | ekmc conformer pool), SPCE_TORSION (0; 1 = torsion moves in the conformer pool and the MC stage),
SPCE_LOCAL_INTERNAL (0; 1 = rejection-free site + torsion slots of the solute with SPCE_LOCAL), SPCE_LOCAL_PHI (torsion slot half-angle in rad, default mc_δθ; π = uniform full rotation), SPCE_SOLVENT_LOCAL (0; 1 = rejection-free local moves of every
water, SolventLocalMoves, ν = ν_D = the ladder ν). mc_sweeps = 0 runs the
MC-free pipeline (DESIGN §11.3): SPCE_EKMC_BURNIN, SPCE_NU_PILOT events of stock eKMC replace the Metropolis stage;
SPCE_NU_SOURCE=ekmc with mc_sweeps > 0 keeps MC only as the starting configuration and takes ν from an eKMC pilot.
"""

using MolSim, Random, StaticArrays, Printf, Dates, Serialization
# SPCE_DEVICE=cuda: device pair rows + solvent-candidate bank (DESIGN §15.5) and, with PME, the mesh gathers (§16) on the GPU,
# exactly as dev/mixture_ekmc.jl (MIX_DEVICE); eKMC stage only (MC stays on the host). Needs an environment with CUDA
# (envs/molsim_gpu_*). Device state is never serialized: detached around checkpoints.
const SPCE_DEVICE = Symbol(get(ENV, "SPCE_DEVICE", "none"))
if SPCE_DEVICE === :cuda
    using CUDA
end
t_script = time()   # SPCE_WALL_SECONDS counts from here, so a first piece's set-up (reservoir, MC or eKMC burn-in) is inside the budget
const MCm = MolSim.MC

solute = Symbol(length(ARGS) >= 1 ? ARGS[1] : "methane")
N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
mc_sweeps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8000
ekmc_events = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 100_000
seed = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1
outdir = length(ARGS) >= 6 ? ARGS[6] : joinpath(@__DIR__, "results", "spce_solute")
mkpath(outdir)
coulomb_req = Symbol(get(ENV, "SPCE_COULOMB", "wolf"))      # wolf | rf | ewald | pme (pme = Ewald real space + SPME mesh path)
coulomb = coulomb_req === :pme ? :ewald : coulomb_req
# Ewald (JOURNAL 121): α from GROMACS's ewald-rtol = 1e-5 at rc = 10 Å unless SPCE_ALPHA is given; site cut-off (required)
α_A = haskey(ENV, "SPCE_ALPHA") ? parse(Float64, ENV["SPCE_ALPHA"]) : coulomb === :ewald ? MCm.ewald_alpha_for(10.0, 1e-5) : 0.20
cutoff = Symbol(get(ENV, "SPCE_CUTOFF", coulomb === :ewald ? "site" : "molecule"))
fast_path = get(ENV, "SPCE_FAST_PATH", "0") == "1"            # overlap fast path switch (approximate for weakly bound species; default off)          # wolf | rf (GROMACS reaction field, SPCE_EPS_RF: 0 = ∞)
ε_rf = (x = parse(Float64, get(ENV, "SPCE_EPS_RF", "0")); x == 0 ? Inf : x)
elec_tag = coulomb === :rf ? "rf$(isinf(ε_rf) ? "inf" : ε_rf)" : coulomb_req === :pme ? "pme" : coulomb === :ewald ? "ewald" : "a$(α_A)"
λfac = parse(Float64, get(ENV, "SPCE_LAMBDA", "0"))      # 0 = stock eKMC (default, user 2026-10-12); >0 = λ-hybrid, λ = fac·⟨R⟩_MC
tag = get(ENV, "SPCE_TAG", "")
ckpt = get(ENV, "SPCE_CHECKPOINT", "")    # checkpoint file: eKMC continues from it if present (chained jobs); "" = off
wall = parse(Float64, get(ENV, "SPCE_WALL_SECONDS", "Inf"))   # wall-time budget of this invocation (set-up included, JOURNAL 70); checkpoint and exit when spent
ckpt_every = parse(Float64, get(ENV, "SPCE_CKPT_EVERY", "1800"))   # seconds of eKMC between checkpoints (with SPCE_CHECKPOINT)
bar_on = get(ENV, "SPCE_BAR", "0") == "1"   # Bennett accumulator on the solute (landing-Widom + BAR, DESIGN §10.3)
ladder_on = get(ENV, "SPCE_LADDER", "0") == "1"   # coupling ladder on the solute (DESIGN §10.4)
ladder_λ = parse.(Float64, split(get(ENV, "SPCE_LADDER_LAMBDA", "0,0.01,0.03,0.1,0.2,0.35,0.5,0.7,0.85,1"), ","))
ladder_νfac = parse(Float64, get(ENV, "SPCE_LADDER_NU", "1.0"))      # ν = fac · ⟨R⟩_MC
ladder_wlfrac = parse(Float64, get(ENV, "SPCE_LADDER_WL", "0.1"))    # fraction of the events spent on Wang–Landau
ladder_ramp_kJ = parse(Float64, get(ENV, "SPCE_LADDER_RAMP", "NaN"))  # βμ_guess for the η ramp from this μ (kJ/mol); NaN = MC removal estimate
ladder_adapt = get(ENV, "SPCE_LADDER_ADAPT", "0") == "1"   # re-flatten η at production block boundaries
ladder_jump = get(ENV, "SPCE_LADDER_JUMP", "0") == "1"     # λ = 0 hub: direct jumps 1 ↔ m ≥ 3, no fractional relocation at m ≥ 3 (JOURNAL 56)
mc_local_A = parse(Float64, get(ENV, "SPCE_MC_LOCAL", "0"))   # Å: MC-kernel sweeps restricted to this radius around the solute (0 = full sweeps)
local_on = get(ENV, "SPCE_LOCAL", "0") == "1"             # rejection-free local moves of the fractional molecule (DESIGN §11.2)
jump_max = parse(Int, get(ENV, "SPCE_JUMP_MAX", "0"))       # hub limited to rungs 3…SPCE_JUMP_MAX (0 = all)
torsion_on = get(ENV, "SPCE_TORSION", "0") == "1"          # torsion moves: pool (φ uniform ±π) and MC stage (±δθ) (JOURNAL 79–80)
internal_on = get(ENV, "SPCE_LOCAL_INTERNAL", "0") == "1"  # rejection-free site + torsion slots of the solute in solution (JOURNAL 80)
solvent_local = get(ENV, "SPCE_SOLVENT_LOCAL", "0") == "1"   # rejection-free local moves of every water (DESIGN §11.2 stage 2)
eta_env = get(ENV, "SPCE_LADDER_ETA", "")                    # fixed ladder weights η (comma list; with SPCE_LADDER_WL=0: no WL)
ladder_eta_fixed = isempty(eta_env) ? nothing : parse.(Float64, split(eta_env, ","))
track_dih = get(ENV, "SPCE_TRACK_DIH", "0") == "1"           # per-rung dihedral histograms of the fractional solute (diagnostic)
res_mode = get(ENV, "SPCE_RESERVOIR", "mc")                 # conformer pool: mc (vacuum Metropolis) | ekmc (vacuum rejection-free, §11.3 E3)
res_mode in ("mc", "ekmc") || error("SPCE_RESERVOIR must be mc or ekmc")
# mc_sweeps = 0: MC-free pipeline (DESIGN §11.3, user 2026-10-14): E1 eKMC burn-in of SPCE_EKMC_BURNIN events (default
# 4000·(N+1), the per-molecule move count of the Metropolis stage), E2 ν and the η-ramp guess from a stock eKMC pilot of
# SPCE_NU_PILOT events (default the same) — time-weighted ⟨(1/N_w) Σ exp(βu_i)⟩ of the water, the canonical quantity ⟨R⟩_MC estimates
ekmc_burnin = parse(Int, get(ENV, "SPCE_EKMC_BURNIN", string(4000 * (N + 1))))
nu_pilot = parse(Int, get(ENV, "SPCE_NU_PILOT", string(4000 * (N + 1))))
# ν source when the Metropolis stage runs: mc (⟨R⟩ of the MC production, default) | ekmc (stock eKMC pilot of SPCE_NU_PILOT events
# from the MC-equilibrated configuration, no extra burn-in: MC then only supplies the starting configuration; JOURNAL 65)
nu_source = get(ENV, "SPCE_NU_SOURCE", mc_sweeps > 0 ? "mc" : "ekmc")
nu_source in ("mc", "ekmc") || error("SPCE_NU_SOURCE must be mc or ekmc")
mc_sweeps == 0 && nu_source == "mc" && error("SPCE_NU_SOURCE=mc needs mc_sweeps > 0")
trace = get(ENV, "SPCE_TRACE", "1") == "1"   # per-event solute (u, Δt) trace for the tail report (16 bytes/event: off for ≳ 1e8 events)

m = MCm.spce_solute_model(solute; α_per_angstrom=α_A, rc_angstrom=10.0, cutoff=cutoff, coulomb=coulomb, epsilon_rf=ε_rf)
cap_mode = get(ENV, "SPCE_CAP", "kT100")   # kT100: 100 kT overlap cap (DESIGN §9, default from 2026-10-13); legacy: 1000 ε_ab at 0.75 σ_ab
cap_mode in ("kT100", "legacy") || error("SPCE_CAP must be kT100 or legacy")
ff = cap_mode == "legacy" ? m.ff : first(MCm.kT_overlap_cap(m.ff, m.templates, 300.0 / m.ϵ_kelvin; U_kT=100.0))
cap_tag = cap_mode == "legacy" ? "cap$(Int(m.ff.ucap_mix[1,1]))" : "capkT100"
out = joinpath(outdir, "spce_$(solute)_N$(N)_seed$(seed)_$(cutoff)_$(elec_tag)_$(cap_tag)$(tag).txt")
T_K = 300.0
T = T_K / m.ϵ_kelvin
ρ_A3 = 1000.0 * 1e3 / 18.0153 * 6.02214129e23 / 1e30                       # water molecules per Å³
ρstar = ρ_A3 * m.σ_angstrom^3
L = cbrt(N / ρstar)
L_A = L * m.σ_angstrom
kJmol_per_eps = m.ϵ_kelvin * 1.3806488e-23 * 6.02214129e23 / 1000
atm_per_pstar = m.ϵ_kelvin * 1.3806488e-23 / (m.σ_angstrom * 1e-10)^3 / 101325.0
Nt = N + 1; isol = Nt                                                       # the solute is molecule N+1

# time-weighted tail diagnostics of a single-molecule removal estimator ⟨exp(βu)⟩ = Σ w_k / Σ Δt_k
function tail_report(io, phi, dt, β, name)
    isempty(phi) && return
    w = [MCm._mobility(β, phi[k]) * dt[k] for k in eachindex(phi)]
    S = sum(w); tt = sum(dt)
    ord = sortperm(w; rev=true)
    n = length(w)
    frac(p) = sum(w[ord[1:max(1, round(Int, p * n))]]) / S
    neff = S^2 / sum(abs2, w)                                                  # Kish (Survey Sampling, 1965) effective sample size
    # time-weighted quantiles of βu
    o2 = sortperm(phi); c = cumsum(dt[o2]) ./ tt
    q(p) = β * phi[o2[searchsortedfirst(c, p)]]
    @printf(io, "# %s tail: samples=%d  Kish n_eff=%.1f  share of Σw from top 0.1%%/1%%/10%% of samples = %.3f/%.3f/%.3f  βu quantiles 5/50/95/99.9%% = %.2f/%.2f/%.2f/%.2f  max βu=%.2f\n",
            name, n, neff, frac(0.001), frac(0.01), frac(0.1), q(0.05), q(0.5), q(0.95), q(0.999), β * maximum(phi))
end

# device residency (SPCE_DEVICE=cuda) for an eKMC run r on sys: attached for the pilot and production, detached
# around every serialize (CuArrays are not checkpointable)
function dev_attach!(sys, r, ff)
    SPCE_DEVICE === :none && return nothing
    S = r.slmoves
    np = S === nothing ? 1 : length(S.kinds) * length(S.mols) + 1
    MCm.attach_device_pairs!(sys, ff; backend=SPCE_DEVICE, precision=:double, max_poses=max(np, 64))
    S === nothing || MCm.slm_attach_device!(S, sys, ff)
    sys.recip isa MCm.PMERecip && MCm.pme_attach_device!(sys.recip, sys; backend=SPCE_DEVICE)
    return nothing
end
function dev_detach!(sys, r)
    sys.dev = nothing
    S = r.slmoves
    S === nothing || (S.dbase = 0; S.crP === nothing || (S.crP.dev = nothing))
    sys.recip isa MCm.PMERecip && (sys.recip.dev = nothing; sys.recip.backend = :cpu)
    return nothing
end

function setup_and_mc(io)
    println(io, "# SPC/E + $(m.solute.name) at infinite dilution  $(Dates.now())  N_water=$N + 1 solute  L=$(round(L_A; digits=3)) Å  T=$T_K K  ρ_water=1000 kg/m³")
    println(io, "# Coulomb=$(coulomb) (Wolf α=$(α_A) 1/Å | RF ε_rf=$(ε_rf)) rc=10 Å cutoff=$(cutoff) cap=$(cap_tag); LJ types σ*=$(round.([ff.σ_mix[a,a] for a in 1:ff.n_types]; digits=4)) ϵ*=$(round.([ff.ϵ_mix[a,a] for a in 1:ff.n_types]; digits=4))")
    flush(io)
    rc_check = ff.rc + m.water.extent + max(m.water.extent, m.solute.extent)
    rc_check < L / 2 || println(io, "# WARNING: rc + molecule extents = $(rc_check) ≥ L/2 = $(L/2)")

    # flexible solute: ideal-gas conformer reservoir (vacuum Metropolis MC, DESIGN §7.2); nothing for rigid solutes
    flex = MCm.isflexible(m.solute)
    res = nothing
    if flex
        t0 = time()
        res = res_mode == "mc" ?
            MCm.build_reservoir(m.templates, 2, ff, 1 / T, Xoshiro(seed + 7919); M=20000, burnin_sweeps=2000, stride_sweeps=20, δ=0.03, torsion=torsion_on) :
            MCm.build_reservoir_ekmc(m.templates, 2, ff, 1 / T, Xoshiro(seed + 7919); M=20000, burnin_sweeps=2000, stride_sweeps=20, δ=0.03, torsion=torsion_on)
        Ures = sum(res.U_intra) / length(res.U_intra)
        @printf(io, "# conformer reservoir (%s): %d conformers, acceptance %.3f, <U_intra>_ideal = %.3f ± %.3f kJ/mol, %.1f s\n",
                res_mode == "mc" ? "vacuum Metropolis" : "vacuum rejection-free eKMC, acceptance = move fraction", length(res.confs), res.acceptance, Ures * kJmol_per_eps,
                sqrt(sum(abs2, res.U_intra .- Ures) / length(res.U_intra)) / sqrt(length(res.U_intra) / 5) * kJmol_per_eps, time() - t0)
        flush(io)
    end
    reservoirs = [nothing, res]
    sys = MCm.RigidSystem(m.templates, vcat(fill(1, N), 2), L, 1 / T, Xoshiro(seed))
    MCm.random_rigid_system!(sys, ff)
    if flex
        MCm._set_body!(sys, isol, MCm.draw_conformer(res, sys.rng))
        MCm.set_molecule!(sys, isol, sys.com[:, isol], MCm.uniform_random_quaternion(sys.rng))
    end
    mres = MCm.minimize_rigid!(sys, ff; nsteps=500)   # steepest descent before MC (JOURNAL entry 25)
    println(io, "# minimize_rigid!: $(mres.steps) steps, U_pair $(round(mres.U_start; sigdigits=6)) → $(round(mres.U_end; sigdigits=6)) (reduced), capped pairs left $(mres.n_capped)"); flush(io)
    if coulomb_req === :pme
        # SPME mesh path (PMERecip.jl): GROMACS defaults fourier-spacing 0.12 nm, pme-order 4 unless SPCE_PME_SPACING / _ORDER
        sp_nm = parse(Float64, get(ENV, "SPCE_PME_SPACING", "0.12")); po = parse(Int, get(ENV, "SPCE_PME_ORDER", "4"))
        R = MCm.attach_pme!(sys, ff; rtol=1e-5, spacing=10sp_nm / m.σ_angstrom, order=po)
        println(io, "# SPME mesh path: α = $(round(α_A; digits=5)) 1/Å (ewald-rtol 1e-5 at rc = 10 Å), mesh $(R.K)³ (spacing ≤ $(sp_nm) nm), order $(po); exact Ewald image terms ($(MCm.n_kvectors(R)) k-vectors); no Widom insertion, no pressure"); flush(io)
    elseif coulomb === :ewald
        # reciprocal-space state from the minimised start; every MC / eKMC move after this keeps it current
        R = MCm.attach_ewald!(sys, ff; rtol=1e-5, backend=Symbol(get(ENV, "SPCE_EWALD_BACKEND", "cpu")))
        println(io, "# reciprocal-space Ewald: α = $(round(α_A; digits=5)) 1/Å (ewald-rtol 1e-5 at rc = 10 Å), $(MCm.n_kvectors(R)) k-vectors, backend $(get(ENV, "SPCE_EWALD_BACKEND", "cpu")); no Widom insertion (removal estimator only), no pressure (reciprocal virial not implemented)"); flush(io)
    end
    const_E = MCm.wolf_constant_energy(sys, ff)
    water_only = MCm.RigidSystem(m.templates, fill(1, N), L, 1 / T, Xoshiro(0))
    ΔU_tail = MCm.lrc_energy(sys, ff) - MCm.lrc_energy(water_only, ff)        # tail energy of the one solute
    println(io, "# solute LJ tail ΔU_tail = $(round(ΔU_tail * kJmol_per_eps; digits=4)) kJ/mol (added to every solute μ_ex below)")
    # per-molecule Coulomb constant of the model (RF: excluded-pair + self terms = −ke k_rf μ²; Wolf: self + intra).
    # For the GROMACS reaction-field decoupling ΔG this constant is part of −ΔG and must be added (printed as
    # mu_sol_rfconst); for Wolf vs PME it is a damping artefact and the pair-only value is the one to compare.
    E_const = MCm.coulomb_molecule_constant(m.solute, ff)
    println(io, "# solute Coulomb constant E_const = $(round(E_const * kJmol_per_eps; digits=4)) kJ/mol (model $(coulomb)); mu + E_const is the GROMACS-RF comparable value")

    δ = 0.1 / m.σ_angstrom * 3.0; δθ = 0.3
    if mc_sweeps > 0
        # --- Metropolis MC: equilibrate + production (Widom insertion and removal estimators of the solute) ---
        st = MCm.RigidMCStats()
        t0 = time()
        for s in 1:(mc_sweeps ÷ 4)
            for _ in 1:Nt
                MCm.mc_translate!(sys, ff, rand(sys.rng, 1:Nt), δ, st)
                MCm.mc_rotate!(sys, ff, rand(sys.rng, 1:Nt), δθ, st)
            end
            if flex
                for _ in 1:m.solute.n_sites
                    MCm.mc_site_move!(sys, ff, isol, 0.03, st)
                end
                MCm.mc_conformer_swap!(sys, ff, isol, res, st)
                if torsion_on
                    for _ in 1:length(MCm.rotatable_bonds(m.solute)); MCm.mc_torsion_move!(sys, ff, isol, δθ, st); end
                end
            end
        end
        println(io, "# MC equilibration $(mc_sweeps ÷ 4) sweeps: $(round(time() - t0; digits=1)) s, acc_t=$(st.trans_acc/st.trans_att), acc_r=$(st.rot_acc/st.rot_att)")
        flush(io)
        sys_mc = deepcopy(sys)
        t0 = time()
        mc = MCm.run_rigid_mc!(sys_mc, ff; sweeps=mc_sweeps, δ=δ, δθ=δθ, sample_every=2,
                               block_size=max(10, mc_sweeps ÷ 2 ÷ 25), widom_template=coulomb === :ewald ? 0 : 2, widom_per_sample=coulomb === :ewald ? 0 : 50,
                               inverse_widom_molecule=isol, R_every=20, reservoirs=reservoirs, δ_site=0.03,
                               torsion_φ=torsion_on ? δθ : 0.0)
        t_mc = time() - t0
        w, ew = MCm.mean(mc.widom), MCm.stderr(mc.widom)
        iw, eiw = MCm.mean(mc.inv_widom), MCm.stderr(mc.inv_widom)
        U_mc = (MCm.mean(mc.U) * Nt + const_E) / Nt * kJmol_per_eps
        @printf(io, "MC   sweeps=%d time=%.0fs  U/N=%.3f ± %.3f kJ/mol  P=%.1f ± %.1f atm  mu_sol_widom=%.2f ± %.2f kJ/mol  mu_sol_removal=%.2f ± %.2f kJ/mol  acc_t=%.3f acc_r=%.3f\n",
                mc_sweeps, t_mc, U_mc, MCm.stderr(mc.U) * kJmol_per_eps,
                MCm.mean(mc.P) * atm_per_pstar, MCm.stderr(mc.P) * atm_per_pstar,
                (-T * log(w) + ΔU_tail) * kJmol_per_eps, T * ew / w * kJmol_per_eps,
                (T * log(iw) + ΔU_tail) * kJmol_per_eps, T * eiw / iw * kJmol_per_eps,
                mc.stats.trans_acc / mc.stats.trans_att, mc.stats.rot_acc / mc.stats.rot_att)
        flex && @printf(io, "# MC flexible solute: <U_intra> = %.3f ± %.3f kJ/mol (ideal gas %.3f); site acc %.3f, conformer-swap acc %.3f, torsion acc %.3f\n",
                        MCm.mean(mc.U_intra) * kJmol_per_eps, MCm.stderr(mc.U_intra) * kJmol_per_eps, Ures * kJmol_per_eps,
                        mc.stats.site_acc / max(mc.stats.site_att, 1), mc.stats.swap_acc / max(mc.stats.swap_att, 1), mc.stats.tors_acc / max(mc.stats.tors_att, 1))
        println(io, "# MC removal-estimator block μ (kJ/mol): ", join(round.((T .* log.(mc.inv_widom.block_means) .+ ΔU_tail) .* kJmol_per_eps; digits=2), " "))
        println(io, "# MC Widom block μ (kJ/mol):             ", join(round.((-T .* log.(mc.widom.block_means) .+ ΔU_tail) .* kJmol_per_eps; digits=2), " "))
        flush(io)


    end
    # --- eKMC (MC-kernel hybrid) from the same equilibrated configuration ---
    t0 = time()
    λ_from = get(ENV, "SPCE_LAMBDA_FROM", "mc")
    share = parse(Float64, get(ENV, "SPCE_SOLUTE_SHARE", "0"))     # 0 = no rate scaling (default; equal-rate scaling fails for strongly bound solutes, JOURNAL entry 20)
    if nu_source == "mc"
        # λ = λfac × ⟨R⟩: ⟨R⟩ from the Metropolis production samples (relaxed-state relocation rate, robust) unless
        # SPCE_LAMBDA_FROM=pilot asks for the old 2e4-event pure-eKMC pilot (which can miss the relaxed states)
        R_mc = (λ_from == "mc" && !isempty(mc.R.block_means)) ? MCm.mean(mc.R) : NaN
        # species rate scaling: the solute gets the share SPCE_SOLUTE_SHARE of the relocation bursts (exact:
        # a constant factor on one species' rate keeps detailed balance, DESIGN §5c / JOURNAL entry 19)
        R_w = isfinite(R_mc) ? MCm.mean(mc.R_species[1]) : NaN
        R_s = isfinite(R_mc) ? MCm.mean(mc.R_species[2]) : NaN
        R_mc_err = isempty(mc.R.block_means) ? NaN : MCm.stderr(mc.R)
    else
        # E1 + E2: stock eKMC (solute fully coupled, no ladder) burn-in then pilot; ν etc. from the pilot's time averages
        tp = time()
        nb_in = mc_sweeps > 0 ? 0 : ekmc_burnin                    # after a Metropolis stage: pilot only
        # with SPCE_SOLVENT_LOCAL the burn-in and pilot also move every water locally (stock eKMC leaves a quench frozen,
        # JOURNAL 66/69); their ν = SPCE_SOLVENT_NU, default exp(β μ_ref) with the published SPC/E μ_ref = −29.5 kJ/mol as in
        # dev/validate_spce_melt.jl (ν only sets frequencies; production uses the pilot's ⟨R⟩ as for every other slot)
        νs_pilot = parse(Float64, get(ENV, "SPCE_SOLVENT_NU", string(exp(-29.5 / kJmol_per_eps / T))))
        rp = MCm.init_rigid_ekmc_run(sys, ff; events=nb_in + nu_pilot, burnin_events=nb_in, sample_every=100,
                                     block_events=max(25, nu_pilot ÷ 25), species_mu=true, reservoirs=reservoirs,
                                     local_solvent=solvent_local, solvent_nu=solvent_local ? νs_pilot : NaN)
        dev_attach!(sys, rp, ff); MCm.advance_rigid_ekmc!(rp, sys, ff); dev_detach!(sys, rp)
        rpr = MCm.rigid_ekmc_results(rp)
        R_w = MCm.ratio_mean(rpr.RN_species[1]); R_s = MCm.ratio_mean(rpr.RN_species[2])
        R_mc = R_w * N / Nt + R_s / Nt                               # per-molecule ⟨R/N⟩ as ⟨R⟩_MC
        R_mc_err = R_w * MCm.ratio_jackknife_stderr(rpr.RN_species[1])
        mc_sweeps > 0 || (iw = R_s)                                  # η-ramp guess: MC removal estimate if MC ran, else the stock eKMC one
        @printf(io, "# eKMC setup (DESIGN §11.3; MC stage %s): eKMC burn-in %d events + pilot %d events, %.0f s; pilot U/N=%.3f kJ/mol  <R_water>=%.4e  <exp(βu_solute)>=%.4e (mu_sol_removal=%.2f kJ/mol)\n",
                mc_sweeps > 0 ? "ran, start configuration only" : (solvent_local ? "off; solvent L-moves on" : "off"), nb_in, nu_pilot, time() - tp, (MCm.mean(rpr.U) * Nt + const_E) / Nt * kJmol_per_eps, R_w, R_s,
                (T * log(R_s) + ΔU_tail) * kJmol_per_eps)
        flush(io)
    end
    c_sol = (share > 0 && isfinite(R_w) && isfinite(R_s) && R_s > 0) ? share / (1 - share) * R_w / R_s : 1.0
    R_eff = isfinite(R_mc) ? R_w + c_sol * R_s : NaN
    λ = λfac > 0 ? λfac * (isfinite(R_eff) ? R_eff : MCm.estimate_relocation_rate(deepcopy(sys), ff; events=20_000)) : 0.0
    println(io, "# species rate scale: <R_water>_MC=$(R_w)  <R_solute>_MC=$(R_s)  c_solute=$(c_sol)  (solute share $(share))"); flush(io)
    println(io, "# eKMC fast_path=$(fast_path); MC-kernel hybrid: SPCE_LAMBDA=$(λfac), λ=$(λ) (⟨R⟩_MC=$(R_mc) ± $(R_mc_err), source=$(nu_source == "mc" ? (isfinite(R_mc) ? "mc" : "pilot") : "eKMC pilot"))"); flush(io)
    se = λfac > 0 ? 1 : 100
    r = MCm.init_rigid_ekmc_run(sys, ff; events=ekmc_events, overlap_fast_path=fast_path, sample_every=se, block_events=max(25, ekmc_events ÷ 25),
                                debug_check_every=ekmc_events ÷ 10, mc_kernel_rate=λ, mc_δ=δ, mc_δθ=δθ,
                                species_mu=true, trace_molecule=trace ? isol : 0, species_rate_scale=[1.0, c_sol], reservoirs=reservoirs,
                                bar_species=bar_on ? 2 : 0, ladder_solute=ladder_on ? isol : 0, ladder_lambda=ladder_λ,
                                ladder_nu=ladder_on ? ladder_νfac * R_mc : NaN, ladder_wl_events=ladder_on ? round(Int, ladder_wlfrac * ekmc_events) : 0,
                                ladder_eta_ramp=ladder_on ? (isnan(ladder_ramp_kJ) ? log(iw) : ladder_ramp_kJ / kJmol_per_eps / T) : 0.0,   # βμ guess: SPCE_LADDER_RAMP or the MC removal estimator
                                ladder_adapt_blocks=ladder_adapt, mc_local_radius=mc_local_A / m.σ_angstrom, ladder_jump=ladder_jump,
                                ladder_jump_max=jump_max, ladder_local=local_on, local_internal=internal_on, local_solvent=solvent_local,
                                solvent_nu=solvent_local ? R_mc : NaN,   # water slots at the unscaled ⟨R⟩ even when SPCE_LADDER_NU boosts the solute
                                ladder_eta=ladder_eta_fixed, track_dihedrals=(track_dih && flex) ? solute_dihedrals(m.solute) : NTuple{4,Int}[],
                                local_φ=parse(Float64, get(ENV, "SPCE_LOCAL_PHI", "NaN")))   # solute torsion half-angle (rad); NaN = mc_δθ
    ladder_eta_fixed === nothing || println(io, "# ladder weights FIXED from SPCE_LADDER_ETA: ", ladder_eta_fixed)
    ladder_on && println(io, "# coupling ladder: λ = $(ladder_λ), ν = $(ladder_νfac)·⟨R⟩_MC = $(ladder_νfac * R_mc), Wang–Landau over the first $(round(Int, ladder_wlfrac * ekmc_events)) events; λ=0 hub $(ladder_jump); U, P, μ_water, mu_sol read at the coupled rung only")
    local_on && println(io, "# L-moves (DESIGN §11.2): slots $(r.lmoves.kinds) (1 T, 2 R, 3 C, 4 site, 5 torsion), a = $(round(r.lmoves.a * m.σ_angstrom; digits=3)) Å, θ_max = $(r.lmoves.θmax) rad, site a = $(round(r.lmoves.site_a * m.σ_angstrom; digits=3)) Å, φ_max = $(r.lmoves.φmax) rad, ν_L = ν_D = $(r.lmoves.ν); hub limited to rungs ≤ $(r.ladder.jump_max)")
    solvent_local && println(io, "# solvent L-moves: $(length(r.slmoves.mols)) movers × slots $(r.slmoves.kinds), ν = ν_D = $(r.slmoves.ν)")
    mc_local_A > 0 && println(io, "# MC kernel: local sweeps within $(mc_local_A) Å of the solute (rate λ = $(λfac)·⟨R⟩_MC)")
    trace || println(io, "# solute trace off (SPCE_TRACE=0): no tail diagnostics; the trace costs 16 bytes per event")
    ctx = (; out, T, kJmol_per_eps, atm_per_pstar, ΔU_tail, E_const, const_E, Nt, flex, Ures=flex ? Ures : NaN, coulomb, ekmc_events, t_ek=0.0)
    return (sys, r, ff, ctx)
end

# --- eKMC: run (or continue from the checkpoint) and report ---
function ekmc_report(io, ek, ctx)
    (; T, kJmol_per_eps, atm_per_pstar, ΔU_tail, E_const, const_E, Nt, flex, Ures, coulomb, ekmc_events, t_ek) = ctx
    rw, rs = ek.RN_species[1], ek.RN_species[2]
    μ_w = T * log(MCm.ratio_mean(rw)); eμ_w = T * MCm.ratio_jackknife_stderr(rw; f=log)
    μ_s = T * log(MCm.ratio_mean(rs)) + ΔU_tail; eμ_s = T * MCm.ratio_jackknife_stderr(rs; f=log)
    U_ek = (MCm.mean(ek.U) * Nt + const_E) / Nt * kJmol_per_eps
    @printf(io, "eKMC events=%d time=%.0fs  U/N=%.3f ± %.3f kJ/mol  P=%.1f ± %.1f atm  mu_water=%.2f ± %.2f kJ/mol  mu_sol=%.2f ± %.2f kJ/mol (jackknife, %d blocks)  mu_mix_acc=%.2f  <dt>=%.3e  mc_acc=%.3f  relocations=%d bursts=%d fast_events=%d\n",
            ekmc_events, t_ek, U_ek, MCm.stderr(ek.U) * kJmol_per_eps,
            MCm.mean(ek.P) * atm_per_pstar, MCm.stderr(ek.P) * atm_per_pstar,
            μ_w * kJmol_per_eps, eμ_w * kJmol_per_eps, μ_s * kJmol_per_eps, eμ_s * kJmol_per_eps,
            length(rs.block_sum_x), MCm.mu_ex(ek.mu, T) * kJmol_per_eps, ek.t_total / ekmc_events, ek.mc_acceptance,
            ek.relocations, ek.bursts, ek.fast_events)
    @printf(io, "# eKMC mu_sol + E_const = %.2f ± %.2f kJ/mol (GROMACS-RF comparable; model %s)\n", (μ_s + E_const) * kJmol_per_eps, eμ_s * kJmol_per_eps, String(coulomb))
    flex && @printf(io, "# eKMC flexible solute: <U_intra> = %.3f ± %.3f kJ/mol (time-weighted; ideal gas %.3f)\n",
                    MCm.mean(ek.U_intra) * kJmol_per_eps, MCm.stderr(ek.U_intra) * kJmol_per_eps, Ures * kJmol_per_eps)
    if ek.bar !== nothing
        corr = (ΔU_tail + E_const) * kJmol_per_eps
        lw, elw = MCm.landing_widom_mu(ek.bar, T); bm, ebm = MCm.bennett_mu(ek.bar, T); rm, erm = MCm.removal_mu(ek.bar, T)
        d = MCm.bennett_diagnostics(ek.bar)
        @printf(io, "# eKMC Bennett (GROMACS-RF comparable, + E_const + tail): landing-Widom = %.2f ± %.2f  BAR = %.2f ± %.2f  removal(check) = %.2f ± %.2f kJ/mol; landings=%d  n_eff_time=%.1f  overlap=%.4f\n",
                lw * kJmol_per_eps + corr, elw * kJmol_per_eps, bm * kJmol_per_eps + corr, ebm * kJmol_per_eps, rm * kJmol_per_eps + corr, erm * kJmol_per_eps,
                d.n_landings, d.n_eff_time, d.overlap)
        nb = MCm.nblocks(ek.bar)
        println(io, "# eKMC Bennett per-block landings: ", join(ek.bar.land_n, " "))
        println(io, "# eKMC Bennett landing-Widom per block (kJ/mol): ",
                join([ek.bar.land_wsum[k] > 0 ? round(-T * log(ek.bar.land_wsum[k] / ek.bar.land_n[k]) * kJmol_per_eps + corr; digits=1) : NaN for k in 1:nb], " "))
    end
    if ek.ladder !== nothing
        corr = (ΔU_tail + E_const) * kJmol_per_eps
        sm = MCm.ladder_summary(ek.ladder)
        oc, eoc = MCm.ladder_mu_occupancy(ek.ladder, T); mb, emb = MCm.ladder_mu_mbar(ek.ladder, T)
        @printf(io, "# eKMC ladder (GROMACS-RF comparable, + E_const + tail): occupancy = %.2f ± %.2f  MBAR = %.2f ± %.2f kJ/mol; recouplings=%d decouplings=%d  WL stages=%d frozen at event %d (f=%.1e)\n",
                oc * kJmol_per_eps + corr, eoc * kJmol_per_eps, mb * kJmol_per_eps + corr, emb * kJmol_per_eps,
                sm.n_recouple, sm.n_decouple, sm.wl_stages, sm.ev_frozen, sm.wl_f)
        println(io, "# eKMC ladder time fraction per rung: ", join(round.(sm.frac_rung; digits=4), " "), "  η = ", join(round.(sm.η; digits=3), " "),
                "  hops per rung (arrivals) = ", join(sm.n_hop, " "), "  hub jumps (arrivals) = ", join(sm.n_jump, " "))
        println(io, "# eKMC ladder D0: longest rung residence / rung time = ", join(round.(sm.res_frac; digits=4), " "),
                "  longest pose residence / rung time = ", join(round.(sm.pose_frac; digits=4), " "))
        ek.lmoves === nothing || println(io, "# eKMC L-moves fired per kind (T R C S D) = ", join(ek.lmoves.n_move, " "), "  per rung (production) = ",
                                        join(ek.lmoves.n_move_rung, " "), "  refreshes = ", ek.lmoves.n_refresh, "  candidate rebuilds = ", ek.lmoves.n_rebuild)
        nb = MCm.nblocks(ek.ladder)
        println(io, "# eKMC ladder block occupancy μ (kJ/mol): ", join([(t = ek.ladder.t_rung[b]; (t[1] > 0 && t[end] > 0) ?
                round((T * ((ek.ladder.η[end] - ek.ladder.η[1]) - log(t[end] / t[1]))) * kJmol_per_eps + corr; digits=2) : NaN) for b in 1:nb], " "))
    end
    println(io, "# eKMC solute block μ (kJ/mol): ", join(round.((T .* log.(rs.block_means) .+ ΔU_tail) .* kJmol_per_eps; digits=2), " "))
    println(io, "# eKMC water block μ (kJ/mol):  ", join(round.(T .* log.(rw.block_means) .* kJmol_per_eps; digits=2), " "))
    println(io, "# eKMC block times: ", join(round.(rs.block_sum_t; sigdigits=3), " "))
    tail_report(io, ek.trace_phi, ek.trace_dt, 1 / T, "eKMC solute")
    println(io, "# block means U/N (eKMC, reduced): ", join(round.(ek.U.block_means; digits=3), " "))
end

# one dihedral per rotatable bond j–k, ends = the heaviest other neighbour of j and of k (propanol: C–C–C–O, C–C–O–H)
function solute_dihedrals(t)
    adj = [Int[] for _ in 1:t.n_sites]
    for b in t.bonded.bonds; push!(adj[b.i], b.j); push!(adj[b.j], b.i); end
    out = NTuple{4,Int}[]
    for (j, k, _) in MCm.rotatable_bonds(t)
        ni = filter(!=(k), adj[j]); nl = filter(!=(j), adj[k])
        (isempty(ni) || isempty(nl)) && continue
        push!(out, (ni[argmax(t.site_mass[ni])], j, k, nl[argmax(t.site_mass[nl])]))
    end
    return out
end
# per-rung trans (|φ| > 120°) / gauche+ / gauche− fractions of the tracked dihedrals (time-weighted, production)
function dihedral_report(io, r)
    isempty(r.dih_atoms) && return
    for (d, a) in enumerate(r.dih_atoms)
        parts = String[]
        for k in eachindex(r.dih_hist)
            h = r.dih_hist[k][d]; tot = sum(h); tot > 0 || (push!(parts, "-"); continue)
            c = [(-π + (b - 0.5) * 2π / MCm.DIH_BINS) for b in 1:MCm.DIH_BINS]
            tr = sum(h[b] for b in eachindex(h) if abs(c[b]) > 2π / 3) / tot; gp = sum(h[b] for b in eachindex(h) if 0 < c[b] <= 2π / 3) / tot
            push!(parts, @sprintf("%.3f/%.3f/%.3f", tr, gp, 1 - tr - gp))
        end
        println(io, "# dihedral $(a) trans/g+/g− per rung: ", join(parts, "  "))
    end
end

# atomic checkpoint write: a job killed while writing leaves the previous checkpoint intact
function save_checkpoint(path, state)
    tmp = path * ".tmp"
    open(io -> serialize(io, state), tmp, "w")
    mv(tmp, path; force=true)
end

if !isempty(ckpt) && isfile(ckpt * ".final")
    println("run already complete (", ckpt, ".final exists); nothing to do"); exit(0)
end
if !isempty(ckpt) && isfile(ckpt)
    sys, r, ff, ctx = deserialize(ckpt)
    io = open(ctx.out, "a")
    println(io, "# resumed from checkpoint at event $(r.ev) of $(r.events)  $(Dates.now())  host=$(gethostname())"); flush(io)
else
    sys, r, ff, ctx = open(setup_and_mc, out, "w")
    isempty(ckpt) || save_checkpoint(ckpt, (sys, r, ff, ctx))
    io = open(ctx.out, "a")
end
dev_attach!(sys, r, ff)
SPCE_DEVICE === :none || println(io, "# device $(SPCE_DEVICE): pair rows, solvent-candidate bank$(sys.recip isa MCm.PMERecip ? ", PME mesh gathers" : "") ($(Dates.now()), host $(gethostname()))")
# eKMC in slices of SPCE_CKPT_EVERY seconds with a checkpoint after each, so a requeued or killed job loses at most
# one slice (Slurm requeued 205893_1 twice, JOURNAL entry 32); the slices do not change the trajectory (test_ekmc_checkpoint.jl)
t_start = t_script
while r.ev < r.events
    left = wall - (time() - t_start)
    left > 0 || break
    t0 = time()
    MCm.advance_rigid_ekmc!(r, sys, ff; wall_seconds=isempty(ckpt) ? left : min(left, ckpt_every))
    global ctx = merge(ctx, (; t_ek=ctx.t_ek + time() - t0))
    if r.ev < r.events && !isempty(ckpt)
        dev_detach!(sys, r); save_checkpoint(ckpt, (sys, r, ff, ctx)); dev_attach!(sys, r, ff)
        println(io, "# checkpoint at event $(r.ev) of $(r.events)  $(Dates.now())  eKMC wall time so far $(round(ctx.t_ek; digits=0)) s"); flush(io)
    end
end
if r.ev < r.events
    isempty(ckpt) && error("eKMC stopped at event $(r.ev) of $(r.events) without a checkpoint path (set SPCE_CHECKPOINT)")
    close(io)
    println("checkpoint written at event $(r.ev) of $(r.events)"); exit(0)
end
ekmc_report(io, MCm.rigid_ekmc_results(r), ctx)
dihedral_report(io, r)
close(io)
isempty(ckpt) || (dev_detach!(sys, r); save_checkpoint(ckpt, (sys, r, ff, ctx)); mv(ckpt, ckpt * ".final"; force=true))
println(read(ctx.out, String))

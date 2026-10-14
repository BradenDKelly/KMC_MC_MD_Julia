"""
V1 of DESIGN §10.6: a strongly bound LJ solute (one site, ε_s = EPS_S ε, σ_s = σ) in N LJ atoms at ρ* = 0.85, T* = 0.9
(the ljbias state point). Three eKMC estimators of the solute μ_ex from ONE stock run — removal (time-weighted
⟨e^{βu}⟩, the species estimator), landing-Widom (jump-chain identity) and Bennett — against an independent Widom
reference: Metropolis MC of the pure solvent with many test insertions. The hard case (ε_s = 9) is the LJ
analogue of methanol in water: the solute is rarely chosen as the mover.

Usage: julia --project=. dev/bennett_lj.jl EPS_S N EVENTS MC_SWEEPS SEED OUTDIR
"""

using MolSim, Random, Printf, Dates
const MCm = MolSim.MC
ε_s = parse(Float64, ARGS[1]); N = parse(Int, ARGS[2]); events = parse(Int, ARGS[3]); sweeps = parse(Int, ARGS[4])
seed = parse(Int, ARGS[5]); outdir = ARGS[6]; mkpath(outdir)
ρ = 0.85; T = 0.9; β = 1 / T; L = cbrt(N / ρ); Nt = N + 1
site = zeros(3, 1)
solv = MCm.RigidTemplate("LJ", site, [1], [0.0], [1.0]); solu = MCm.RigidTemplate("S", site, [2], [0.0], [1.0])
ff = MCm.SiteForceField([1.0, 1.0], [1.0, ε_s]; rc=2.5, coulomb=:none, mixing=:geometric)
ff = first(MCm.kT_overlap_cap(ff, [solv, solu], T; U_kT=100.0))
out = joinpath(outdir, "bennett_lj_eps$(ε_s)_N$(N)_seed$(seed)$(get(ENV, "LADDER", "0") == "1" ? "_ladder" : "").txt")
open(out, "w") do io
    println(io, "# Bennett V1  $(Dates.now())  ε_s=$ε_s N=$N ρ*=$ρ T*=$T events=$events mc_sweeps=$sweeps seed=$seed")
    # Widom reference into the pure solvent
    sys0 = MCm.RigidSystem([solv, solu], fill(1, N), L, β, Xoshiro(seed)); MCm.random_rigid_system!(sys0, ff); MCm.minimize_rigid!(sys0, ff; nsteps=500)
    st = MCm.RigidMCStats()
    for _ in 1:(sweeps ÷ 4), _ in 1:N
        MCm.mc_translate!(sys0, ff, rand(sys0.rng, 1:N), 0.12, st)
    end
    t0 = time()
    mc0 = MCm.run_rigid_mc!(sys0, ff; sweeps=sweeps, δ=0.12, δθ=0.0, sample_every=2, block_size=max(10, sweeps ÷ 50), widom_template=2, widom_per_sample=200)
    w0, ew0 = MCm.mean(mc0.widom), MCm.stderr(mc0.widom)
    @printf(io, "MC Widom reference (pure solvent, %d insertions): mu = %.4f ± %.4f  U/N = %.4f  time %.0f s\n",
            sweeps ÷ 2 * 200, -T * log(w0), T * ew0 / w0, MCm.mean(mc0.U), time() - t0)
    # solvent + solute, stock eKMC with the Bennett accumulator
    sys = MCm.RigidSystem([solv, solu], vcat(fill(1, N), 2), L, β, Xoshiro(seed + 1)); MCm.random_rigid_system!(sys, ff); MCm.minimize_rigid!(sys, ff; nsteps=500)
    st = MCm.RigidMCStats()
    for _ in 1:(sweeps ÷ 4), _ in 1:Nt
        MCm.mc_translate!(sys, ff, rand(sys.rng, 1:Nt), 0.12, st)
    end
    sys_l = deepcopy(sys)                                        # same equilibrated start for the ladder run
    t0 = time()
    ek = MCm.run_rigid_ekmc!(sys, ff; events=events, sample_every=10, block_events=events ÷ 25, species_mu=true, bar_species=2)
    b = ek.bar; d = MCm.bennett_diagnostics(b)
    r2 = ek.RN_species[2]
    @printf(io, "eKMC %d events, %.0f s: removal = %.4f ± %.4f  landing-Widom = %.4f ± %.4f  BAR = %.4f ± %.4f  BAR(M=0) = %.4f  U/N = %.4f\n",
            events, time() - t0, T * log(MCm.ratio_mean(r2)), T * MCm.ratio_jackknife_stderr(r2; f=log),
            MCm.landing_widom_mu(b, T)..., MCm.bennett_mu(b, T)..., MCm.bennett_mu(b, T; M=0.0)[1], MCm.mean(ek.U))
    @printf(io, "# landings=%d (%.3e per event)  n_eff_time=%.1f  overlap=%.5f  solute events per block: %s\n",
            d.n_landings, d.n_landings / events, d.n_eff_time, d.overlap, join(b.land_n, " "))
    println(io, "# removal block μ: ", join(round.(T .* log.(r2.block_means); digits=2), " "))
    println(io, "# landing-Widom block μ: ", join([b.land_wsum[k] > 0 ? round(-T * log(b.land_wsum[k] / b.land_n[k]); digits=2) : NaN for k in 1:MCm.nblocks(b)], " "))
    # coupling ladder (DESIGN §10.4) from the same start: LADDER=1, rungs LADDER_LAMBDA, ν = LADDER_NU·R_pilot, WL on LADDER_WL of the events
    if get(ENV, "LADDER", "0") == "1"
        λs = parse.(Float64, split(get(ENV, "LADDER_LAMBDA", "0,0.01,0.03,0.1,0.2,0.35,0.5,0.7,0.85,1"), ","))
        R_pilot = MCm.estimate_relocation_rate(deepcopy(sys_l), ff; events=20_000)
        ν = parse(Float64, get(ENV, "LADDER_NU", "1.0")) * R_pilot
        wl = round(Int, parse(Float64, get(ENV, "LADDER_WL", "0.1")) * events)
        t0 = time()
        ramp = T * log(MCm.ratio_mean(r2)) / T                # βμ guess from the stock removal estimate (η_k = λ_k·βμ; WL refines)
        lad = MCm.run_rigid_ekmc!(sys_l, ff; events=events, sample_every=10, block_events=events ÷ 25, species_mu=true,
                                  ladder_solute=Nt, ladder_nu=ν, ladder_lambda=λs, ladder_wl_events=wl, ladder_eta_ramp=ramp)
        Ld = lad.ladder; sm = MCm.ladder_summary(Ld)
        oc, eoc = MCm.ladder_mu_occupancy(Ld, T); mb, emb = MCm.ladder_mu_mbar(Ld, T)
        rK = lad.RN_species[2]
        @printf(io, "ladder %d events, %.0f s: occupancy = %.4f ± %.4f  MBAR = %.4f ± %.4f  removal@coupled = %.4f ± %.4f  U/N@coupled = %.4f  ν=%.3e λ=%s\n",
                events, time() - t0, oc, eoc, mb, emb, T * log(MCm.ratio_mean(rK)), T * MCm.ratio_jackknife_stderr(rK; f=log), MCm.mean(lad.U), ν, string(λs))
        @printf(io, "# ladder: recouplings=%d decouplings=%d  time fraction per rung %s  η %s  WL stages %d frozen at %d\n", sm.n_recouple, sm.n_decouple,
                string(round.(sm.frac_rung; digits=4)), string(round.(sm.η; digits=3)), sm.wl_stages, sm.ev_frozen)
        println(io, "# ladder block occupancy μ: ", join([(t = Ld.t_rung[k]; (t[1] > 0 && t[end] > 0) ? round(T * ((Ld.η[end] - Ld.η[1]) - log(t[end] / t[1])); digits=2) : NaN) for k in 1:MCm.nblocks(Ld)], " "))
    end
end
println(read(out, String))

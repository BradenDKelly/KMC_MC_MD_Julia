"""
SPC/E liquid water with Wolf electrostatics: Metropolis MC vs eKMC vs NIST reference data.

State point: T = 300 K, ρ = 1000 kg/m³ (NIST LAMMPS SPC/E equation-of-state page, N = 1500, Ewald,
r_c = 10 Å + LRC: U = −1.6744E+04 ± 1.8 kcal/mol for 1500 molecules = −11.163 kcal/mol = −46.70 kJ/mol
per molecule, p = 43.2 ± 5.1 atm). Here: Wolf α = 0.16 Å⁻¹, r_c = 10 Å (dev/wolf_parameter_scan.jl),
N as given on the command line (default 500 → L = 24.6 Å).

Usage: julia --project=. dev/validate_spce_wolf.jl [N] [mc_sweeps] [ekmc_events] [seed] [outdir]
Writes <outdir>/spce_wolf_N<N>_seed<seed>.txt with U/N (kJ/mol, with the Wolf constants), P (atm),
μ_ex (kJ/mol) from Widom and from ⟨R/N⟩, block standard errors, and timings.
"""

using MolSim, Random, StaticArrays, Printf, Dates
const MCm = MolSim.MC

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500
mc_sweeps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4000
ekmc_events = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20_000_000
seed = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
outdir = length(ARGS) >= 5 ? ARGS[5] : joinpath(@__DIR__, "results")
mkpath(outdir)
# force-field variant from the environment (defaults = the site-cutoff α = 0.16 Å⁻¹ model of the first runs)
α_A = parse(Float64, get(ENV, "SPCE_ALPHA", "0.16"))
cutoff = Symbol(get(ENV, "SPCE_CUTOFF", "site"))
coulomb = Symbol(get(ENV, "SPCE_COULOMB", "wolf"))
fast_path = get(ENV, "SPCE_FAST_PATH", "0") == "1"            # overlap fast path switch (approximate for weakly bound species; default off)          # wolf | rf (GROMACS reaction field, SPCE_EPS_RF: 0 = ∞)
ε_rf = (x = parse(Float64, get(ENV, "SPCE_EPS_RF", "0")); x == 0 ? Inf : x)
elec_tag = coulomb === :rf ? "rf$(isinf(ε_rf) ? "inf" : ε_rf)" : "a$(α_A)"

m = MCm.spce_model(; α_per_angstrom=α_A, rc_angstrom=10.0, cutoff=cutoff, coulomb=coulomb, epsilon_rf=ε_rf)
tag = get(ENV, "SPCE_TAG", "")          # appended to the file name (e.g. "_ev1e5") so reruns do not overwrite
out = joinpath(outdir, "spce_wolf_N$(N)_seed$(seed)_$(cutoff)_$(elec_tag)_cap$(Int(m.ff.ucap_mix[1,1]))$(tag).txt")
ff = m.ff
T_K = 300.0
T = T_K / m.ϵ_kelvin
ρ_A3 = 1000.0 * 1e3 / 18.0153 * 6.02214129e23 / 1e30                       # molecules per Å³ at 1000 kg/m³
ρstar = ρ_A3 * m.σ_angstrom^3
L = cbrt(N / ρstar)
L_A = L * m.σ_angstrom
# unit conversions
kJmol_per_eps = m.ϵ_kelvin * 1.3806488e-23 * 6.02214129e23 / 1000
atm_per_pstar = m.ϵ_kelvin * 1.3806488e-23 / (m.σ_angstrom * 1e-10)^3 / 101325.0

open(out, "w") do io
    println(io, "# SPC/E Wolf validation  $(Dates.now())  N=$N  L=$(round(L_A; digits=3)) Å  T=$T_K K  ρ=1000 kg/m³")
    println(io, "# Wolf α=$(α_A) 1/Å rc=10 Å cutoff=$(cutoff) (reduced α*=$(ff.wolf.α), rc*=$(ff.rc)); LRC on; overlap cap on")
    println(io, "# NIST LAMMPS reference (Ewald, N=1500): U/N = -46.70 kJ/mol, p = 43.2 ± 5.1 atm")
    flush(io)

    rc_check = ff.rc + 2 * 1.0 / m.σ_angstrom   # cutoff + molecular extent must be < L/2
    rc_check < L / 2 || println(io, "# WARNING: rc + molecule extent = $(rc_check) ≥ L/2 = $(L/2)")

    sys = MCm.RigidSystem([m.template], fill(1, N), L, 1 / T, Xoshiro(seed))
    MCm.random_rigid_system!(sys, ff)
    mres = MCm.minimize_rigid!(sys, ff; nsteps=500)   # steepest descent before MC (JOURNAL entry 25)
    println(io, "# minimize_rigid!: $(mres.steps) steps, U_pair $(round(mres.U_start; sigdigits=6)) → $(round(mres.U_end; sigdigits=6)) (reduced), capped pairs left $(mres.n_capped)"); flush(io)
    const_E = MCm.wolf_constant_energy(sys, ff)

    # --- Metropolis MC: equilibrate + production ---
    st = MCm.RigidMCStats()
    δ = 0.1 / m.σ_angstrom * 3.0; δθ = 0.3        # ~0.3 Å, ~17°
    t0 = time()
    for s in 1:(mc_sweeps ÷ 4), _ in 1:N
        MCm.mc_translate!(sys, ff, rand(sys.rng, 1:N), δ, st)
        MCm.mc_rotate!(sys, ff, rand(sys.rng, 1:N), δθ, st)
    end
    println(io, "# MC equilibration $(mc_sweeps ÷ 4) sweeps: $(round(time() - t0; digits=1)) s, acc_t=$(st.trans_acc/st.trans_att), acc_r=$(st.rot_acc/st.rot_att)")
    flush(io)
    sys_mc = deepcopy(sys)
    t0 = time()
    mc = MCm.run_rigid_mc!(sys_mc, ff; sweeps=mc_sweeps, δ=δ, δθ=δθ, sample_every=2,
                           block_size=max(10, mc_sweeps ÷ 2 ÷ 40), widom_template=1, widom_per_sample=20, R_every=20)
    t_mc = time() - t0
    w, ew = MCm.mean(mc.widom), MCm.stderr(mc.widom)
    U_mc = (MCm.mean(mc.U) + const_E / N) * kJmol_per_eps
    eU_mc = MCm.stderr(mc.U) * kJmol_per_eps
    @printf(io, "MC   sweeps=%d time=%.0fs  U/N=%.3f ± %.3f kJ/mol  P=%.1f ± %.1f atm  mu_widom=%.2f ± %.2f kJ/mol  acc_t=%.3f acc_r=%.3f\n",
            mc_sweeps, t_mc, U_mc, eU_mc, MCm.mean(mc.P) * atm_per_pstar, MCm.stderr(mc.P) * atm_per_pstar,
            -T * log(w) * kJmol_per_eps, T * ew / w * kJmol_per_eps,
            mc.stats.trans_acc / mc.stats.trans_att, mc.stats.rot_acc / mc.stats.rot_att)
    flush(io)

    # --- eKMC from the same equilibrated configuration ---
    t0 = time()
    # MC-kernel hybrid (λ = SPCE_LAMBDA × ⟨R⟩ from a pilot; 0 = pure eKMC). With λ > 0 most events are
    # Metropolis sweeps (2N trials), so EKMC_EVENTS should be ~1e4–1e5, not 1e7.
    λfac = parse(Float64, get(ENV, "SPCE_LAMBDA", "0"))   # 0 = stock (pure) eKMC; the Slurm wrappers pass 100 for the hybrid
    # λ = λfac × ⟨R⟩: ⟨R⟩ from the Metropolis production samples (relaxed-state relocation rate, robust) unless
    # SPCE_LAMBDA_FROM=pilot asks for the old 2e4-event pure-eKMC pilot (which can miss the relaxed states)
    λ_from = get(ENV, "SPCE_LAMBDA_FROM", "mc")
    R_mc = (λ_from == "mc" && !isempty(mc.R.block_means)) ? MCm.mean(mc.R) : NaN
    λ = λfac > 0 ? λfac * (isfinite(R_mc) ? R_mc : MCm.estimate_relocation_rate(deepcopy(sys), ff; events=20_000)) : 0.0
    println(io, "# eKMC fast_path=$(fast_path); MC-kernel hybrid: SPCE_LAMBDA=$(λfac), λ=$(λ) (⟨R⟩_MC=$(R_mc) ± $(isempty(mc.R.block_means) ? NaN : MCm.stderr(mc.R)), source=$(isfinite(R_mc) ? "mc" : "pilot"))"); flush(io)
    se = λfac > 0 ? 1 : 100
    ek = MCm.run_rigid_ekmc!(sys, ff; events=ekmc_events, overlap_fast_path=fast_path, sample_every=se, block_events=max(25, ekmc_events ÷ 25),
                             debug_check_every=ekmc_events ÷ 10, mc_kernel_rate=λ, mc_δ=δ, mc_δθ=δθ)
    t_ek = time() - t0
    rn, ern = MCm.mean(ek.RN), MCm.stderr(ek.RN)
    U_ek = (MCm.mean(ek.U) + const_E / N) * kJmol_per_eps
    eU_ek = MCm.stderr(ek.U) * kJmol_per_eps
    # mu_RN: mean of block ratios (biased when block times differ); mu_acc: global time average (the
    # estimator); mu_jk: the same global ratio with its delete-one-block jackknife error (Efron 1982)
    @printf(io, "eKMC events=%d time=%.0fs  U/N=%.3f ± %.3f kJ/mol  P=%.1f ± %.1f atm  mu_RN=%.2f ± %.2f kJ/mol  mu_acc=%.2f kJ/mol  mu_jk=%.2f ± %.2f kJ/mol (%d blocks)  <dt>=%.3e  relocations=%d bursts=%d fast_events=%d\n",
            ekmc_events, t_ek, U_ek, eU_ek, MCm.mean(ek.P) * atm_per_pstar, MCm.stderr(ek.P) * atm_per_pstar,
            T * log(rn) * kJmol_per_eps, T * ern / rn * kJmol_per_eps, MCm.mu_ex(ek.mu, T) * kJmol_per_eps,
            T * log(MCm.ratio_mean(ek.RN)) * kJmol_per_eps, T * MCm.ratio_jackknife_stderr(ek.RN; f=log) * kJmol_per_eps,
            length(ek.RN.block_sum_x), ek.t_total / ekmc_events, ek.relocations, ek.bursts, ek.fast_events)
    println(io, "# eKMC block μ (global-ratio blocks, kJ/mol): ", join(round.(T .* log.(ek.RN.block_means) .* kJmol_per_eps; digits=2), " "))
    println(io, "# eKMC block times: ", join(round.(ek.RN.block_sum_t; sigdigits=3), " "))
    println(io, "# block means U/N (eKMC, reduced): ", join(round.(ek.U.block_means; digits=3), " "))
    println(io, "# block means U/N (MC, reduced):   ", join(round.(mc.U.block_means; digits=3), " "))
end
println(read(out, String))

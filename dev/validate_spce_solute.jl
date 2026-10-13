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
Env: SPCE_ALPHA (0.20), SPCE_CUTOFF (molecule), SPCE_COULOMB (wolf|rf), SPCE_LAMBDA (100; λ = fac·⟨R⟩_MC, 0 = pure
eKMC), SPCE_SOLUTE_SHARE (0.5; share of relocation bursts given to the solute by rate scaling), SPCE_TAG.
"""

using MolSim, Random, StaticArrays, Printf, Dates
const MCm = MolSim.MC

solute = Symbol(length(ARGS) >= 1 ? ARGS[1] : "methane")
N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
mc_sweeps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8000
ekmc_events = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 100_000
seed = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1
outdir = length(ARGS) >= 6 ? ARGS[6] : joinpath(@__DIR__, "results", "spce_solute")
mkpath(outdir)
α_A = parse(Float64, get(ENV, "SPCE_ALPHA", "0.20"))
cutoff = Symbol(get(ENV, "SPCE_CUTOFF", "molecule"))
coulomb = Symbol(get(ENV, "SPCE_COULOMB", "wolf"))
fast_path = get(ENV, "SPCE_FAST_PATH", "0") == "1"            # overlap fast path switch (approximate for weakly bound species; default off)          # wolf | rf (GROMACS reaction field, SPCE_EPS_RF: 0 = ∞)
ε_rf = (x = parse(Float64, get(ENV, "SPCE_EPS_RF", "0")); x == 0 ? Inf : x)
elec_tag = coulomb === :rf ? "rf$(isinf(ε_rf) ? "inf" : ε_rf)" : "a$(α_A)"
λfac = parse(Float64, get(ENV, "SPCE_LAMBDA", "0"))      # 0 = stock eKMC (default, user 2026-10-12); >0 = λ-hybrid, λ = fac·⟨R⟩_MC
tag = get(ENV, "SPCE_TAG", "")

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

open(out, "w") do io
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
        res = MCm.build_reservoir(m.templates, 2, ff, 1 / T, Xoshiro(seed + 7919); M=20000, burnin_sweeps=2000, stride_sweeps=20, δ=0.03)
        Ures = sum(res.U_intra) / length(res.U_intra)
        @printf(io, "# conformer reservoir: %d conformers, acceptance %.3f, <U_intra>_ideal = %.3f ± %.3f kJ/mol, %.1f s\n",
                length(res.confs), res.acceptance, Ures * kJmol_per_eps,
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
    const_E = MCm.wolf_constant_energy(sys, ff)
    water_only = MCm.RigidSystem(m.templates, fill(1, N), L, 1 / T, Xoshiro(0))
    ΔU_tail = MCm.lrc_energy(sys, ff) - MCm.lrc_energy(water_only, ff)        # tail energy of the one solute
    println(io, "# solute LJ tail ΔU_tail = $(round(ΔU_tail * kJmol_per_eps; digits=4)) kJ/mol (added to every solute μ_ex below)")
    # per-molecule Coulomb constant of the model (RF: excluded-pair + self terms = −ke k_rf μ²; Wolf: self + intra).
    # For the GROMACS reaction-field decoupling ΔG this constant is part of −ΔG and must be added (printed as
    # mu_sol_rfconst); for Wolf vs PME it is a damping artefact and the pair-only value is the one to compare.
    E_const = MCm.coulomb_molecule_constant(m.solute, ff)
    println(io, "# solute Coulomb constant E_const = $(round(E_const * kJmol_per_eps; digits=4)) kJ/mol (model $(coulomb)); mu + E_const is the GROMACS-RF comparable value")

    # --- Metropolis MC: equilibrate + production (Widom insertion and removal estimators of the solute) ---
    st = MCm.RigidMCStats()
    δ = 0.1 / m.σ_angstrom * 3.0; δθ = 0.3
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
        end
    end
    println(io, "# MC equilibration $(mc_sweeps ÷ 4) sweeps: $(round(time() - t0; digits=1)) s, acc_t=$(st.trans_acc/st.trans_att), acc_r=$(st.rot_acc/st.rot_att)")
    flush(io)
    sys_mc = deepcopy(sys)
    t0 = time()
    mc = MCm.run_rigid_mc!(sys_mc, ff; sweeps=mc_sweeps, δ=δ, δθ=δθ, sample_every=2,
                           block_size=max(10, mc_sweeps ÷ 2 ÷ 25), widom_template=2, widom_per_sample=50,
                           inverse_widom_molecule=isol, R_every=20, reservoirs=reservoirs, δ_site=0.03)
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
    flex && @printf(io, "# MC flexible solute: <U_intra> = %.3f ± %.3f kJ/mol (ideal gas %.3f); site acc %.3f, conformer-swap acc %.3f\n",
                    MCm.mean(mc.U_intra) * kJmol_per_eps, MCm.stderr(mc.U_intra) * kJmol_per_eps, Ures * kJmol_per_eps,
                    mc.stats.site_acc / max(mc.stats.site_att, 1), mc.stats.swap_acc / max(mc.stats.swap_att, 1))
    println(io, "# MC removal-estimator block μ (kJ/mol): ", join(round.((T .* log.(mc.inv_widom.block_means) .+ ΔU_tail) .* kJmol_per_eps; digits=2), " "))
    println(io, "# MC Widom block μ (kJ/mol):             ", join(round.((-T .* log.(mc.widom.block_means) .+ ΔU_tail) .* kJmol_per_eps; digits=2), " "))
    flush(io)

    # --- eKMC (MC-kernel hybrid) from the same equilibrated configuration ---
    t0 = time()
    # λ = λfac × ⟨R⟩: ⟨R⟩ from the Metropolis production samples (relaxed-state relocation rate, robust) unless
    # SPCE_LAMBDA_FROM=pilot asks for the old 2e4-event pure-eKMC pilot (which can miss the relaxed states)
    λ_from = get(ENV, "SPCE_LAMBDA_FROM", "mc")
    R_mc = (λ_from == "mc" && !isempty(mc.R.block_means)) ? MCm.mean(mc.R) : NaN
    # species rate scaling: the solute gets the share SPCE_SOLUTE_SHARE of the relocation bursts (exact:
    # a constant factor on one species' rate keeps detailed balance, DESIGN §5c / JOURNAL entry 19)
    share = parse(Float64, get(ENV, "SPCE_SOLUTE_SHARE", "0"))     # 0 = no rate scaling (default; equal-rate scaling fails for strongly bound solutes, JOURNAL entry 20)
    R_w = isfinite(R_mc) ? MCm.mean(mc.R_species[1]) : NaN
    R_s = isfinite(R_mc) ? MCm.mean(mc.R_species[2]) : NaN
    c_sol = (share > 0 && isfinite(R_w) && isfinite(R_s) && R_s > 0) ? share / (1 - share) * R_w / R_s : 1.0
    R_eff = isfinite(R_mc) ? R_w + c_sol * R_s : NaN
    λ = λfac > 0 ? λfac * (isfinite(R_eff) ? R_eff : MCm.estimate_relocation_rate(deepcopy(sys), ff; events=20_000)) : 0.0
    println(io, "# species rate scale: <R_water>_MC=$(R_w)  <R_solute>_MC=$(R_s)  c_solute=$(c_sol)  (solute share $(share))"); flush(io)
    println(io, "# eKMC fast_path=$(fast_path); MC-kernel hybrid: SPCE_LAMBDA=$(λfac), λ=$(λ) (⟨R⟩_MC=$(R_mc) ± $(isempty(mc.R.block_means) ? NaN : MCm.stderr(mc.R)), source=$(isfinite(R_mc) ? "mc" : "pilot"))"); flush(io)
    se = λfac > 0 ? 1 : 100
    ek = MCm.run_rigid_ekmc!(sys, ff; events=ekmc_events, overlap_fast_path=fast_path, sample_every=se, block_events=max(25, ekmc_events ÷ 25),
                             debug_check_every=ekmc_events ÷ 10, mc_kernel_rate=λ, mc_δ=δ, mc_δθ=δθ,
                             species_mu=true, trace_molecule=isol, species_rate_scale=[1.0, c_sol], reservoirs=reservoirs)
    t_ek = time() - t0
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
    println(io, "# eKMC solute block μ (kJ/mol): ", join(round.((T .* log.(rs.block_means) .+ ΔU_tail) .* kJmol_per_eps; digits=2), " "))
    println(io, "# eKMC water block μ (kJ/mol):  ", join(round.(T .* log.(rw.block_means) .* kJmol_per_eps; digits=2), " "))
    println(io, "# eKMC block times: ", join(round.(rs.block_sum_t; sigdigits=3), " "))
    tail_report(io, ek.trace_phi, ek.trace_dt, 1 / T, "eKMC solute")
    println(io, "# block means U/N (eKMC, reduced): ", join(round.(ek.U.block_means; digits=3), " "))
end
println(read(out, String))

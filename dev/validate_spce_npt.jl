"""
SPC/E liquid water at T = 298.15 K and P = 1 bar, Wolf electrostatics: Metropolis NPT MC vs NPT eKMC
(Kelly Newton + Robbins–Monro volume control) vs published SPC/E data (see literature/NOTES.md §11).

Reports density (g/cm³), U/N (kJ/mol, with the Wolf constant terms), ⟨P⟩ (atm), μ_ex (kJ/mol) from
NPT Widom (−kT ln(<V e^{-βΔU}>/<V>), Kelly A.133) and from eKMC (A.132, kT ln(<(R/N)/V><V>)).
Both runs use the same tail route (`tail_route`, default :thermodynamic so that the eKMC controller
targets the ensemble that the MC acceptance with U_tail(V) samples; :virial matches LAMMPS/GROMACS).

Usage: julia --project=. dev/validate_spce_npt.jl [N] [mc_sweeps] [ekmc_events] [seed] [outdir] [tail_route]
"""

using MolSim, Random, StaticArrays, Printf, Dates
const MCm = MolSim.MC

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500
mc_sweeps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4000
ekmc_events = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20_000_000
seed = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
outdir = length(ARGS) >= 5 ? ARGS[5] : joinpath(@__DIR__, "results")
tail_route = length(ARGS) >= 6 ? Symbol(ARGS[6]) : :thermodynamic
mkpath(outdir)
# force-field variant from the environment (defaults = the site-cutoff α = 0.16 Å⁻¹ model of the first runs)
α_A = parse(Float64, get(ENV, "SPCE_ALPHA", "0.16"))
cutoff = Symbol(get(ENV, "SPCE_CUTOFF", "site"))
coulomb = Symbol(get(ENV, "SPCE_COULOMB", "wolf"))          # wolf | rf (GROMACS reaction field, SPCE_EPS_RF: 0 = ∞)
ε_rf = (x = parse(Float64, get(ENV, "SPCE_EPS_RF", "0")); x == 0 ? Inf : x)
elec_tag = coulomb === :rf ? "rf$(isinf(ε_rf) ? "inf" : ε_rf)" : "a$(α_A)"

m = MCm.spce_model(; α_per_angstrom=α_A, rc_angstrom=10.0, cutoff=cutoff, coulomb=coulomb, epsilon_rf=ε_rf)
tag = get(ENV, "SPCE_TAG", "")
out = joinpath(outdir, "spce_npt_N$(N)_seed$(seed)_$(tail_route)_$(cutoff)_$(elec_tag)_cap$(Int(m.ff.ucap_mix[1,1]))$(tag).txt")
ff = m.ff
T_K = 298.15; P_bar = 1.0
T = T_K / m.ϵ_kelvin
kB = 1.3806488e-23; NA = 6.02214129e23
P_star = P_bar * 1e5 * (m.σ_angstrom * 1e-10)^3 / (m.ϵ_kelvin * kB)
kJmol_per_eps = m.ϵ_kelvin * kB * NA / 1000
atm_per_pstar = m.ϵ_kelvin * kB / (m.σ_angstrom * 1e-10)^3 / 101325.0
gcm3_per_rhostar = 18.0153 / NA / (m.σ_angstrom * 1e-8)^3   # (g/molecule) / σ³ in cm³
ρ0 = 1.0 / gcm3_per_rhostar                                  # start at 1.000 g/cm³
L = cbrt(N / ρ0)

open(out, "w") do io
    println(io, "# SPC/E NPT validation $(Dates.now())  N=$N  T=$T_K K  P=$P_bar bar (P*=$(round(P_star; sigdigits=4)))  tail_route=$tail_route")
    println(io, "# Wolf α=$(α_A) 1/Å rc=10 Å cutoff=$(cutoff); LRC on; overlap cap on; start ρ=1.000 g/cm³, L=$(round(L*m.σ_angstrom; digits=2)) Å")
    flush(io)
    sys = MCm.RigidSystem([m.template], fill(1, N), L, 1 / T, Xoshiro(seed))
    MCm.random_rigid_system!(sys, ff)
    const_E = MCm.wolf_constant_energy(sys, ff)
    δ = 0.3 / m.σ_angstrom; δθ = 0.3
    st = MCm.RigidMCStats()
    t0 = time()
    for s in 1:(mc_sweeps ÷ 4)
        for _ in 1:N
            MCm.mc_translate!(sys, ff, rand(sys.rng, 1:N), δ, st)
            MCm.mc_rotate!(sys, ff, rand(sys.rng, 1:N), δθ, st)
        end
        U = MCm.total_energy(sys, ff)
        MCm.mc_volume_trial!(sys, ff, P_star, 0.02, st; U_current=U)
    end
    println(io, "# MC NPT equilibration $(mc_sweeps ÷ 4) sweeps: $(round(time() - t0; digits=1)) s, acc_t=$(st.trans_acc/st.trans_att) acc_r=$(st.rot_acc/st.rot_att) acc_v=$(st.vol_acc/max(1,st.vol_att)), ρ=$(round(N / sys.L^3 * gcm3_per_rhostar; digits=4)) g/cm³")
    flush(io)
    sys_mc = deepcopy(sys)
    t0 = time()
    mc = MCm.run_rigid_mc!(sys_mc, ff; sweeps=mc_sweeps, δ=δ, δθ=δθ, Pext=P_star, max_dlnV=0.02, vol_every=1,
                           sample_every=2, block_size=max(10, mc_sweeps ÷ 2 ÷ 40), widom_template=1,
                           widom_per_sample=20, tail_route=tail_route)
    t_mc = time() - t0
    V̄ = MCm.mean(mc.V); wV, ewV = MCm.mean(mc.widomV), MCm.stderr(mc.widomV)
    @printf(io, "MC   NPT sweeps=%d time=%.0fs  rho=%.4f ± %.4f g/cm3  U/N=%.3f ± %.3f kJ/mol  P=%.1f ± %.1f atm  mu_widomNPT=%.2f ± %.2f kJ/mol  acc_v=%.3f\n",
            mc_sweeps, t_mc, MCm.mean(mc.rho) * gcm3_per_rhostar, MCm.stderr(mc.rho) * gcm3_per_rhostar,
            (MCm.mean(mc.U) + const_E / N) * kJmol_per_eps, MCm.stderr(mc.U) * kJmol_per_eps,
            MCm.mean(mc.P) * atm_per_pstar, MCm.stderr(mc.P) * atm_per_pstar,
            -T * log(wV / V̄) * kJmol_per_eps, T * ewV / wV * kJmol_per_eps, mc.stats.vol_acc / mc.stats.vol_att)
    flush(io)
    V0 = sys.L^3
    burn = min(200 * 10 * N + ekmc_events ÷ 5, ekmc_events ÷ 2)   # vol_equil_moves + 20 %, at most half the run
    t0 = time()
    λfac = parse(Float64, get(ENV, "SPCE_LAMBDA", "0"))
    λ = λfac > 0 ? λfac * MCm.estimate_relocation_rate(deepcopy(sys), ff; events=20_000) : 0.0
    println(io, "# eKMC MC-kernel hybrid: SPCE_LAMBDA=$(λfac), λ=$(λ)"); flush(io)
    se = λfac > 0 ? 1 : 100
    # with the hybrid an event is ~one sweep, so volume moves go every 10 events (not 10·N)
    vee = λfac > 0 ? 10 : 0
    # full-gain Newton phase length (volume moves); with the hybrid a move comes every 10 events, and
    # 200 moves were not enough to converge the volume at N = 500 (JOURNAL 2026-10-12)
    veq = parse(Int, get(ENV, "SPCE_VOL_EQUIL", "200"))
    λfac > 0 && (burn = min(veq * vee + ekmc_events ÷ 5, ekmc_events ÷ 2))
    println(io, "# NPT eKMC: vol_every_events=$(vee) vol_equil_moves=$(veq) burnin_events=$(burn)"); flush(io)
    ek = MCm.run_rigid_ekmc!(sys, ff; events=ekmc_events, Pext=P_star, max_dV=0.02 * V0, vol_every_sweeps=10,
                             vol_every_events=vee, sample_every=se, block_events=max(25, (ekmc_events - burn) ÷ 25),
                             vol_equil_moves=veq, burnin_events=burn, debug_check_every=ekmc_events ÷ 10,
                             tail_route=tail_route, mc_kernel_rate=λ, mc_δ=δ, mc_δθ=δθ)
    t_ek = time() - t0
    rnv, ernv = MCm.mean(ek.RNV), MCm.stderr(ek.RNV); Vm, eVm = MCm.mean(ek.V), MCm.stderr(ek.V)
    @printf(io, "eKMC NPT events=%d time=%.0fs  rho=%.4f ± %.4f g/cm3  U/N=%.3f ± %.3f kJ/mol  P=%.1f ± %.1f atm  mu_A132=%.2f ± %.2f kJ/mol (acc %.2f)  vol_moves=%d\n",
            ekmc_events, t_ek, MCm.mean(ek.rho) * gcm3_per_rhostar, MCm.stderr(ek.rho) * gcm3_per_rhostar,
            (MCm.mean(ek.U) + const_E / N) * kJmol_per_eps, MCm.stderr(ek.U) * kJmol_per_eps,
            MCm.mean(ek.P) * atm_per_pstar, MCm.stderr(ek.P) * atm_per_pstar,
            T * log(rnv * Vm) * kJmol_per_eps, T * sqrt((ernv / rnv)^2 + (eVm / Vm)^2) * kJmol_per_eps,
            MCm.mu_ex_npt(ek.mu_npt, T) * kJmol_per_eps, ek.vol_moves)
    println(io, "# eKMC block means rho (reduced): ", join(round.(ek.rho.block_means; digits=4), " "))
    println(io, "# MC   block means rho (reduced): ", join(round.(mc.rho.block_means; digits=4), " "))
end
println(read(out, String))

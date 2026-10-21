# The protocol of test/test_flexible_solvent_slow.jl (liquid OPLS-AA ethanol N = 64, RF, rc 9 Å, 298 K, 0.785 g/cm³; MC with
# site + torsion moves vs eKMC + flexible solvent L-moves) at a chosen seed, to get seed-spread errors for U, U_intra, P
# (two eKMC runs of the fixed-seed test differed by 3× their block errors). Usage: julia -t 4 --project=. dev/ethanol_liquid_seeds.jl <seed> [eq]
# eq: start eKMC from the END of the MC production run (equilibrated) instead of the common MC-equilibrated start.
# ENV ETH_NU = ν (default 1e-3) | mc (= ⟨R⟩_MC, R_every 20); ETH_EVENTS (default 3e6).
# ENV ETH_POOL_TORSION = 1 (default: pool with torsion moves) | 0 (the pre-2026-10-20 pool: site moves only, torsions not equilibrated).
# Third argument (JOURNAL 121): electrostatics rf (default) | wolf (α = 0.2/Å) | ewald (α from ewald-rtol 1e-5; no pressure).
using MolSim, Random, Printf
const MCs = MolSim.MC
seed = parse(Int, ARGS[1]); eqstart = length(ARGS) >= 2 && ARGS[2] == "eq"
coul = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :rf
function compress_to!(sys, ff, L; steps=12, nsteps=150)
    f = (L / sys.L)^(1 / steps)
    for _ in 1:steps
        sys.L *= f
        for i in 1:sys.n_mol; MCs.set_molecule!(sys, i, sys.com[:, i] .* f, sys.quat[:, i]); end
        MCs.minimize_rigid!(sys, ff; nsteps=nsteps)
    end
    sys.L = L
end
N = 64; T = 298.0
m = MCs.spce_solute_model(:ethanol_flex; coulomb=coul, epsilon_rf=Inf, cutoff=:site, rc_angstrom=9.0,
                          α_per_angstrom=coul === :ewald ? MCs.ewald_alpha_for(9.0, 1e-5) : 0.20)
t = m.templates[2]; ff = m.ff; β = m.ϵ_kelvin / T; kJ = m.ϵ_kelvin * 0.0083144626
ρ = 0.785 / 46.07 * 6.02214076e23 / 1e24
L = cbrt(N / ρ) / m.σ_angstrom
sys = MCs.RigidSystem([t], fill(1, N), 2L, β, Xoshiro(100 + seed)); MCs.random_rigid_system!(sys, ff)
compress_to!(sys, ff, L)
coul === :ewald && MCs.attach_ewald!(sys, ff; rtol=1e-5)
res = MCs.build_reservoir([t], 1, ff, β, Xoshiro(200 + seed); M=4000, burnin_sweeps=500, stride_sweeps=20, δ=0.02,
                         torsion=get(ENV, "ETH_POOL_TORSION", "1") == "1")   # torsion moves in the pool: site steps alone do not equilibrate the torsions (JOURNAL 2026-10-20, liquid ethanol pool)
δ, δθ, δs, φ = 0.05, 0.15, 0.01, 0.5
MCs.run_rigid_mc!(sys, ff; sweeps=1000, δ=δ, δθ=δθ, sample_every=100, block_size=5, reservoirs=[res], δ_site=δs, torsion_φ=φ)
sysmc = deepcopy(sys)
nu_opt = get(ENV, "ETH_NU", "1e-3")                  # solvent L-move ν: a number, or "mc" = ⟨R⟩ of the MC production (the production drivers' recipe)
mc = MCs.run_rigid_mc!(sysmc, ff; sweeps=12_000, δ=δ, δθ=δθ, sample_every=2, block_size=200, reservoirs=[res], δ_site=δs, torsion_φ=φ,
                       R_every=nu_opt == "mc" ? 20 : 0)
ν = nu_opt == "mc" ? MCs.mean(mc.R) : parse(Float64, nu_opt)
events = parse(Int, get(ENV, "ETH_EVENTS", "3000000"))
ek = MCs.run_rigid_ekmc!(eqstart ? deepcopy(sysmc) : deepcopy(sys), ff; events=events, sample_every=20, block_events=events ÷ 25, burnin_events=events ÷ 10,
        debug_check_every=events ÷ 6, reservoirs=[res], local_solvent=true, solvent_flexible=true, solvent_nu=ν,
        mc_δ=δ, mc_δθ=δθ, solvent_site_a=δs, solvent_φ=φ)
@printf("RESULT coulomb=%s seed=%d start=%s  MC U %.4f ± %.4f Uintra %.4f ± %.4f P %.4f ± %.4f | eKMC U %.4f ± %.4f Uintra %.4f ± %.4f P %.4f ± %.4f (kJ/mol, P reduced) relocations %d nu %.4g events %d\n",
        coul, seed, eqstart ? "after-MC" : "common", MCs.mean(mc.U) * kJ, MCs.stderr(mc.U) * kJ, MCs.mean(mc.U_intra) * kJ, MCs.stderr(mc.U_intra) * kJ, MCs.mean(mc.P), MCs.stderr(mc.P),
        MCs.mean(ek.U) * kJ, MCs.stderr(ek.U) * kJ, MCs.mean(ek.U_intra) * kJ, MCs.stderr(ek.U_intra) * kJ, MCs.mean(ek.P), MCs.stderr(ek.P), ek.relocations, ν, events)

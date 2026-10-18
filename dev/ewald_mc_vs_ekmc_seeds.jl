# The protocol of test/test_ewald_ekmc_slow.jl (SPC/E N = 48, rc 4.5 Å, Ewald :mixed, eKMC + solvent L-moves vs Metropolis)
# with a chosen seed and run length, to tell a bias from a short-run fluctuation (slow test 218551: −72.26 ± 0.20 vs
# MC −71.18 ± 0.18 at seed 21, 3e5 events). Usage: julia --project=. dev/ewald_mc_vs_ekmc_seeds.jl <seed> <factor>
using MolSim, Random, Printf
const M = MolSim.MC
seed = parse(Int, ARGS[1]); f = parse(Int, ARGS[2])
N = 48; rc = 4.5; ρ = 0.0334
α = M.ewald_alpha_for(rc, 1e-5)
m = M.spce_model(; α_per_angstrom=α, rc_angstrom=rc, coulomb=:ewald)
L = cbrt(N / ρ) / m.σ_angstrom
sys = M.RigidSystem([m.template], fill(1, N), L, m.ϵ_kelvin / 300.0, Xoshiro(seed))
M.random_rigid_system!(sys, m.ff)
M.attach_ewald!(sys, m.ff; rtol=1e-5)
st = M.RigidMCStats()
for _ in 1:300, _ in 1:N
    M.mc_translate!(sys, m.ff, rand(sys.rng, 1:N), 0.1, st); M.mc_rotate!(sys, m.ff, rand(sys.rng, 1:N), 0.3, st)
end
mc = M.run_rigid_mc!(deepcopy(sys), m.ff; sweeps=3000f, δ=0.1, δθ=0.3, sample_every=1, block_size=100f)
ν = M.estimate_relocation_rate(deepcopy(sys), m.ff; events=5_000) / N
events = 300_000f
ek = M.run_rigid_ekmc!(sys, m.ff; events=events, sample_every=10, block_events=events ÷ 25,
                       debug_check_every=events ÷ 5, local_solvent=true, solvent_nu=ν)
@printf("RESULT seed=%d factor=%d  MC U = %.4f ± %.4f   eKMC U = %.4f ± %.4f   diff/σ = %.2f   relocations %d\n", seed, f,
        M.mean(mc.U), M.stderr(mc.U), M.mean(ek.U), M.stderr(ek.U),
        (M.mean(ek.U) - M.mean(mc.U)) / sqrt(M.stderr(mc.U)^2 + M.stderr(ek.U)^2), ek.relocations)

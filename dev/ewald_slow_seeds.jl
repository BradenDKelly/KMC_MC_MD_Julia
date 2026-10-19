# Seed study of the redesigned Ewald / PME slow test (test/test_ewald_ekmc_slow.jl, JOURNAL 123): same protocol at a chosen
# seed (start and RNG). Usage: julia --project=. dev/ewald_slow_seeds.jl <ewald|pme> <seed> [N rc_Å mc_sweeps ekmc_events]
using MolSim, Random, Printf
const MCs = MolSim.MC
recip = Symbol(ARGS[1]); seed = parse(Int, ARGS[2])
N = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 48; rc = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 4.5; ρ = 0.0334
nmc = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 3000; events = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 600_000
α = MCs.ewald_alpha_for(rc, 1e-5)
m = MCs.spce_model(; α_per_angstrom=α, rc_angstrom=rc, coulomb=:ewald)
sys = MCs.RigidSystem([m.template], fill(1, N), cbrt(N / ρ) / m.σ_angstrom, m.ϵ_kelvin / 300.0, Xoshiro(seed))
MCs.random_rigid_system!(sys, m.ff)
recip === :ewald ? MCs.attach_ewald!(sys, m.ff; rtol=1e-5) : MCs.attach_pme!(sys, m.ff; spacing=1.2 / m.σ_angstrom, order=4)
st = MCs.RigidMCStats()
for _ in 1:300, _ in 1:N
    MCs.mc_translate!(sys, m.ff, rand(sys.rng, 1:N), 0.1, st); MCs.mc_rotate!(sys, m.ff, rand(sys.rng, 1:N), 0.3, st)
end
mc = MCs.run_rigid_mc!(sys, m.ff; sweeps=nmc, δ=0.1, δθ=0.3, sample_every=1, block_size=100)
ek = MCs.run_rigid_ekmc!(sys, m.ff; events=events, sample_every=10, block_events=events ÷ 25, burnin_events=events ÷ 10,
                         debug_check_every=events ÷ 5, local_solvent=true, solvent_nu=1e-3, mc_δ=0.1, mc_δθ=0.3)
d = (MCs.mean(ek.U) - MCs.mean(mc.U)) / sqrt(MCs.stderr(mc.U)^2 + MCs.stderr(ek.U)^2)
@printf("RESULT %s N=%d seed=%d  MC U = %.4f ± %.4f   eKMC U = %.4f ± %.4f   diff/σ = %.2f   relocations %d\n", recip, N, seed,
        MCs.mean(mc.U), MCs.stderr(mc.U), MCs.mean(ek.U), MCs.stderr(ek.U), d, ek.relocations, MCs.mean(mc.P), MCs.stderr(mc.P), MCs.mean(ek.P), MCs.stderr(ek.P))

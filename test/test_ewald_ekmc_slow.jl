# SPC/E with reciprocal-space Ewald (DESIGN §13): eKMC with solvent L-moves samples the same canonical U as Metropolis MC
# (cross-method consistency; slow suite, MOLSIM_SLOW_TESTS=1).
using Test, Random
using MolSim
const MCs = MolSim.MC

@testset "SPC/E Ewald NVT: eKMC + solvent L-moves vs Metropolis MC (U)" begin
    N = 48; rc = 4.5; ρ = 0.0334
    α = MCs.ewald_alpha_for(rc, 1e-5)
    m = MCs.spce_model(; α_per_angstrom=α, rc_angstrom=rc, coulomb=:ewald)
    L = cbrt(N / ρ) / m.σ_angstrom
    sys = MCs.RigidSystem([m.template], fill(1, N), L, m.ϵ_kelvin / 300.0, Xoshiro(21))
    MCs.random_rigid_system!(sys, m.ff)
    MCs.attach_ewald!(sys, m.ff; rtol=1e-5)
    st = MCs.RigidMCStats()
    for _ in 1:300, _ in 1:N
        MCs.mc_translate!(sys, m.ff, rand(sys.rng, 1:N), 0.1, st); MCs.mc_rotate!(sys, m.ff, rand(sys.rng, 1:N), 0.3, st)
    end
    mc = MCs.run_rigid_mc!(deepcopy(sys), m.ff; sweeps=3000, δ=0.1, δθ=0.3, sample_every=1, block_size=100)
    ν = MCs.estimate_relocation_rate(deepcopy(sys), m.ff; events=5_000) / N
    events = 300_000
    ek = MCs.run_rigid_ekmc!(sys, m.ff; events=events, sample_every=10, block_events=events ÷ 25,
                             debug_check_every=events ÷ 5, local_solvent=true, solvent_nu=ν)
    @info "SPC/E Ewald MC vs eKMC" MCs.mean(mc.U) MCs.stderr(mc.U) MCs.mean(ek.U) MCs.stderr(ek.U) ek.relocations
    @test abs(MCs.mean(mc.U) - MCs.mean(ek.U)) <= 3 * sqrt(MCs.stderr(mc.U)^2 + MCs.stderr(ek.U)^2)
end

# SPC/E with reciprocal-space Ewald (DESIGN §13) and the SPME mesh path (§16): eKMC with solvent L-moves samples the same
# canonical U as Metropolis MC (cross-method consistency; slow suite, MOLSIM_SLOW_TESTS=1). Design after JOURNAL 114/117:
# the solvent L-move prefactor ν is FIXED (an efficiency choice; a pilot ⟨R⟩/N varied by orders of magnitude between seeds and
# made the test unstable) and eKMC starts from the END of the MC production run (equilibrated start).
using Test, Random
using MolSim
const MCs = MolSim.MC

@testset "SPC/E $(recip) NVT: eKMC + solvent L-moves vs Metropolis MC (U)" for recip in (:ewald, :pme)
    N = 48; rc = 4.5; ρ = 0.0334
    α = MCs.ewald_alpha_for(rc, 1e-5)
    m = MCs.spce_model(; α_per_angstrom=α, rc_angstrom=rc, coulomb=:ewald)
    L = cbrt(N / ρ) / m.σ_angstrom
    sys = MCs.RigidSystem([m.template], fill(1, N), L, m.ϵ_kelvin / 300.0, Xoshiro(21))
    MCs.random_rigid_system!(sys, m.ff)
    recip === :ewald ? MCs.attach_ewald!(sys, m.ff; rtol=1e-5) :   # :mixed (GROMACS convention), as in production
                       MCs.attach_pme!(sys, m.ff; spacing=1.2 / m.σ_angstrom, order=4)   # GROMACS defaults
    st = MCs.RigidMCStats()
    for _ in 1:300, _ in 1:N
        MCs.mc_translate!(sys, m.ff, rand(sys.rng, 1:N), 0.1, st); MCs.mc_rotate!(sys, m.ff, rand(sys.rng, 1:N), 0.3, st)
    end
    mc = MCs.run_rigid_mc!(sys, m.ff; sweeps=3000, δ=0.1, δθ=0.3, sample_every=1, block_size=100)
    events = 600_000
    ek = MCs.run_rigid_ekmc!(sys, m.ff; events=events, sample_every=10, block_events=events ÷ 25, burnin_events=events ÷ 10,
                             debug_check_every=events ÷ 5, local_solvent=true, solvent_nu=1e-3, mc_δ=0.1, mc_δθ=0.3)
    @info "SPC/E $(recip) MC vs eKMC" MCs.mean(mc.U) MCs.stderr(mc.U) MCs.mean(ek.U) MCs.stderr(ek.U) MCs.mean(mc.P) MCs.stderr(mc.P) MCs.mean(ek.P) MCs.stderr(ek.P) ek.relocations
    @test abs(MCs.mean(mc.U) - MCs.mean(ek.U)) <= 3 * sqrt(MCs.stderr(mc.U)^2 + MCs.stderr(ek.U)^2)
    # pressure with the reciprocal molecular virial (E6: Ewald and the PME split)
    @test abs(MCs.mean(mc.P) - MCs.mean(ek.P)) <= 3 * sqrt(MCs.stderr(mc.P)^2 + MCs.stderr(ek.P)^2)
end

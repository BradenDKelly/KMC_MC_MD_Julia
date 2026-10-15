# Rejection-free local moves of every solvent molecule (DESIGN §11.2 stage 2): involution, exact bookkeeping, and
# time-weighted U, P, μ (from ⟨R/N⟩) equal to Metropolis MC / Widom on the rigid LJ diatomic fluid.
using Test, Random
using MolSim
const MCm = MolSim.MC

slm_within3σ(a, ea, b, eb) = abs(a - b) <= 3 * sqrt(ea^2 + eb^2)

function slm_diatomic(N, ρ, T, seed; equil=300)
    sites = zeros(3, 2); sites[1, 1] = -0.3; sites[1, 2] = 0.3
    tmpl = MCm.RigidTemplate("LJ2", sites, [1, 1], [0.0, 0.0], [1.0, 1.0])
    ff = MCm.SiteForceField([1.0], [1.0]; rc=2.5, coulomb=:none)
    sys = MCm.RigidSystem([tmpl], fill(1, N), cbrt(N / ρ), 1 / T, Xoshiro(seed)); MCm.random_rigid_system!(sys, ff)
    st = MCm.RigidMCStats()
    for _ in 1:equil, _ in 1:N
        MCm.mc_translate!(sys, ff, rand(sys.rng, 1:N), 0.15, st); MCm.mc_rotate!(sys, ff, rand(sys.rng, 1:N), 0.4, st)
    end
    return tmpl, ff, sys
end

@testset "Solvent L-moves: involution of every slot and exact column bookkeeping" begin
    tmpl, ff, sys = slm_diatomic(32, 0.5, 1.5, 3)
    ek = MCm.init_rigid_ekmc(sys, ff)
    S = MCm.SolventLocalMoves(sys, ff, collect(1:32); kinds=[MCm.LM_T, MCm.LM_R], a=0.2, θmax=0.4, ν=1.0, νD=1.0)
    for (ii, j) in ((5, 1), (17, 2))
        i = S.mols[ii]
        com0 = copy(sys.com); q0 = copy(sys.quat); φ0 = copy(ek.phi)
        MCm.slm_total_rate!(S, ek, sys.β)
        ξ = sum(S.Wm[1:ii-1]) + sum(S.rates[1:j-1, ii]) + 0.5 * S.rates[j, ii]
        @test MCm.slm_fire!(S, sys, ff, ek, ξ) == i
        @test !(sys.com[:, i] ≈ com0[:, i] && sys.quat[:, i] ≈ q0[:, i])
        @test MCm.slm_check!(S, sys, ff) < 1e-9                      # incremental columns = rebuild
        MCm.slm_total_rate!(S, ek, sys.β)
        @test S.cu[j, ii] ≈ φ0[i] atol = 1e-9                        # the slot now holds the way back
        ξ = sum(S.Wm[1:ii-1]) + sum(S.rates[1:j-1, ii]) + 0.5 * S.rates[j, ii]
        MCm.slm_fire!(S, sys, ff, ek, ξ)
        @test isapprox(sys.com, com0; atol=1e-12) && isapprox(sys.quat, q0; atol=1e-12)
        @test isapprox(ek.phi, φ0; atol=1e-9)
    end
    # a stock relocation of one molecule followed by column updates = full rebuild
    MCm.slm_total_rate!(S, ek, sys.β)
    for _ in 1:20
        k = MCm.select_by_mobility(ek, sys.rng)
        MCm.set_molecule!(sys, k, sys.L .* rand(sys.rng, 3), MCm.uniform_random_quaternion(sys.rng))
        MCm.update_rigid_ekmc_after_move!(ek, sys, ff, k)
        MCm.slm_moved!(S, sys, ff, k)
    end
    @test MCm.slm_check!(S, sys, ff) < 1e-9
end

@testset "Rigid LJ diatomic NVT: eKMC + solvent L-moves vs Metropolis MC (U, P) and Widom (μ from ⟨R/N⟩)" begin
    N = 32; ρ = 0.3; T = 2.0
    tmpl, ff, sys = slm_diatomic(N, ρ, T, 3; equil=500)
    mc = MCm.run_rigid_mc!(deepcopy(sys), ff; sweeps=4000, δ=0.15, δθ=0.4, sample_every=1, block_size=100,
                           widom_template=1, widom_per_sample=50)
    ν = MCm.estimate_relocation_rate(deepcopy(sys), ff; events=20_000) / N
    events = 1_000_000
    ek = @test_logs min_level=Base.CoreLogging.Warn MCm.run_rigid_ekmc!(sys, ff; events=events, sample_every=10,
            block_events=events ÷ 25, debug_check_every=events ÷ 10, local_solvent=true, solvent_nu=ν, mc_δ=0.15, mc_δθ=0.4)
    S = ek.slmoves
    w, ew = MCm.mean(mc.widom), MCm.stderr(mc.widom)
    μ_w, eμ_w = -T * log(w), T * ew / w
    μ_e, eμ_e = T * log(MCm.ratio_mean(ek.RN)), T * MCm.ratio_jackknife_stderr(ek.RN; f=log)
    @info "LJ diatomic + solvent L-moves" MCm.mean(mc.U) MCm.stderr(mc.U) MCm.mean(ek.U) MCm.stderr(ek.U) MCm.mean(mc.P) MCm.stderr(mc.P) MCm.mean(ek.P) MCm.stderr(ek.P) μ_w eμ_w μ_e eμ_e S.n_move S.n_refresh ek.relocations
    @test sum(S.n_move) > events ÷ 10 && ek.relocations > events ÷ 10 && S.n_refresh > 0
    @test slm_within3σ(MCm.mean(mc.U), MCm.stderr(mc.U), MCm.mean(ek.U), MCm.stderr(ek.U))
    @test slm_within3σ(MCm.mean(mc.P), MCm.stderr(mc.P), MCm.mean(ek.P), MCm.stderr(ek.P))
    @test slm_within3σ(μ_w, eμ_w, μ_e, eμ_e)
end

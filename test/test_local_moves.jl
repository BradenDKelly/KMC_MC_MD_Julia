# Rejection-free local moves (L-moves, DESIGN §11.2) and the vacuum eKMC conformer pool (§11.3 E3): fast checks.
# The statistical checks (ladder + L-moves μ = stock on the LJ dimer) are in test_ladder.jl (run separately, ~1 h).
using Test, Random
using MolSim
const MCm = MolSim.MC

lm_within3σ(a, ea, c, ec) = abs(a - c) <= 3 * sqrt(ea^2 + ec^2)

function lm_dimer_system(ε_s, N, T, ρ, seed)
    sites = zeros(3, 2); sites[1, 1] = -0.3; sites[1, 2] = 0.3
    solv = MCm.RigidTemplate("LJ2", sites, [1, 1], [0.0, 0.0], [1.0, 1.0])
    bt = MCm.BondedTerms(2; bonds=[MCm.HarmonicBond(1, 2, 0.6, 400.0)])
    dimer = MCm.RigidTemplate("flexdimer", sites, [2, 2], [0.0, 0.0], [1.0, 1.0]; bonded=bt)
    ff = MCm.SiteForceField([1.0, 1.1], [1.0, ε_s]; rc=2.5, coulomb=:none)
    β = 1 / T; L = cbrt(N / ρ); Nt = N + 1
    res = MCm.build_reservoir([solv, dimer], 2, ff, β, Xoshiro(3); M=5000, burnin_sweeps=500, stride_sweeps=10, δ=0.05)
    sys = MCm.RigidSystem([solv, dimer], vcat(fill(1, N), 2), L, β, Xoshiro(seed)); MCm.random_rigid_system!(sys, ff)
    MCm.minimize_rigid!(sys, ff; nsteps=200)
    st = MCm.RigidMCStats()
    for _ in 1:300, _ in 1:Nt
        MCm.mc_translate!(sys, ff, rand(sys.rng, 1:Nt), 0.15, st); MCm.mc_rotate!(sys, ff, rand(sys.rng, 1:Nt), 0.4, st)
        MCm.mc_site_move!(sys, ff, Nt, 0.05, st); MCm.mc_conformer_swap!(sys, ff, Nt, res, st)
    end
    return (; solv, dimer, ff, res, sys, L, β, Nt)
end


@testset "L-moves: every slot is an involution (fire j twice → the same state)" begin
    S = lm_dimer_system(1.0, 32, 1.5, 0.4, 11)
    sys = deepcopy(S.sys); ff = S.ff; s = S.Nt
    ek = MCm.init_rigid_ekmc(sys, ff)
    ek.frac = s; ek.lam[s] = 0.5; MCm.rebuild_rigid_ekmc!(ek, sys, ff)
    lm = MCm.LocalMoves(sys, s, 10; kinds=[MCm.LM_T, MCm.LM_R, MCm.LM_C], a=0.2, θmax=0.4, ν=1.0, νD=1.0, reservoir=S.res)
    for j in 1:3
        com0 = copy(sys.com[:, s]); q0 = copy(sys.quat[:, s]); B0 = copy(sys.body[s]); u0 = ek.frac_u
        W = MCm.lm_total_rate!(lm, sys, ff, sys.β, 0.5, ek.frac_u)
        @test W > 0 && !lm.stale
        @test lm.cand_u[j] ≈ (MCm.lm_rebuild!(lm, sys, ff); lm.cand_u[j])
        ξ = sum(lm.rates[1:j-1]) + 0.5 * lm.rates[j]
        @test MCm.lm_fire!(lm, sys, ff, ek, ξ) == j
        @test !(sys.com[:, s] ≈ com0 && sys.quat[:, s] ≈ q0 && sys.body[s] ≈ B0)
        MCm.lm_total_rate!(lm, sys, ff, sys.β, 0.5, ek.frac_u)
        @test lm.cand_u[j] ≈ u0 atol = 1e-9                      # the slot now holds the way back
        ξ = sum(lm.rates[1:j-1]) + 0.5 * lm.rates[j]
        MCm.lm_fire!(lm, sys, ff, ek, ξ)
        @test isapprox(sys.com[:, s], com0; atol=1e-12) && isapprox(sys.quat[:, s], q0; atol=1e-12) && sys.body[s] == B0
        @test ek.frac_u ≈ u0 atol = 1e-9
    end
end

lm_mean(x) = sum(x) / length(x)
function blockse(x; nb=20)
    n = length(x) ÷ nb; bm = [sum(x[(b-1)*n+1:b*n]) / n for b in 1:nb]; m = lm_mean(bm)
    return sqrt(sum(abs2, bm .- m) / (nb - 1) / nb)
end

@testset "Conformer pool by vacuum rejection-free eKMC = vacuum Metropolis pool (<U_intra>, dimer and ethanol)" begin
    S = lm_dimer_system(1.0, 8, 1.5, 0.4, 3)
    for (tmpls, idx, ff, β, δ) in ((S.sys.templates, 2, S.ff, S.β, 0.05),
                                   (begin m = MCm.spce_solute_model(:ethanol_flex; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=10.0)
                                        (m.templates, 2, m.ff, m.ϵ_kelvin / 300.0, 0.03) end))
        r_mc = MCm.build_reservoir(tmpls, idx, ff, β, Xoshiro(5); M=4000, burnin_sweeps=500, stride_sweeps=20, δ=δ)
        r_ek = MCm.build_reservoir_ekmc(tmpls, idx, ff, β, Xoshiro(6); M=4000, burnin_sweeps=500, stride_sweeps=20, δ=δ)
        a, ea = lm_mean(r_mc.U_intra), blockse(r_mc.U_intra); b, eb = lm_mean(r_ek.U_intra), blockse(r_ek.U_intra)
        @info "vacuum <U_intra> (reduced): Metropolis vs rejection-free eKMC" tmpls[idx].name a ea b eb r_ek.acceptance
        @test length(r_ek.confs) == 4000 && all(c -> size(c) == size(r_mc.confs[1]), r_ek.confs)
        @test abs(a - b) <= 4 * sqrt(ea^2 + eb^2)
    end
end

@testset "Ladder + hub + L-moves: bookkeeping exact (debug rebuilds), moves at every rung with λ > 0, D0 diagnostics" begin
    S = lm_dimer_system(1.0, 32, 1.5, 0.4, 11)
    ν = MCm.estimate_relocation_rate(deepcopy(S.sys), S.ff; events=20_000)
    ev = 100_000
    lad = @test_logs min_level=Base.CoreLogging.Warn MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=ev, sample_every=1,
            block_events=ev ÷ 10, species_mu=true, reservoirs=[nothing, S.res], ladder_solute=S.Nt, ladder_nu=ν,
            ladder_wl_events=ev ÷ 3, ladder_jump=true, ladder_jump_max=7, ladder_local=true, debug_check_every=ev ÷ 50)
    lm = lad.lmoves; sm = MCm.ladder_summary(lad.ladder)
    @test lm.kinds == [1, 1, 1, 2, 2, 2, 3, 3]
    @test all(lm.n_move .> 0) && lm.n_move_rung[1] == 0 && sum(lm.n_move_rung) > 0 && lm.n_refresh > 0
    @test sum(sm.n_jump[8:end]) == 0 && sum(sm.n_jump[3:7]) > 0          # hub limited to rungs 3…7
    @test all(0 .< sm.res_frac[sm.t_rung .> 0] .<= 1) && all(sm.pose_frac .<= sm.res_frac .+ 1e-12)
end

# CUDA backend of the reciprocal-space Ewald products (DESIGN §13.3 E2): the same bookkeeping as test_ewald_ekmc.jl on the
# GPU, rows and candidate products equal to the CPU backend, incremental = rebuild. Run in an environment with CUDA:
#   julia --project=/shared/kmc_polyatomic/envs/molsim_gpu test/test_ewald_cuda.jl
using Test, Random
using CUDA
using MolSim
const MCg = MolSim.MC

include(joinpath(@__DIR__, "test_ewald_ekmc.jl"))           # CPU tests + helpers (ewald_water, ek_drift, …)

@testset "CUDA backend: rows, rebuild and candidate products = CPU; incremental = rebuild" begin
    CUDA.functional() || error("CUDA not functional on $(gethostname())")
    m, sys, _ = ewald_water(40, 12)
    ff = m.ff; N = sys.n_mol; s = N
    sysc = deepcopy(sys)
    MCg.set_recip_backend!(sys.recip, :cuda; precision=:double)
    @test sys.recip.Sd isa CuArray
    rg = zeros(N); rc = zeros(N); w = zeros(N)
    MCg.eval_row!(rg, w, sys, ff, 3); MCg.eval_row!(rc, w, sysc, ff, 3)
    @test maximum(abs.(rg .- rc)) < 1e-11
    ek = MCg.init_rigid_ekmc(sys, ff); ekc = MCg.init_rigid_ekmc(sysc, ff)
    @test maximum(abs.(ek.pair_u .- ekc.pair_u)) < 1e-11
    ek.frac = s; ek.lam[s] = 0.6; MCg.rebuild_rigid_ekmc!(ek, sys, ff)
    S = MCg.SolventLocalMoves(sys, ff, collect(1:N-1); kinds=[MCg.LM_T, MCg.LM_R, MCg.LM_T], a=0.1, θmax=0.3, ν=1.0,
                              νD=1.0, frac=s)
    @test S.crSd isa CuArray
    lm = MCg.LocalMoves(sys, s, 2; kinds=[MCg.LM_T, MCg.LM_R], a=0.1, θmax=0.3, ν=1.0, νD=1.0)
    MCg.lm_rebuild!(lm, sys, ff)
    for it in 1:60
        if it % 3 == 0
            MCg.slm_total_rate!(S, ek, sys.β)
            i = MCg.slm_fire!(S, sys, ff, ek, rand(sys.rng) * S.W)
            MCg.lm_molecule_moved!(lm, sys, ff, i)
        elseif it % 3 == 1
            k = rand(sys.rng, 1:N-1)
            MCg.set_molecule!(sys, k, sys.L .* rand(sys.rng, 3), MCg.uniform_random_quaternion(sys.rng))
            MCg.update_rigid_ekmc_after_move!(ek, sys, ff, k)
            MCg.slm_moved!(S, sys, ff, k)
            MCg.lm_molecule_moved!(lm, sys, ff, k)
        else
            MCg.lm_total_rate!(lm, sys, ff, sys.β, ek.lam[s], ek.frac_u)
            MCg.lm_fire!(lm, sys, ff, ek, rand(sys.rng) * sum(lm.rates))
            MCg.slm_moved!(S, sys, ff, s)
            MCg.lm_rebuild!(lm, sys, ff)
        end
    end
    @test MCg.slm_check!(S, sys, ff) < 1e-9
    @test ek_drift(ek, sys, ff) < 1e-9
    @test Array(sys.recip.Sd) ≈ sys.recip.S rtol = 1e-14
end

@testset "CUDA backend, mixed precision (GROMACS convention): rows within the Float32 rounding bound" begin
    m, sys, _ = ewald_water(40, 12)
    ff = m.ff; N = sys.n_mol; R = sys.recip; Nk = MCg.n_kvectors(R)
    sysg = deepcopy(sys); MCg.set_recip_backend!(sysg.recip, :cuda; precision=:mixed)
    @test sysg.recip.Sd isa CuArray{ComplexF32}
    rd = zeros(N); rg = zeros(N); w = zeros(N)
    MCg.eval_row!(rd, w, sys, ff, 7); MCg.eval_row!(rg, w, sysg, ff, 7)
    bound = [γ32(2Nk + 4) * sum(R.c .* abs.(R.S[:, 7]) .* abs.(R.S[:, j])) for j in 1:N]
    @test all(abs.(rg .- rd) .<= bound)
    ek = MCg.init_rigid_ekmc(sysg, ff)
    S = MCg.SolventLocalMoves(sysg, ff, collect(1:N-1); kinds=[MCg.LM_T, MCg.LM_R], a=0.1, θmax=0.3, ν=1.0, νD=1.0)
    @test S.crSd isa CuArray{ComplexF32}
    for _ in 1:50
        k = rand(sysg.rng, 1:N-1)
        MCg.set_molecule!(sysg, k, sysg.L .* rand(sysg.rng, 3), MCg.uniform_random_quaternion(sysg.rng))
        MCg.update_rigid_ekmc_after_move!(ek, sysg, ff, k)
        MCg.slm_moved!(S, sysg, ff, k)
    end
    # worst-case drift of 50 single-precision increments: each ≤ γ Σ_k c_k |S_cand||ΔS|, |ΔS| ≤ 2 max|S|
    Smax = max(maximum(abs.(sysg.recip.S)), maximum(abs.(S.crS)))
    @test MCg.slm_check!(S, sysg, ff) <= 50 * γ32(2Nk + 4) * sum(sysg.recip.c) * 2Smax * Smax
end

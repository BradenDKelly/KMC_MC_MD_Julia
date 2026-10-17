# Cost per MC trial and per eKMC event (with solvent L-moves) for SPC/E: RF vs Ewald on the CPU backend and, when CUDA is
# loaded and functional, Ewald on the CUDA backend (DESIGN §13). Usage (GPU environment for the :cuda column):
#   julia -t 4 --project=/shared/kmc_polyatomic/envs/molsim_gpu dev/bench_ewald_backend.jl
using MolSim, Random, Printf
const M = MolSim.MC
has_cuda = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
cases = [(64, 4.5), (256, 4.5), (500, 10.0)]
println("# threads=$(Threads.nthreads()) cuda=$has_cuda host=$(gethostname())")
for (N, rc) in cases, (coul, be, prec) in ((:rf, :cpu, :double), (:ewald, :cpu, :double), (:ewald, :cpu, :mixed),
                                         (:ewald, :cuda, :double), (:ewald, :cuda, :mixed))
    be === :cuda && !has_cuda && continue
    ρ = 0.0334
    α = M.ewald_alpha_for(rc, 1e-5)
    m = M.spce_model(; α_per_angstrom=α, rc_angstrom=rc, coulomb=coul, epsilon_rf=Inf)
    L = cbrt(N / ρ) / m.σ_angstrom
    sys = M.RigidSystem([m.template], fill(1, N), L, m.ϵ_kelvin / 300.0, Xoshiro(3))
    M.random_rigid_system!(sys, m.ff)
    coul === :ewald && M.attach_ewald!(sys, m.ff; rtol=1e-5, backend=be, precision=prec)
    nk = coul === :ewald ? M.n_kvectors(sys.recip) : 0
    st = M.RigidMCStats()
    for _ in 1:2N; M.mc_translate!(sys, m.ff, rand(sys.rng, 1:N), 0.1, st); end
    t = @elapsed for _ in 1:10N; M.mc_translate!(sys, m.ff, rand(sys.rng, 1:N), 0.1, st); end
    M.run_rigid_ekmc!(sys, m.ff; events=500, sample_every=10, block_events=100, local_solvent=true, solvent_nu=1e-4)
    nev = N >= 500 ? 1000 : 3000
    te = @elapsed M.run_rigid_ekmc!(sys, m.ff; events=nev, sample_every=10, block_events=nev ÷ 4, local_solvent=true, solvent_nu=1e-4)
    @printf("%-6s %-5s %-6s N=%4d rc=%4.1f Å Nk=%5d  MC trial %8.1f µs   eKMC+solvent-L event %9.1f µs\n",
            coul, be, prec, N, rc, nk, 1e6 * t / (10N), 1e6 * te / nev)
end

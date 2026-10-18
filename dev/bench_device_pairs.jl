# Device pair kernel benchmark (DESIGN §15 G1): the pair work of one eKMC event with flexible solvent L-moves in a dense
# 200-ibuprofen liquid (LigParGen OPLS-AA, RF ε_rf = ∞, site cut-off 10 Å): the mover's row (N pairs), its K candidate
# rows (K·N) and the K-candidate columns of every mover against the moved molecule (K·N). CPU: eval_row! /
# pair_energy_virial loops at the Julia thread count; device: DevicePairs (:double, :mixed) on :cpu-KA (only with
# --ka) and on :cuda when CUDA is loaded and functional. Usage:
#   julia -t 4 --project=/shared/kmc_polyatomic/envs/molsim_gpu dev/bench_device_pairs.jl <nptc.gro> [K]
has_cuda = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
using MolSim, Random, Printf, Statistics
const M = MolSim.MC
gro = ARGS[1]; K = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
mi = M.spce_solute_model(:ibuprofen_lpg; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=10.0)
nat = parse(Int, strip(readlines(gro)[2])); N = nat ÷ 33
sys = M.RigidSystem([mi.templates[2]], fill(1, N), 1.0, mi.ϵ_kelvin / 400.0, Xoshiro(1))
M.load_gro!(sys, gro; σ_angstrom=mi.σ_angstrom)
ff = mi.ff
@printf("# N=%d molecules, %d sites, L=%.3f Å, rc=%.1f Å, K=%d, threads=%d, cuda=%s host=%s\n", N, sys.n_sites_total,
        sys.L * mi.σ_angstrom, ff.rc * mi.σ_angstrom, K, Threads.nthreads(), has_cuda, gethostname())
med(f, reps) = (f(); median([(t = time_ns(); f(); (time_ns() - t) / 1e3) for _ in 1:reps]))   # µs, first call excluded
ru = zeros(N); rw = zeros(N)
t_row = med(() -> M.eval_row!(ru, rw, sys, ff, rand(sys.rng, 1:N); recip=false), 30)
m0 = 1
colsum = zeros(Threads.nthreads() * 8)
function cpu_columns!(sys, ff, m, K)
    N = sys.n_mol
    Threads.@threads :static for c in 1:Threads.nthreads()
        acc = 0.0
        for ii in c:Threads.nthreads():N
            ii == m && continue
            for _ in 1:K
                acc += M.pair_energy_virial(sys, ff, ii, m)[1]
            end
        end
        colsum[c] = acc
    end
end
t_col = med(() -> cpu_columns!(sys, ff, m0, K), 10)
# pairs within range (for a work count)
npair = count(j -> j != m0 && M.pair_energy_virial(sys, ff, m0, j)[1] != 0.0, 1:N)
@printf("cpu    row %9.1f µs   K+1 rows %9.1f µs   columns (K·N) %9.1f µs   event pair work %9.1f µs  (%d of %d molecules interact with molecule 1)\n",
        t_row, (K + 1) * t_row, t_col, (K + 1) * t_row + t_col, npair, N - 1)
backends = has_cuda ? [:cuda] : Symbol[]
"--ka" in ARGS && pushfirst!(backends, :cpu)
for be in backends, prec in (:double, :mixed)
    D = M.DevicePairs(sys, ff; backend=be, precision=prec, max_poses=K * N, max_tasks=(K + 1) * N)
    # correctness spot check on the dense system: one row against eval_row!
    M.dp_stage_pose!(D, 1, sys, 7); M.dp_upload_poses!(D, 1); U, _ = M.dp_rows!(D, 1)
    M.eval_row!(ru, rw, sys, ff, 7; recip=false)
    dev = maximum(abs.(U .- ru)); scale = sum(abs.(ru))
    # rows: K+1 poses (stage + upload + kernel + download)
    ks = rand(sys.rng, 1:N, K + 1)
    t_rows = med(() -> (for (s, k) in enumerate(ks); M.dp_stage_pose!(D, s, sys, k); end; M.dp_upload_poses!(D, K + 1); M.dp_rows!(D, K + 1)), 30)
    # columns: a bank of K·N candidate poses (committed poses as geometric stand-ins), tasks (candidate of ii, m)
    for ii in 1:N, s in 1:K; M.dp_stage_pose!(D, (ii - 1) * K + s, sys, ii); end
    M.dp_upload_poses!(D, K * N)
    ps = [p for p in 1:K * N if (p - 1) ÷ K + 1 != m0]; js = fill(m0, length(ps))
    t_cols = med(() -> M.dp_tasks!(D, ps, js), 30)
    @printf("%-5s %-6s K+1 rows %9.1f µs   columns (K·N) %9.1f µs   event pair work %9.1f µs   speed-up vs cpu %.1f×   row check max|Δ| = %.2e (Σ|u| = %.2e)\n",
            be, prec, t_rows, t_cols, t_rows + t_cols, ((K + 1) * t_row + t_col) / (t_rows + t_cols), dev, scale)
end

#=
Equal-CPU benchmark B1 (JOURNAL 68): pure SPC/E (RF, site cut-off 10 Å, 100 kT cap), N = 500, 300 K, 1 g/cm³.
Every method starts from the same MC-melted configuration of its seed (made once by `prep`) and runs on ONE thread for the
same wall budget; the cost-per-accuracy table is made by dev/benchmark_water_table.py.

Usage: julia -t 1 --project=. dev/benchmark_water.jl prep  seed outdir              # quench + 2000 MC sweeps → start_seed<s>.jls
       julia -t 1 --project=. dev/benchmark_water.jl <method> seed wall_seconds outdir
  method = mc     Metropolis (translation + rotation, δ = 0.3 Å, δθ = 0.3 rad), Widom insertion of a water every 10 sweeps
           stock  eKMC, relocations only
           local  eKMC + rejection-free local moves of every water (SolventLocalMoves, ν = exp(β·(−29.5 kJ/mol)))
Output line: method seed wall cpu_s n_moves  U/N ± (block)  P ± (block)  mu_water ± (block), plus the block means.
=#
using MolSim, Random, Printf, Dates, Serialization, StaticArrays
const MCm = MolSim.MC

mode = ARGS[1]; seed = parse(Int, ARGS[2])
outdir = mode == "prep" ? ARGS[3] : ARGS[4]
mkpath(outdir)
m = MCm.spce_solute_model(:methanol_flex; rc_angstrom=10.0, cutoff=:site, coulomb=:rf, epsilon_rf=Inf)
ff = first(MCm.kT_overlap_cap(m.ff, m.templates, 300.0 / m.ϵ_kelvin; U_kT=100.0))
T = 300.0 / m.ϵ_kelvin; β = 1 / T; N = parse(Int, get(ENV, "BENCH_N", "500"))   # 500 for the benchmark; smaller only for smoke tests
ρ_A3 = 1000.0 * 1e3 / 18.0153 * 6.02214129e23 / 1e30
L = cbrt(N / (ρ_A3 * m.σ_angstrom^3))
kJ = m.ϵ_kelvin * 1.3806488e-23 * 6.02214129e23 / 1000
atm = m.ϵ_kelvin * 1.3806488e-23 / (m.σ_angstrom * 1e-10)^3 / 101325.0
δ = 0.1 / m.σ_angstrom * 3.0; δθ = 0.3
startfile = joinpath(outdir, "start_seed$(seed).jls")
nblk = 25

if mode == "prep"
    sys = MCm.RigidSystem(m.templates, fill(1, N), L, β, Xoshiro(seed))
    MCm.random_rigid_system!(sys, ff); MCm.minimize_rigid!(sys, ff; nsteps=500)
    st = MCm.RigidMCStats()
    for _ in 1:2000, _ in 1:N
        MCm.mc_translate!(sys, ff, rand(sys.rng, 1:N), δ, st); MCm.mc_rotate!(sys, ff, rand(sys.rng, 1:N), δθ, st)
    end
    open(io -> serialize(io, sys), startfile, "w")
    println("prep seed $seed: wrote $startfile")
    exit(0)
end

method = mode; wall = parse(Float64, ARGS[3])
method in ("mc", "stock", "local") || error("method must be mc, stock or local")
sys = open(deserialize, startfile)
sys.rng = Xoshiro(1000 * seed + (method == "mc" ? 1 : method == "stock" ? 2 : 3))   # independent streams per method
const_E = MCm.wolf_constant_energy(sys, ff)
UkJ(u) = (u * N + const_E) / N * kJ
out = joinpath(outdir, "bench_$(method)_seed$(seed).txt")
io = open(out, "w")
println(io, "# B1 water benchmark  $(Dates.now())  host=$(gethostname())  method=$method seed=$seed wall=$wall threads=$(Threads.nthreads())")
t0 = time()
if method == "mc"
    # warm-up (compilation, excluded from the cost for every method) on a copy, then a timing pilot (part of the cost)
    MCm.run_rigid_mc!(deepcopy(sys), ff; sweeps=2, δ=δ, δθ=δθ, sample_every=1, block_size=1, widom_template=1, widom_per_sample=50)
    t0 = time()
    tp = time(); MCm.run_rigid_mc!(sys, ff; sweeps=10, δ=δ, δθ=δθ, sample_every=1, block_size=5); tps = (time() - tp) / 10
    sweeps = max(nblk * 2, floor(Int, (wall - (time() - t0)) / (tps * 1.10)))   # 10 % margin for the Widom insertions
    mc = MCm.run_rigid_mc!(sys, ff; sweeps=sweeps, δ=δ, δθ=δθ, sample_every=1, block_size=max(1, sweeps ÷ nblk),
                           widom_template=1, widom_per_sample=50)
    cpu = time() - t0
    w = MCm.mean(mc.widom)
    @printf(io, "RESULT method=mc seed=%d cpu_s=%.0f moves=%d  U=%.4f ± %.4f  P=%.1f ± %.1f  mu_w=%.3f ± %.3f\n", seed, cpu, 2 * sweeps * N,
            UkJ(MCm.mean(mc.U)), MCm.stderr(mc.U) * kJ, MCm.mean(mc.P) * atm, MCm.stderr(mc.P) * atm, -T * log(w) * kJ, T * MCm.stderr(mc.widom) / w * kJ)
    println(io, "# U block means (kJ/mol): ", join(round.(UkJ.(mc.U.block_means); digits=3), " "))
    println(io, "# P block means (atm): ", join(round.(mc.P.block_means .* atm; digits=1), " "))
else
    # timing pilot (part of the cost) → number of events and blocks that fill the wall budget
    νs = exp(β * (-29.5 / kJ))
    mk(ev, be) = MCm.init_rigid_ekmc_run(sys, ff; events=ev, sample_every=10, block_events=be, species_mu=false,
                                         mc_δ=δ, mc_δθ=δθ, local_solvent=(method == "local"), solvent_nu=νs)
    s0 = deepcopy(sys); rw = mk(2_000, 100); MCm.advance_rigid_ekmc!(rw, s0, ff)    # warm-up on a copy (compilation, not costed)
    t0 = time()
    rp = mk(20_000, 1000); tp = time(); MCm.advance_rigid_ekmc!(rp, sys, ff); tpe = (time() - tp) / 20_000
    ev = max(nblk * 1000, floor(Int, (wall - (time() - t0)) / (tpe * 1.05)))
    r = mk(ev, max(100, ev ÷ nblk))
    MCm.advance_rigid_ekmc!(r, sys, ff; wall_seconds=wall - (time() - t0))
    cpu = time() - t0
    res = MCm.rigid_ekmc_results(r)
    nloc = r.slmoves === nothing ? 0 : sum(r.slmoves.n_move)
    @printf(io, "RESULT method=%s seed=%d cpu_s=%.0f moves=%d events=%d  U=%.4f ± %.4f  P=%.1f ± %.1f  mu_w=%.3f ± %.3f\n", method, seed, cpu,
            r.n_reloc + nloc, r.ev, UkJ(MCm.ratio_mean(res.U)), MCm.ratio_jackknife_stderr(res.U) * kJ, MCm.ratio_mean(res.P) * atm,
            MCm.ratio_jackknife_stderr(res.P) * atm, T * log(MCm.ratio_mean(res.RN)) * kJ, T * MCm.ratio_jackknife_stderr(res.RN; f=log) * kJ)
    println(io, "# relocations=$(r.n_reloc) local_moves=$nloc events=$(r.ev) of planned $ev")
    println(io, "# U block means (kJ/mol): ", join(round.(UkJ.(res.U.block_means); digits=3), " "))
    println(io, "# P block means (atm): ", join(round.(res.P.block_means .* atm; digits=1), " "))
end
close(io)
println(read(out, String))

# eKMC with device pair rows (DESIGN §15 G2, JOURNAL 122): wall time per event of the full eKMC loop with solvent L-moves on
# the host (Julia threads) and with `device = :cuda` (:double, :mixed) on the same node, plus the run's own debug rebuilds
# (host pair_energy_virial) as the correctness check on the GPU. Systems: SPC/E RF (N waters at 1000 kg/m³, rc 10 Å) and the
# dense 200-ibuprofen liquid (flexible, LigParGen, RF; gro of the GROMACS NPT box).
# Usage: julia -t 4 --project=<env with CUDA> dev/bench_ekmc_device.jl <water|ibuprofen> <N or gro> <events> [modes...]
#   modes: host cuda_double cuda_mixed (default all)
has_cuda = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
using MolSim, Random, Printf
const M = MolSim.MC
which = ARGS[1]; spec = ARGS[2]; events = parse(Int, ARGS[3])
modes = length(ARGS) >= 4 ? ARGS[4:end] : ["host", "cuda_double", "cuda_mixed"]

function build(which, spec, seed)
    if which == "water"
        N = parse(Int, spec)
        m = M.spce_model(; rc_angstrom=10.0, coulomb=:rf, epsilon_rf=Inf, cutoff=:site)
        ff = first(M.kT_overlap_cap(m.ff, [m.template], 300.0 / m.ϵ_kelvin; U_kT=100.0))
        sys = M.RigidSystem([m.template], fill(1, N), cbrt(N / 0.0334) / m.σ_angstrom, m.ϵ_kelvin / 300.0, Xoshiro(seed))
        M.random_rigid_system!(sys, ff); M.minimize_rigid!(sys, ff; nsteps=300)
        return sys, ff, nothing, false
    else
        mi = M.spce_solute_model(:ibuprofen_lpg; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=10.0)
        nat = parse(Int, strip(readlines(spec)[2])); N = nat ÷ 33
        t = mi.templates[2]
        sys = M.RigidSystem([t], fill(1, N), 1.0, mi.ϵ_kelvin / 400.0, Xoshiro(seed))
        M.load_gro!(sys, spec; σ_angstrom=mi.σ_angstrom)
        ff = mi.ff
        res = M.build_reservoir_ekmc([t], 1, ff, sys.β, Xoshiro(seed + 1); M=200, burnin_sweeps=50, stride_sweeps=2, δ=0.03, torsion=true)
        return sys, ff, [res], true
    end
end

@printf("# %s %s, events %d, threads %d, cuda %s, host %s\n", which, spec, events, Threads.nthreads(), has_cuda, gethostname())
for mode in modes
    (startswith(mode, "cuda") && !has_cuda) && (println("# skip $mode (no CUDA)"); continue)
    sys, ff, res, flex = build(which, spec, 1)
    dev = mode == "host" ? :none : startswith(mode, "ka") ? :cpu : :cuda    # ka_double: KernelAbstractions CPU backend
    prec = endswith(mode, "mixed") ? :mixed : :double
    kw = (; events=events, sample_every=50, block_events=max(1, events ÷ 5), reservoirs=res, local_solvent=true,
          solvent_flexible=flex, solvent_nu=1e-3, mc_δ=0.05, mc_δθ=0.15, solvent_site_a=0.01, solvent_φ=0.5,
          debug_check_every=prec === :double ? max(1, events ÷ 4) : 0, device=dev, device_precision=prec)
    r = M.init_rigid_ekmc_run(sys, ff; kw...)
    t0 = time(); M.advance_rigid_ekmc!(r, sys, ff; wall_seconds=Inf); t = time() - t0
    @printf("RESULT %-12s %d events  %.3f s  %.1f µs/event  N=%d sites=%d\n", mode, r.ev, t, 1e6 * t / max(r.ev, 1), sys.n_mol,
            sys.n_sites_total)
    flush(stdout)
end

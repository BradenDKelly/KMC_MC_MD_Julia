# eKMC with device pair rows (DESIGN §15 G2, JOURNAL 122): wall time per event of the full eKMC loop with solvent L-moves on
# the host (Julia threads) and with `device = :cuda` (:double, :mixed) on the same node, plus the run's own debug rebuilds
# (host pair_energy_virial) as the correctness check on the GPU. Systems: SPC/E RF (N waters at 1000 kg/m³, rc 10 Å) and the
# dense 200-ibuprofen liquid (flexible, LigParGen, RF; gro of the GROMACS NPT box).
# Usage: julia -t 4 --project=<env with CUDA> dev/bench_ekmc_device.jl <water|ibuprofen|mixEW_pme> <N or gro> <events> [modes...]
#   modes: host cuda_double cuda_mixed (default all); mixEW_pme: 1:1 ethanol-water (GROMACS NVT box, gro path) with the SPME
#   mesh path — host: everything on the CPU; cuda_*: pair rows on the device AND the mesh gathers (pme_attach_device!)
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
    elseif which == "mixEW_pme"
        m = M.spce_mixture_model([:ethanol_flex]; rc_angstrom=10.0, cutoff=:site, coulomb=:ewald, α_per_angstrom=M.ewald_alpha_for(10.0, 1e-5))
        T = 300.0 / m.ϵ_kelvin
        ff = first(M.kT_overlap_cap(m.ff, m.templates, T; U_kT=100.0))
        lines = readlines(spec); nat = parse(Int, strip(lines[2])); box = parse.(Float64, split(lines[end])[1:3])
        tmpl = vcat(2, 1, fill(2, 199), fill(1, 199)); L = box[1] * 10 / m.σ_angstrom
        X = zeros(3, nat); for k in 1:nat; l = lines[2+k]; X[:, k] .= parse.(Float64, (l[21:28], l[29:36], l[37:44])) .* (10 / m.σ_angstrom); end
        sys = M.RigidSystem(m.templates, tmpl, L, 1 / T, Xoshiro(seed)); s = 0
        for i in eachindex(tmpl)
            t = m.templates[tmpl[i]]; Y = X[:, s+1:s+t.n_sites]; s += t.n_sites
            for a in 2:t.n_sites, d in 1:3; Y[d, a] = Y[d, 1] + M._min_image(Y[d, a] - Y[d, 1], L); end
            M.isflexible(t) ? (M.set_conformation!(sys, i, Y); M.set_molecule!(sys, i, (Y * t.site_mass) ./ sum(t.site_mass), sys.quat[:, i])) : M.molecule_from_sites!(sys, i, Y)
        end
        M.attach_pme!(sys, ff; spacing=1.2 / m.σ_angstrom, order=4)
        res = M.build_reservoir_ekmc(m.templates, 2, ff, 1 / T, Xoshiro(seed + 1); M=200, burnin_sweeps=50, stride_sweeps=2, δ=0.03, torsion=true)
        return sys, ff, [nothing, res], true
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
    (dev !== :none && sys.recip isa M.PMERecip) && M.pme_attach_device!(sys.recip, sys; backend=dev)   # mesh gathers too
    t0 = time(); M.advance_rigid_ekmc!(r, sys, ff; wall_seconds=Inf); t = time() - t0
    @printf("RESULT %-12s %d events  %.3f s  %.1f µs/event  N=%d sites=%d\n", mode, r.ev, t, 1e6 * t / max(r.ev, 1), sys.n_mol,
            sys.n_sites_total)
    flush(stdout)
end

# MC equilibration probe (JOURNAL 2026-10-22): the MC references of test_ewald_ekmc_slow.jl / dev/ewald_slow_seeds*.jl
# (300 sweeps from a random start) and of test_flexible_solvent_slow.jl / dev/ethanol_liquid_seeds*.jl (compression +
# 1000 sweeps) showed relaxation transients inside the production. This runs the same MC from the same start for a long
# time and prints the block series, for automated equilibration detection (Chodera, J. Chem. Theory Comput. 12 (2016) 1799).
# Usage: julia --project=. dev/mc_equil_probe.jl <water_ewald|water_pme|ethanol_rf> <seed> <sweeps> [N rc_Å]
using MolSim, Random, Printf
const MCs = MolSim.MC
sysname = ARGS[1]; seed = parse(Int, ARGS[2]); sweeps = parse(Int, ARGS[3])
if startswith(sysname, "water")
    recip = sysname == "water_pme" ? :pme : :ewald
    N = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 48; rc = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 4.5; ρ = 0.0334
    α = MCs.ewald_alpha_for(rc, 1e-5)
    m = MCs.spce_model(; α_per_angstrom=α, rc_angstrom=rc, coulomb=:ewald)
    sys = MCs.RigidSystem([m.template], fill(1, N), cbrt(N / ρ) / m.σ_angstrom, m.ϵ_kelvin / 300.0, Xoshiro(seed))
    MCs.random_rigid_system!(sys, m.ff)
    recip === :ewald ? MCs.attach_ewald!(sys, m.ff; rtol=1e-5) : MCs.attach_pme!(sys, m.ff; spacing=1.2 / m.σ_angstrom, order=4)
    st = MCs.RigidMCStats()
    for _ in 1:300, _ in 1:N
        MCs.mc_translate!(sys, m.ff, rand(sys.rng, 1:N), 0.1, st); MCs.mc_rotate!(sys, m.ff, rand(sys.rng, 1:N), 0.3, st)
    end
    mc = MCs.run_rigid_mc!(sys, m.ff; sweeps=sweeps, δ=0.1, δθ=0.3, sample_every=1, block_size=50)
    kJ = 1.0
else
    function compress_to!(sys, ff, L; steps=12, nsteps=150)
        f = (L / sys.L)^(1 / steps)
        for _ in 1:steps
            sys.L *= f
            for i in 1:sys.n_mol; MCs.set_molecule!(sys, i, sys.com[:, i] .* f, sys.quat[:, i]); end
            MCs.minimize_rigid!(sys, ff; nsteps=nsteps)
        end
        sys.L = L
    end
    N = 64; T = 298.0
    m = MCs.spce_solute_model(:ethanol_flex; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=9.0, α_per_angstrom=0.20)
    t = m.templates[2]; ff = m.ff; β = m.ϵ_kelvin / T; kJ = m.ϵ_kelvin * 0.0083144626
    ρ = 0.785 / 46.07 * 6.02214076e23 / 1e24
    L = cbrt(N / ρ) / m.σ_angstrom
    sys = MCs.RigidSystem([t], fill(1, N), 2L, β, Xoshiro(100 + seed)); MCs.random_rigid_system!(sys, ff)
    compress_to!(sys, ff, L)
    res = MCs.build_reservoir([t], 1, ff, β, Xoshiro(200 + seed); M=4000, burnin_sweeps=500, stride_sweeps=20, δ=0.02, torsion=true)
    δ, δθ, δs, φ = 0.05, 0.15, 0.01, 0.5
    MCs.run_rigid_mc!(sys, ff; sweeps=1000, δ=δ, δθ=δθ, sample_every=100, block_size=5, reservoirs=[res], δ_site=δs, torsion_φ=φ)
    mc = MCs.run_rigid_mc!(sys, ff; sweeps=sweeps, δ=δ, δθ=δθ, sample_every=2, block_size=50, reservoirs=[res], δ_site=δs, torsion_φ=φ)
    println("MCBLOCKS Uintra ", join((@sprintf("%.4f", x * kJ) for x in mc.U_intra.block_means), " "))
end
println("MCBLOCKS U ", join((@sprintf("%.4f", x * kJ) for x in mc.U.block_means), " "))
@printf("PROBE %s seed=%d sweeps=%d block_sweeps=%d mean U %.4f\n", sysname, seed, sweeps, startswith(sysname, "water") ? 50 : 100, MCs.mean(mc.U) * kJ)

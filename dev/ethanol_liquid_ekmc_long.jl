# Convergence trace of eKMC (flexible solvent L-moves) for the ethanol liquid seed study (dev/ethanol_liquid_seeds.jl, same
# start for the same seed), f × 3e6 events, printing every 3e6 events: running ⟨U⟩, ⟨U_intra⟩, ⟨P⟩, kMC time, relocations,
# capped (fast-path) landings, L-moves. Decides between under-sampling (drift towards MC) and bias (stays off).
# Usage: julia -t 4 --project=. dev/ethanol_liquid_ekmc_long.jl <seed> <f>
using MolSim, Random, Printf
const MCs = MolSim.MC
seed = parse(Int, ARGS[1]); f = parse(Int, ARGS[2])
function compress_to!(sys, ff, L; steps=12, nsteps=150)
    fac = (L / sys.L)^(1 / steps)
    for _ in 1:steps
        sys.L *= fac
        for i in 1:sys.n_mol; MCs.set_molecule!(sys, i, sys.com[:, i] .* fac, sys.quat[:, i]); end
        MCs.minimize_rigid!(sys, ff; nsteps=nsteps)
    end
    sys.L = L
end
N = 64; T = 298.0
m = MCs.spce_solute_model(:ethanol_flex; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=9.0)
t = m.templates[2]; ff = m.ff; β = m.ϵ_kelvin / T; kJ = m.ϵ_kelvin * 0.0083144626
ρ = 0.785 / 46.07 * 6.02214076e23 / 1e24
L = cbrt(N / ρ) / m.σ_angstrom
sys = MCs.RigidSystem([t], fill(1, N), 2L, β, Xoshiro(100 + seed)); MCs.random_rigid_system!(sys, ff)
compress_to!(sys, ff, L)
res = MCs.build_reservoir([t], 1, ff, β, Xoshiro(200 + seed); M=4000, burnin_sweeps=500, stride_sweeps=20, δ=0.02)
δ, δθ, δs, φ = 0.05, 0.15, 0.01, 0.5
MCs.run_rigid_mc!(sys, ff; sweeps=1000, δ=δ, δθ=δθ, sample_every=100, block_size=5, reservoirs=[res], δ_site=δs, torsion_φ=φ)
chunk = 3_000_000; events = f * chunk
sysE = deepcopy(sys)
r = MCs.init_rigid_ekmc_run(sysE, ff; events=events, sample_every=20, block_events=120_000, burnin_events=300_000,
        reservoirs=[res], local_solvent=true, solvent_flexible=true, solvent_nu=1e-3, mc_δ=δ, mc_δθ=δθ, solvent_site_a=δs, solvent_φ=φ)
for c in 1:f
    MCs.advance_rigid_ekmc!(r, sysE, ff; until=c * chunk)
    ek = MCs.rigid_ekmc_results(r)
    @printf("TRACE seed=%d events=%.1e  U %.4f ± %.4f  Uintra %.4f ± %.4f  P %.4f ± %.4f  t_kMC %.4e  relocations %d  capped landings %d  L-moves %s\n",
            seed, c * chunk, MCs.mean(ek.U) * kJ, MCs.stderr(ek.U) * kJ, MCs.mean(ek.U_intra) * kJ, MCs.stderr(ek.U_intra) * kJ,
            MCs.mean(ek.P), MCs.stderr(ek.P), ek.t_total, ek.relocations, ek.fast_events, string(r.slmoves.n_move))
    flush(stdout)
end

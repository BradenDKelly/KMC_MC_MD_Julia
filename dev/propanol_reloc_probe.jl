# Which molecule takes the relocation events in the propanol eKMC diagnostic (dev/propanol_torsion_diag.jl ekmc: 199 999 of
# 200 000 events were relocations, 1 solvent L-move)? Same set-up; advances in chunks and reports, per chunk, the molecules that
# moved, the top mobilities m_i with their u_i, the relocation total R and the L-move total W.
# Usage: julia -t 2 --project=. dev/propanol_reloc_probe.jl <seed> <chunks> <events per chunk>
using MolSim, Random, Printf
const M = MolSim.MC
seed = parse(Int, ARGS[1]); nch = parse(Int, ARGS[2]); per = parse(Int, ARGS[3]); φ = 0.3
m = M.spce_solute_model(:propanol_flex; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=10.0)
T = 300.0 / m.ϵ_kelvin
ff = first(M.kT_overlap_cap(m.ff, m.templates, T; U_kT=100.0))
N = 500; L = cbrt(N / (1000.0 * 1e3 / 18.01528 * 6.02214076e23 / 1e30)) / m.σ_angstrom
sys = M.RigidSystem(m.templates, vcat(fill(1, N), 2), L, 1 / T, Xoshiro(seed))
M.random_rigid_system!(sys, ff)
res = M.build_reservoir_ekmc(m.templates, 2, ff, 1 / T, Xoshiro(seed + 7919); M=4000, burnin_sweeps=500, stride_sweeps=20, δ=0.03, torsion=true)
isol = N + 1
M._set_body!(sys, isol, M.draw_conformer(res, sys.rng))
M.minimize_rigid!(sys, ff; nsteps=500)
ek0 = M.init_rigid_ekmc(sys, ff); Rbar = ek0.R / sys.n_mol
kJ = m.ϵ_kelvin * 1.380649e-23 * 6.02214076e23 / 1000
@printf("# start: R = %.4g, Rbar = ν = %.4g; top m: %s\n", ek0.R, Rbar,
        join([@sprintf("%d (u=%.1f kJ/mol, m=%.3g)", i, ek0.phi[i] * kJ, ek0.m[i]) for i in partialsortperm(ek0.m, 1:5; rev=true)], ", "))
r = M.init_rigid_ekmc_run(sys, ff; events=nch * per, sample_every=100, block_events=max(1000, per), reservoirs=[nothing, res],
                          burnin_events=0, local_solvent=true, solvent_flexible=true, solvent_nu=Rbar,
                          solvent_φ=φ, solvent_site_a=0.03 / m.σ_angstrom, mc_δ=0.3 / m.σ_angstrom, mc_δθ=0.3)
for c in 1:nch
    com0 = copy(sys.com)
    rl0 = r.n_reloc; lm0 = sum(r.slmoves.n_move)
    M.advance_rigid_ekmc!(r, sys, ff; until=c * per)
    moved = findall(i -> any(abs.(sys.com[:, i] .- com0[:, i]) .> 1e-12), 1:sys.n_mol)
    ek = r.ek
    top = partialsortperm(ek.m, 1:3; rev=true)
    @printf("chunk %d ev=%d: relocations +%d, L-moves +%d, moved %d molecules %s; R = %.4g, W = %.4g; top m: %s; solute u = %.1f kJ/mol\n",
            c, r.ev, r.n_reloc - rl0, sum(r.slmoves.n_move) - lm0, length(moved), string(first(moved, 6)), ek.R, r.slmoves.W,
            join([@sprintf("%d (u=%.1f, m=%.3g)", i, ek.phi[i] * kJ, ek.m[i]) for i in top], ", "), ek.phi[isol] * kJ)
end

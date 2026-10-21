# Propanol torsions in the COUPLED state (JOURNAL 123): Metropolis MC with site + torsion moves (the yardstick) and eKMC with
# the solute's rejection-free local moves (T R C S D slots) at a two-rung ladder λ = (0.85, 1) without the λ = 0 hub, both
# tracking the three dihedrals of 1-propanol (H2-C1-C5-C8, C1-C5-C8-O11, C5-C8-O11-H12) in 500 SPC/E (RF, site cut-off
# 10 Å, the production model), against GROMACS (gromacs_fe/propanol_flex_rf/dihref, λ_00 = fully coupled).
# Usage: julia -t 4 --project=. dev/propanol_torsion_diag.jl <mc|ekmc> <seed> <sweeps|events> [φmax rad (default 0.3)]
using MolSim, Random, Printf
const M = MolSim.MC
mode = ARGS[1]; seed = parse(Int, ARGS[2]); n = parse(Int, ARGS[3]); φ = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.3
m = M.spce_solute_model(:propanol_flex; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=10.0)
T = 300.0 / m.ϵ_kelvin
ff = first(M.kT_overlap_cap(m.ff, m.templates, T; U_kT=100.0))
N = 500; L = cbrt(N / (1000.0 * 1e3 / 18.01528 * 6.02214076e23 / 1e30)) / m.σ_angstrom   # 1000 kg/m³ in molecules/Å³
sys = M.RigidSystem(m.templates, vcat(fill(1, N), 2), L, 1 / T, Xoshiro(seed))
M.random_rigid_system!(sys, ff)
res = M.build_reservoir_ekmc(m.templates, 2, ff, 1 / T, Xoshiro(seed + 7919); M=4000, burnin_sweeps=500, stride_sweeps=20, δ=0.03, torsion=true)
isol = N + 1
M._set_body!(sys, isol, M.draw_conformer(res, sys.rng))
M.minimize_rigid!(sys, ff; nsteps=500)
dih = [(2, 1, 5, 8), (1, 5, 8, 11), (5, 8, 11, 12)]
function dihedral(X, a, b, c, d)
    b1 = X[:, b] - X[:, a]; b2 = X[:, c] - X[:, b]; b3 = X[:, d] - X[:, c]
    n1 = [b1[2]b2[3] - b1[3]b2[2], b1[3]b2[1] - b1[1]b2[3], b1[1]b2[2] - b1[2]b2[1]]
    n2 = [b2[2]b3[3] - b2[3]b3[2], b2[3]b3[1] - b2[1]b3[3], b2[1]b3[2] - b2[2]b3[1]]
    m1 = [n1[2]b2[3] - n1[3]b2[2], n1[3]b2[1] - n1[1]b2[3], n1[1]b2[2] - n1[2]b2[1]] ./ sqrt(sum(abs2, b2))
    return atan(sum(m1 .* n2), sum(n1 .* n2))
end
wells(x) = abs(x) > 2π / 3 ? 1 : (x > 0 ? 2 : 3)
pop = zeros(3, 3); changes = zeros(Int, 3); last = zeros(Int, 3); wsum = 0.0
function sample!(w)
    X = sys.body[isol]
    for (k, d) in enumerate(dih)
        c = wells(dihedral(X, d...))
        pop[k, c] += w
        last[k] != 0 && c != last[k] && (changes[k] += 1)
        last[k] = c
    end
    global wsum += w
end
t0 = time()
if mode == "mc"
    st = M.RigidMCStats()
    M.run_rigid_mc!(sys, ff; sweeps=200, δ=0.3 / m.σ_angstrom, δθ=0.3, reservoirs=[nothing, res], δ_site=0.03, torsion_φ=φ)
    for s in 1:n
        M.run_rigid_mc!(sys, ff; sweeps=1, δ=0.3 / m.σ_angstrom, δθ=0.3, reservoirs=[nothing, res], δ_site=0.03, torsion_φ=φ)
        sample!(1.0)
    end
else
    # ν = ⟨R⟩ of the relaxed liquid from Metropolis samples, as the production drivers (validate_spce_solute.jl R_mc,
    # mixture_ekmc.jl). The R of the minimised start (a quenched state, ~1e-12) starved the L-moves: one relocation then
    # bounced through overlaps for ~1e5 events (dev/propanol_reloc_probe.jl, JOURNAL 2026-10-20).
    mc = M.run_rigid_mc!(sys, ff; sweeps=500, δ=0.3 / m.σ_angstrom, δθ=0.3, reservoirs=[nothing, res], δ_site=0.03,
                         torsion_φ=0.3, sample_every=1, block_size=50, R_every=10)
    Rbar = M.mean(mc.R)
    @printf("# ν = <R>_MC = %.4g (500 MC sweeps)\n", Rbar)
    # no ladder: the solute is an ordinary flexible molecule with the solvent-style rejection-free moves (T, R, site,
    # torsion slots, half-angle φ), every water with T/R slots — the move set validated on liquid ethanol (JOURNAL 117)
    r = M.init_rigid_ekmc_run(sys, ff; events=n, sample_every=100, block_events=max(1000, n ÷ 20), reservoirs=[nothing, res],
                              burnin_events=n ÷ 10, local_solvent=true, solvent_flexible=true, solvent_nu=Rbar,
                              solvent_φ=φ, solvent_site_a=0.03 / m.σ_angstrom, mc_δ=0.3 / m.σ_angstrom, mc_δθ=0.3,
                              track_dihedrals=[(2, 1, 5, 8), (1, 5, 8, 11), (5, 8, 11, 12)], track_molecule=isol)
    M.advance_rigid_ekmc!(r, sys, ff; wall_seconds=Inf)
    for k in 1:3, b in 1:M.DIH_BINS
        c = -π + (b - 0.5) * 2π / M.DIH_BINS
        pop[k, wells(c)] += r.dih_hist[1][k][b]
    end
    wsum = sum(pop[1, :])
    @printf("# eKMC: solvent L-moves fired per kind (T R C S D) %s; solute is mover %d\n", string(r.slmoves.n_move), r.slmoves.pos[isol])
    @printf("# eKMC event breakdown: events %d, relocations %d, overlap bursts %d, solvent L-moves %d, refreshes %d\n", r.ev, r.n_reloc, r.n_bursts, sum(r.slmoves.n_move), r.slmoves.n_refresh)
    changes .= -1                                         # eKMC mode samples histograms only; transitions are not counted
end
@printf("RESULT %s seed=%d n=%d φmax=%.3f time=%.0fs  ", mode, seed, n, φ, time() - t0)
for k in 1:3
    @printf("d%d t/g+/g- %.3f/%.3f/%.3f (%d changes)  ", k, (pop[k, :] ./ sum(pop[k, :]))..., changes[k])
end
println()

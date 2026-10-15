#=
Torsion populations of the vacuum conformer pools (JOURNAL 78): for each RB dihedral with heavy central atoms, the trans
fraction (|φ| > 120°) in the pool, in its two halves, and the number of trans↔gauche flips along the pool sequence (the
pool is a time series of the vacuum chain), for the pool construction and seeds the solute driver uses.
Usage: julia --project=. dev/pool_torsions.jl solute mode seed1 [seed2 …]     (mode = mc | ekmc)
=#
using MolSim, Random, Printf, LinearAlgebra
const MCm = MolSim.MC
sol = Symbol(ARGS[1]); mode = ARGS[2]; seeds = parse.(Int, ARGS[3:end])
m = MCm.spce_solute_model(sol; rc_angstrom=10.0, cutoff=:site, coulomb=:rf, epsilon_rf=Inf)
ff = first(MCm.kT_overlap_cap(m.ff, m.templates, 300.0 / m.ϵ_kelvin; U_kT=100.0))
T = 300.0 / m.ϵ_kelvin; kJ = m.ϵ_kelvin * 1.3806488e-23 * 6.02214129e23 / 1000
t = m.templates[2]; bt = t.bonded; mass = t.site_mass
el(i) = mass[i] > 15.5 ? "O" : mass[i] > 11.5 ? "C" : "H"
function dihed(X, i, j, k, l)
    b1 = X[:, j] - X[:, i]; b2 = X[:, k] - X[:, j]; b3 = X[:, l] - X[:, k]
    n1 = cross(b1, b2); n2 = cross(b2, b3); m1 = cross(n1, b2 / norm(b2))
    return atan(dot(m1, n2), dot(n1, n2))
end
# one representative per central bond: heavy i and l preferred, else the hydroxyl H
dl = [d for d in bt.dihedrals if el(d.j) != "H" && el(d.k) != "H" && (el(d.i) != "H" || el(d.j) == "O") && (el(d.l) != "H" || el(d.k) == "O")]
for seed in seeds
    t0 = time()
    res = mode == "mc" ?
        MCm.build_reservoir(m.templates, 2, ff, 1 / T, Xoshiro(seed + 7919); M=20000, burnin_sweeps=2000, stride_sweeps=20, δ=0.03) :
        MCm.build_reservoir_ekmc(m.templates, 2, ff, 1 / T, Xoshiro(seed + 7919); M=20000, burnin_sweeps=2000, stride_sweeps=20, δ=0.03)
    M = length(res.confs)
    @printf("%s %s seed %d (%.0f s): <U_intra> = %.3f kJ/mol\n", sol, mode, seed, time() - t0, sum(res.U_intra) / M * kJ)
    for d in dl
        φ = [dihed(c, d.i, d.j, d.k, d.l) for c in res.confs]
        tr = abs.(φ) .> 2π / 3
        flips = count(i -> tr[i] != tr[i-1], 2:M)
        gp = count(i -> !tr[i] && φ[i] > 0, 1:M) / M
        @printf("   %s%d-%s%d-%s%d-%s%d  trans %.3f (halves %.3f / %.3f)  g+ %.3f  flips %d\n", el(d.i), d.i, el(d.j), d.j, el(d.k), d.k, el(d.l), d.l,
                count(tr) / M, count(tr[1:M÷2]) / (M ÷ 2), count(tr[M÷2+1:end]) / (M - M ÷ 2), gp, flips)
    end
end

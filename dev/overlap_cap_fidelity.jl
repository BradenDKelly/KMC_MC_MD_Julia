# Fidelity of the overlap cap on thermal configurations (JOURNAL entry 27, DESIGN §9): for frames of the GROMACS
# fully coupled trajectory (gromacs_fe/contacts/<sys>/traj.gro, same RF Hamiltonian), count molecule pairs whose
# capped energy differs from the bare energy, for the kT cap and the legacy cap.
# Usage: julia --project=. dev/overlap_cap_fidelity.jl traj.gro solute(:methanol|:methane) [stride]
using MolSim, Random, Printf
const MCm = MolSim.MC
path = ARGS[1]; solute = Symbol(ARGS[2]); stride = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
m = MCm.spce_solute_model(solute; coulomb=:rf, epsilon_rf=Inf, cutoff=:site)
T = 300.0 / m.ϵ_kelvin
ffk, _ = MCm.kT_overlap_cap(m.ff, m.templates, T)
ffu = MCm.uncapped(m.ff)
io = open(path); nfr = 0; ndiff_k = 0; ndiff_l = 0; npairs = 0; dmax_k = 0.0; dmax_l = 0.0
while !eof(io)
    readline(io); nat = parse(Int, strip(readline(io)))
    lines = [readline(io) for _ in 1:nat]; box = parse(Float64, split(readline(io))[1])
    global nfr += 1
    (nfr - 1) % stride == 0 || continue
    pos = zeros(3, nat)
    for k in 1:nat
        l = lines[k]; pos[:, k] .= (parse(Float64, l[21:28]), parse(Float64, l[29:36]), parse(Float64, l[37:44]))
    end
    sys = MCm.RigidSystem(m.templates, vcat(2, fill(1, 500)), box * 10 / m.σ_angstrom, 1 / T, Xoshiro(1))
    k = 0
    for i in 1:sys.n_mol
        n = sys.templates[sys.tmpl[i]].n_sites
        X = pos[:, k+1:k+n]                       # trjconv output is not whole: unwrap to the first site
        for c in 2:n, d in 1:3
            X[d, c] -= box * round((X[d, c] - X[d, 1]) / box)
        end
        MCm.molecule_from_sites!(sys, i, X .* (10 / m.σ_angstrom)); k += n
    end
    for i in 1:sys.n_mol-1, j in i+1:sys.n_mol
        ub = MCm.pair_energy(sys, ffu, i, j)
        ub == 0.0 && continue
        global npairs += 1
        dk = abs(MCm.pair_energy(sys, ffk, i, j) - ub); dl = abs(MCm.pair_energy(sys, m.ff, i, j) - ub)
        dk > 0 && (global ndiff_k += 1; global dmax_k = max(dmax_k, dk))
        dl > 0 && (global ndiff_l += 1; global dmax_l = max(dmax_l, dl))
    end
end
@printf("%s: %d frames used (of %d), %d interacting molecule pairs\n", path, cld(nfr, stride), nfr, npairs)
@printf("  kT cap (100 kT):   pairs with capped ≠ bare = %d (max |Δu| = %.3g kJ/mol)\n", ndiff_k, dmax_k * m.ϵ_kelvin * 0.0083144626)
@printf("  legacy (1000 ε):   pairs with capped ≠ bare = %d (max |Δu| = %.3g kJ/mol)\n", ndiff_l, dmax_l * m.ϵ_kelvin * 0.0083144626)

# GROMACS topology of flexible solutes written from MolSim's own parsed model (OPLS itp in oplsaa.ff convention, or LigParGen),
# so the GROMACS free-energy reference and eKMC use the same Hamiltonian by construction: [ defaults ] (comb-rule 3,
# fudgeLJ/fudgeQQ from the model), [ atomtypes ] (SPC/E OW/HW as gromacs_fe/ff/opls_spce_flex.itp, then one type per distinct
# solute LJ type, σ/ϵ explicit), SPC/E [ moleculetype ] (settles), and one [ moleculetype ] per solute with explicit bonds,
# angles, RB propers (type 3), periodic dihedrals (type 4, LigParGen impropers) and 1-4 [ pairs ] (nrexcl 3).
# Usage: julia --project=. dev/gromacs_solute_itp.jl <out.itp> <solute:MOLTYPE> ...   e.g. butanol_flex:BUTF caffeine_lpg:CAFF
# A solute may appear twice under different MOLTYPE names (GROMACS couple-moltype decouples every copy of a type, so the one
# decoupled molecule of a mixture is its own type); `water:NAME` adds an extra SPC/E type (e.g. SOLX, the decoupled water).
using MolSim, Printf
const M = MolSim.MC

function solute_source(sym::Symbol)
    if haskey(M.OPLS_FLEX_ITP, sym)
        return M.opls_itp_molecule(joinpath(M.OPLS_FFDIR, M.OPLS_FLEX_ITP[sym]))
    elseif haskey(M.LIGPARGEN_ITP, sym)
        d = joinpath(M.LIGPARGEN_DIR, M.LIGPARGEN_ITP[sym])
        return M.ligpargen_itp_molecule(joinpath(d, "LIG.gmx.itp"), joinpath(d, "LIG.gmx.gro"))
    end
    error("unknown solute $sym")
end

out = ARGS[1]
specs = [(Symbol(split(a, ":")[1]), String(split(a, ":")[2])) for a in ARGS[2:end]]
water_copies = [mt for (sym, mt) in specs if sym === :water]
mols = [(sym, mt, solute_source(sym)) for (sym, mt) in specs if sym !== :water]
fl = unique([(m.bt.fudge_lj, m.bt.fudge_qq) for (_, _, m) in mols])
length(fl) == 1 || error("solutes with different fudge factors cannot share one [ defaults ]")
io = IOBuffer()
println(io, "; Flexible solutes + SPC/E for GROMACS free-energy references, written by dev/gromacs_solute_itp.jl from MolSim's parsed")
println(io, "; models (kmc_polyatomic): ", join(["$s ($mt)" for (s, mt, _) in mols], ", "), ". SPC/E as gromacs_fe/ff/opls_spce_flex.itp.")
println(io, "[ defaults ]\n; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ")
@printf(io, "  1       3          yes        %.4f   %.4f\n\n", fl[1][1], fl[1][2])
println(io, "[ atomtypes ]\n; name  at.num  mass      charge  ptype  sigma(nm)     epsilon(kJ/mol)")
println(io, "  OW    8       15.9994   0.0     A      0.316555789   0.650170")
println(io, "  HW    1       1.008     0.0     A      0.0           0.0")
tname(mt, k) = k == 0 ? "$(mt)_z" : "$(mt)_$k"
for (sym, mt, m) in mols
    for (k, (σ, ϵ)) in enumerate(m.lj)
        @printf(io, "  %-8s 0  1.000  0.0  A  %.9f  %.9f\n", tname(mt, k), σ, ϵ)
    end
    any(==(0), m.types) && @printf(io, "  %-8s 0  1.000  0.0  A  0.0  0.0\n", tname(mt, 0))
end
println(io, """

[ moleculetype ]
; name  nrexcl
  SOL   2
[ atoms ]
  1  OW   1     SOL OW   1    -0.8476 15.9994
  2  HW   1     SOL HW1  1     0.4238 1.008
  3  HW   1     SOL HW2  1     0.4238 1.008
[ settles ]
  1   1      0.1    0.1632993
[ exclusions ]
  1 2 3
  2 1 3
  3 1 2
""")
for wt in water_copies
    println(io, """[ moleculetype ]
; name  nrexcl
  $wt   2
[ atoms ]
  1  OW   1     $wt OW   1    -0.8476 15.9994
  2  HW   1     $wt HW1  1     0.4238 1.008
  3  HW   1     $wt HW2  1     0.4238 1.008
[ constraints ]
; a second SETTLE moleculetype is not allowed: rigid water as three constraints (as gromacs_fe/ff/opls_spce_solx.itp)
  1  2  1  0.1
  1  3  1  0.1
  2  3  1  0.1632993
[ exclusions ]
  1 2 3
  2 1 3
  3 1 2
""")
end
for (sym, mt, m) in mols
    bt = m.bt
    println(io, "[ moleculetype ]\n; name  nrexcl\n  $mt  3\n[ atoms ]")
    for a in 1:bt.n_sites
        @printf(io, "  %3d  %-8s 1  %s  A%d  1  %10.6f  %8.4f\n", a, tname(mt, m.types[a]), mt[1:3], a, m.q[a], m.mass[a])
    end
    println(io, "[ bonds ]")
    for b in bt.bonds; @printf(io, "  %3d %3d  1  %.6f  %.4f\n", b.i, b.j, b.b0, b.kb); end
    println(io, "[ angles ]")
    for a in bt.angles; @printf(io, "  %3d %3d %3d  1  %.6f  %.6f\n", a.i, a.j, a.k, rad2deg(a.θ0), a.kθ); end
    println(io, "[ dihedrals ]")
    for d in bt.dihedrals
        @printf(io, "  %3d %3d %3d %3d  3  %s\n", d.i, d.j, d.k, d.l, join([@sprintf("%.6f", c) for c in d.C], "  "))
    end
    for d in bt.periodic
        @printf(io, "  %3d %3d %3d %3d  4  %.6f  %.6f  %d\n", d.i, d.j, d.k, d.l, rad2deg(d.φs), d.kφ, d.n)
    end
    println(io, "[ pairs ]")
    for (a, b) in bt.pairs14; @printf(io, "  %3d %3d  1\n", a, b); end
    println(io)
end
write(out, take!(io))
println("wrote $out: ", join(["$s → $mt ($(m.bt.n_sites) atoms, $(length(m.bt.dihedrals)) RB, $(length(m.bt.periodic)) periodic)" for (s, mt, m) in mols], "; "))

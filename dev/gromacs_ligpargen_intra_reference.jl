# GROMACS vacuum reruns of LigParGen ibuprofen conformations (DESIGN §14 V4, T7.0-type check): writes
# test/data/ligpargen_ibuprofen_intra_reference.txt with the coordinates (nm) and every GROMACS energy term (kJ/mol).
# Conformations: the LigParGen geometry, and copies with random rotations of every rotatable bond plus 0.003 nm Gaussian
# displacements (so bonds, angles and impropers are off their minima). Vacuum emulated by one molecule in an 8 nm box,
# cut-offs 3.0 nm (> the molecule's largest intramolecular distance), no potential modifiers, no dispersion correction.
# Usage (head node, seconds): source /shared/opt/gromacs-2024.3-plumed2.10.1/env.sh; julia --project=. dev/gromacs_ligpargen_intra_reference.jl
using MolSim, Random, Printf
const M = MolSim.MC
dir = joinpath(@__DIR__, "..", "data", "ligpargen", "ibuprofen")
work = "/shared/kmc_polyatomic/gromacs_fe/ibuprofen_intra"
mol = M.ligpargen_itp_molecule(joinpath(dir, "LIG.gmx.itp"), joinpath(dir, "LIG.gmx.gro"))
n = mol.bt.n_sites
names = [split(l)[2] for l in readlines(joinpath(dir, "LIG.gmx.gro"))[3:2+n]]
s = M.spce_solute_model(:ibuprofen_lpg; coulomb=:none, cutoff=:site)
tors = M.rotatable_bonds(s.solute)
cp(joinpath(dir, "LIG.gmx.itp"), joinpath(work, "LIG.gmx.itp"); force=true)
write(joinpath(work, "vac.top"), "#include \"LIG.gmx.itp\"\n[ system ]\nibuprofen in vacuum (LigParGen OPLS-AA)\n[ molecules ]\nLIG 1\n")
write(joinpath(work, "rerun.mdp"), """
integrator = md
nsteps = 0
cutoff-scheme = Verlet
coulombtype = cut-off
coulomb-modifier = None
rcoulomb = 3.0
vdwtype = cut-off
vdw-modifier = None
rvdw = 3.0
DispCorr = no
pbc = xyz
constraints = none
nstcalcenergy = 1
nstenergy = 1
""")
terms = ["Bond", "Angle", "Ryckaert-Bell.", "Per. Imp. Dih.", "LJ-14", "Coulomb-14", "LJ (SR)", "Coulomb (SR)", "Potential"]
rng = Xoshiro(2026)
confs = Matrix{Float64}[]
push!(confs, copy(mol.X0))
for c in 1:6
    X = copy(mol.X0)
    for (a, b, mov) in tors
        M.rotate_fragment!(X, a, b, mov, (2rand(rng) - 1) * π)
    end
    X .+= 0.003 .* randn(rng, 3, n)
    push!(confs, X .- sum(X; dims=2) ./ n)
end
out = open(joinpath(@__DIR__, "..", "test", "data", "ligpargen_ibuprofen_intra_reference.txt"), "w")
println(out, "# LigParGen ibuprofen (data/ligpargen/ibuprofen) vacuum rerun, GROMACS 2024.3 (dev/gromacs_ligpargen_intra_reference.jl)")
println(out, "# per line: 3n coordinates (nm, site order of the itp), then ", join(terms, " | "), " (kJ/mol)")
for (c, X) in enumerate(confs)
    g = joinpath(work, "conf$c.gro")
    open(g, "w") do io
        println(io, "ibuprofen conformation $c"); println(io, n)
        for s in 1:n
            @printf(io, "%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n", 1, "LIG", names[s], s, X[1, s] + 4, X[2, s] + 4, X[3, s] + 4)
        end
        println(io, "   8.00000   8.00000   8.00000")
    end
    # the .gro has 3 decimals: re-read the rounded coordinates so both codes see the same geometry
    Xr = zeros(3, n)
    for (s, l) in enumerate(readlines(g)[3:2+n]); Xr[:, s] .= parse.(Float64, [l[21:28], l[29:36], l[37:44]]) .- 4; end
    cd(work) do
        run(pipeline(`gmx_mpi grompp -f rerun.mdp -c conf$c.gro -p vac.top -o rr$c.tpr -maxwarn 2`; stdout=devnull, stderr="grompp$c.log"))
        run(pipeline(`gmx_mpi mdrun -s rr$c.tpr -rerun conf$c.gro -deffnm rr$c -ntomp 1`; stdout=devnull, stderr="mdrun$c.log"))
        run(pipeline(pipeline(`printf "1\n2\n3\n4\n5\n6\n7\n8\n9\n0\n"`, `gmx_mpi energy -f rr$c.edr -o e$c.xvg`); stdout=devnull, stderr="energy$c.log"))   # by number (names contain spaces)
    end
    vals = Float64[]
    lines = filter(l -> !startswith(l, "#") && !startswith(l, "@"), readlines(joinpath(work, "e$c.xvg")))
    legends = [match(r"\"(.*)\"", l)[1] for l in readlines(joinpath(work, "e$c.xvg")) if occursin(r"^@ s\d+ legend", l)]
    v = parse.(Float64, split(lines[1]))[2:end]
    for tname in terms
        k = findfirst(==(tname), legends); k === nothing && error("term $tname not in $legends")
        push!(vals, v[k])
    end
    println(out, join([@sprintf("%.3f", x) for x in vec(Xr)], " "), " ", join([@sprintf("%.6f", x) for x in vals], " "))
    @printf("conf %d: %s\n", c, join([@sprintf("%s=%.4f", t, x) for (t, x) in zip(terms, vals)], " "))
end
close(out)

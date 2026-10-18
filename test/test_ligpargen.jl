# LigParGen OPLS-AA ibuprofen (data/ligpargen/ibuprofen, DESIGN §14 V4): the topology reader against GROMACS 2024.3 vacuum
# reruns of 7 conformations (dev/gromacs_ligpargen_intra_reference.jl), every energy term, and the reduced-unit model.
# Tolerance derived, not chosen: GROMACS (mixed precision) reads the coordinates in single precision (molecule placed at
# 4 nm in the box), so each term V_t carries |∂V_t/∂x| u₃₂ (|x| + 4 nm) per coordinate, plus the single-precision sum
# γ₃₂(n) Σ|V_t| (Higham Eq. 3.8) and the xvg print rounding (6 decimals).
using Test
using MolSim
const MCl = MolSim.MC

const LPG_DIR = joinpath(@__DIR__, "..", "data", "ligpargen", "ibuprofen")

function lpg_terms(mol)
    bt = mol.bt; K = length(mol.lj)
    σm = [sqrt(mol.lj[a][1] * mol.lj[b][1]) for a in 1:K, b in 1:K]; ϵm = [sqrt(mol.lj[a][2] * mol.lj[b][2]) for a in 1:K, b in 1:K]
    lj(Y, a, b) = (mol.types[a] > 0 && mol.types[b] > 0) ?
        (s6 = (σm[mol.types[a], mol.types[b]] / MCl._dist(Y, a, b))^6; 4ϵm[mol.types[a], mol.types[b]] * (s6^2 - s6)) : 0.0
    qq(Y, a, b) = MCl.KE_KJMOL_NM * mol.q[a] * mol.q[b] / MCl._dist(Y, a, b)
    T = Tuple{Int,Vector{Int},Function}[]            # (GROMACS term index, atoms, V(X))
    for b in bt.bonds; push!(T, (1, [b.i, b.j], Y -> 0.5b.kb * (MCl._dist(Y, b.i, b.j) - b.b0)^2)); end
    for a in bt.angles; push!(T, (2, [a.i, a.j, a.k], Y -> 0.5a.kθ * (MCl.bond_angle(Y, a.i, a.j, a.k) - a.θ0)^2)); end
    for d in bt.dihedrals
        push!(T, (3, [d.i, d.j, d.k, d.l], Y -> (c = -cos(MCl.dihedral_angle(Y, d.i, d.j, d.k, d.l)); sum(d.C[m] * c^(m - 1) for m in 1:6))))
    end
    for d in bt.periodic
        push!(T, (4, [d.i, d.j, d.k, d.l], Y -> d.kφ * (1 + cos(d.n * MCl.dihedral_angle(Y, d.i, d.j, d.k, d.l) - d.φs))))
    end
    for (a, b) in bt.pairs14
        push!(T, (5, [a, b], Y -> bt.fudge_lj * lj(Y, a, b))); push!(T, (6, [a, b], Y -> bt.fudge_qq * qq(Y, a, b)))
    end
    for (a, b) in bt.nonbonded; push!(T, (7, [a, b], Y -> lj(Y, a, b))); push!(T, (8, [a, b], Y -> qq(Y, a, b))); end
    return T, σm, ϵm
end

@testset "LigParGen ibuprofen: topology, every intramolecular term and the reduced model = GROMACS vacuum reruns" begin
    mol = MCl.ligpargen_itp_molecule(joinpath(LPG_DIR, "LIG.gmx.itp"), joinpath(LPG_DIR, "LIG.gmx.gro"))
    bt = mol.bt; n = bt.n_sites
    @test n == 33 && length(bt.bonds) == 33 && length(bt.angles) == 58 && length(bt.dihedrals) == 80 && length(bt.periodic) == 7
    @test abs(sum(mol.q)) < 1e-12 && count(==(0), mol.types) == 1             # the hydroxyl H has no LJ
    @test bt.pairs14 == MCl.BondedTerms(n; bonds=bt.bonds, nrexcl=3).pairs14   # [ pairs ] = the 1-4 pairs of the bond graph
    s = MCl.spce_solute_model(:ibuprofen_lpg; coulomb=:none, cutoff=:site)
    tors = MCl.rotatable_bonds(s.solute)
    @test length(tors) == 8 && all(((a, b, _),) -> !(a in 1:3 || a in 9:11) || !(b in 1:3 || b in 9:11), tors)   # no ring bond
    kJ = s.ϵ_kelvin * 0.0083144626
    T, σm, ϵm = lpg_terms(mol)
    u32 = eps(Float32) / 2; γ32(k) = k * u32 / (1 - k * u32)
    nconf = 0
    for l in eachline(joinpath(@__DIR__, "data", "ligpargen_ibuprofen_intra_reference.txt"))
        startswith(l, "#") && continue
        v = parse.(Float64, split(l)); X = reshape(v[1:3n], 3, n); g = v[3n+1:end]
        E = zeros(8); A = zeros(8); B = zeros(8); cnt = zeros(Int, 8)
        for (c, at, V) in T
            e = V(X); E[c] += e; A[c] += abs(e); cnt[c] += 1
            for a in at, d in 1:3
                h = 1e-6; Y = copy(X); Y[d, a] += h; Z = copy(X); Z[d, a] -= h
                B[c] += abs((V(Y) - V(Z)) / 2h) * u32 * (abs(X[d, a]) + 4.0)
            end
        end
        bound = [B[c] + γ32(cnt[c] + 1) * A[c] + 5e-7 for c in 1:8]
        # the library functions give the same terms
        eb, ea, ed = MCl.bonded_energy(bt, X); ep = MCl.periodic_dihedral_energy(bt, X)
        lj14, qq14, lj, qq = MCl.intra_nonbonded_energy(bt, X, mol.types, mol.q, σm, ϵm, MCl.KE_KJMOL_NM)
        lib = [eb, ea, ed - ep, ep, lj14, qq14, lj, qq]
        @test all(abs.(lib .- E) .<= 1e-9 .* max.(1.0, A))
        @test all(abs.(lib .- g[1:8]) .<= bound)
        # the reduced-unit model (plain Coulomb) gives GROMACS's potential
        red = MCl.flex_intra_energy(s.solute, s.ff, X .* (10 / s.σ_angstrom)) * kJ
        @test abs(red - g[9]) <= sum(bound) + γ32(length(T)) * sum(A)
        nconf += 1
    end
    @test nconf == 7
end

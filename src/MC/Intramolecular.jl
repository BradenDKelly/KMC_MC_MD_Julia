# ============================================================================
# Intramolecular (bonded + intramolecular nonbonded) energies for flexible molecules
# ============================================================================
#
# Functional forms are those of GROMACS/OPLS-AA (GROMACS Reference Manual §4.2–4.3; Jorgensen, Maxwell,
# Tirado-Rives, JACS 118 (1996) 11225), so that a molecule's bonded energy can be checked term by term against
# `gmx mdrun -rerun` (test T7.0, DESIGN.md §7.5). All functions are unit-agnostic: coordinates, constants and
# charges must simply be consistent (nm / kJ mol⁻¹ / e with ke = 138.935458 for the GROMACS comparison; reduced
# units inside the simulation). DESIGN.md §7.

"Harmonic bond, GROMACS type 1: V = ½ k_b (r − b0)²."
struct HarmonicBond
    i::Int; j::Int; b0::Float64; kb::Float64
end

"Harmonic angle, GROMACS type 1: V = ½ k_θ (θ − θ0)², θ0 in radians."
struct HarmonicAngle
    i::Int; j::Int; k::Int; θ0::Float64; kθ::Float64
end

"Ryckaert–Bellemans dihedral, GROMACS type 3: V = Σ_{n=0}^{5} C_n cosⁿψ with ψ = φ − 180° (φ = 0 for cis)."
struct RBDihedral
    i::Int; j::Int; k::Int; l::Int; C::NTuple{6,Float64}
end

"""
    BondedTerms(n_sites; bonds, angles, dihedrals, fudge_lj=0.5, fudge_qq=0.5, nrexcl=3)

Bonded terms of one molecule plus the intramolecular nonbonded pair lists derived from the bond graph as
GROMACS does with `nrexcl` and `gen-pairs`: pairs separated by fewer than `nrexcl` bonds are excluded, pairs at
exactly `nrexcl` bonds (1-4 for nrexcl = 3) are `pairs14` scaled by `fudge_lj`/`fudge_qq`, pairs further apart
are `nonbonded` at full strength (plain LJ and plain Coulomb, no cutoff).
"""
struct BondedTerms
    n_sites::Int
    bonds::Vector{HarmonicBond}
    angles::Vector{HarmonicAngle}
    dihedrals::Vector{RBDihedral}
    pairs14::Vector{Tuple{Int,Int}}
    nonbonded::Vector{Tuple{Int,Int}}
    fudge_lj::Float64
    fudge_qq::Float64
end

function BondedTerms(n_sites::Int; bonds::Vector{HarmonicBond}=HarmonicBond[],
                     angles::Vector{HarmonicAngle}=HarmonicAngle[], dihedrals::Vector{RBDihedral}=RBDihedral[],
                     fudge_lj::Float64=0.5, fudge_qq::Float64=0.5, nrexcl::Int=3)
    # bond-graph distances by breadth-first search
    adj = [Int[] for _ in 1:n_sites]
    for b in bonds
        push!(adj[b.i], b.j); push!(adj[b.j], b.i)
    end
    dist = fill(typemax(Int), n_sites, n_sites)
    for s in 1:n_sites
        dist[s, s] = 0
        queue = [s]
        while !isempty(queue)
            a = popfirst!(queue)
            for b in adj[a]
                if dist[s, b] == typemax(Int)
                    dist[s, b] = dist[s, a] + 1
                    push!(queue, b)
                end
            end
        end
    end
    p14 = Tuple{Int,Int}[]; nb = Tuple{Int,Int}[]
    for a in 1:n_sites-1, b in a+1:n_sites
        d = dist[a, b]
        if d == nrexcl
            push!(p14, (a, b))
        elseif d > nrexcl
            push!(nb, (a, b))
        end
    end
    return BondedTerms(n_sites, bonds, angles, dihedrals, p14, nb, fudge_lj, fudge_qq)
end

@inline _dist(X, a, b) = sqrt((X[1, a] - X[1, b])^2 + (X[2, a] - X[2, b])^2 + (X[3, a] - X[3, b])^2)

"Bond angle i–j–k (radians) from the coordinate columns of X."
function bond_angle(X::AbstractMatrix{Float64}, i::Int, j::Int, k::Int)::Float64
    ax, ay, az = X[1, i] - X[1, j], X[2, i] - X[2, j], X[3, i] - X[3, j]
    bx, by, bz = X[1, k] - X[1, j], X[2, k] - X[2, j], X[3, k] - X[3, j]
    c = (ax * bx + ay * by + az * bz) / sqrt((ax * ax + ay * ay + az * az) * (bx * bx + by * by + bz * bz))
    return acos(clamp(c, -1.0, 1.0))
end

"Dihedral angle i–j–k–l (radians, 0 = cis, IUPAC sign) from the coordinate columns of X."
function dihedral_angle(X::AbstractMatrix{Float64}, i::Int, j::Int, k::Int, l::Int)::Float64
    b1 = (X[1, j] - X[1, i], X[2, j] - X[2, i], X[3, j] - X[3, i])
    b2 = (X[1, k] - X[1, j], X[2, k] - X[2, j], X[3, k] - X[3, j])
    b3 = (X[1, l] - X[1, k], X[2, l] - X[2, k], X[3, l] - X[3, k])
    n1 = (b1[2] * b2[3] - b1[3] * b2[2], b1[3] * b2[1] - b1[1] * b2[3], b1[1] * b2[2] - b1[2] * b2[1])
    n2 = (b2[2] * b3[3] - b2[3] * b3[2], b2[3] * b3[1] - b2[1] * b3[3], b2[1] * b3[2] - b2[2] * b3[1])
    nb2 = sqrt(b2[1]^2 + b2[2]^2 + b2[3]^2)
    m1 = (n1[2] * b2[3] - n1[3] * b2[2], n1[3] * b2[1] - n1[1] * b2[3], n1[1] * b2[2] - n1[2] * b2[1]) ./ nb2
    x = n1[1] * n2[1] + n1[2] * n2[2] + n1[3] * n2[3]
    y = m1[1] * n2[1] + m1[2] * n2[2] + m1[3] * n2[3]
    return atan(y, x)
end

"""
    bonded_energy(bt, X) -> (bond, angle, dihedral)

The three bonded contributions of one molecule with site coordinates X (3 × n_sites).
"""
function bonded_energy(bt::BondedTerms, X::AbstractMatrix{Float64})
    eb = 0.0
    for b in bt.bonds
        eb += 0.5 * b.kb * (_dist(X, b.i, b.j) - b.b0)^2
    end
    ea = 0.0
    for a in bt.angles
        ea += 0.5 * a.kθ * (bond_angle(X, a.i, a.j, a.k) - a.θ0)^2
    end
    ed = 0.0
    for d in bt.dihedrals
        cψ = -cos(dihedral_angle(X, d.i, d.j, d.k, d.l))      # ψ = φ − 180°
        p = 1.0
        for n in 1:6
            ed += d.C[n] * p
            p *= cψ
        end
    end
    return eb, ea, ed
end

"""
    intra_nonbonded_energy(bt, X, types, q, σ_mix, ϵ_mix, ke) -> (lj14, qq14, lj, qq)

Intramolecular LJ and plain Coulomb (ke q_a q_b / r) over the 1-4 pairs (scaled by the fudge factors) and
over the pairs beyond (full strength). `types[s] = 0` means no LJ on site s. No cutoff.
"""
function intra_nonbonded_energy(bt::BondedTerms, X::AbstractMatrix{Float64}, types::AbstractVector{Int},
                                q::AbstractVector{Float64}, σ_mix::AbstractMatrix{Float64},
                                ϵ_mix::AbstractMatrix{Float64}, ke::Float64)
    function pair(a, b)
        r = _dist(X, a, b)
        lj = 0.0
        if types[a] > 0 && types[b] > 0
            s6 = (σ_mix[types[a], types[b]] / r)^6
            lj = 4.0 * ϵ_mix[types[a], types[b]] * (s6 * s6 - s6)
        end
        return lj, ke * q[a] * q[b] / r
    end
    lj14 = 0.0; qq14 = 0.0
    for (a, b) in bt.pairs14
        lj, qq = pair(a, b)
        lj14 += bt.fudge_lj * lj; qq14 += bt.fudge_qq * qq
    end
    lj = 0.0; qq = 0.0
    for (a, b) in bt.nonbonded
        l, c = pair(a, b)
        lj += l; qq += c
    end
    return lj14, qq14, lj, qq
end

"Total intramolecular energy (bonded + 1-4 + further nonbonded)."
function intra_energy(bt::BondedTerms, X::AbstractMatrix{Float64}, types, q, σ_mix, ϵ_mix, ke::Float64)::Float64
    eb, ea, ed = bonded_energy(bt, X)
    lj14, qq14, lj, qq = intra_nonbonded_energy(bt, X, types, q, σ_mix, ϵ_mix, ke)
    return eb + ea + ed + lj14 + qq14 + lj + qq
end

"""
    opls_methanol_bonded() -> (bt, types, q, σ_types, ϵ_types)   [nm, kJ/mol, e]

OPLS-AA methanol (GROMACS oplsaa.ff): sites C, H1, H2, H3, O, HO; bonds CT–HC 0.1090/284512, CT–OH
0.1410/267776, OH–HO 0.0945/462750.4; angles HC–CT–HC 107.8°/276.144, HC–CT–OH 109.5°/292.880, CT–OH–HO
108.5°/460.240; dihedrals HC–CT–OH–HO RB (0.94140, 2.82420, 0, −3.76560, 0, 0); 1-4 fudge 0.5/0.5.
LJ types 1 = CT (0.350 nm, 0.276144), 2 = HC (0.250, 0.12552), 3 = OH (0.312, 0.71128), HO none; geometric mixing.
"""
function opls_methanol_bonded()
    d2r = π / 180
    bonds = [HarmonicBond(1, 2, 0.1090, 284512.0), HarmonicBond(1, 3, 0.1090, 284512.0),
             HarmonicBond(1, 4, 0.1090, 284512.0), HarmonicBond(1, 5, 0.1410, 267776.0),
             HarmonicBond(5, 6, 0.0945, 462750.4)]
    angles = [HarmonicAngle(2, 1, 3, 107.8 * d2r, 276.144), HarmonicAngle(2, 1, 4, 107.8 * d2r, 276.144),
              HarmonicAngle(3, 1, 4, 107.8 * d2r, 276.144), HarmonicAngle(2, 1, 5, 109.5 * d2r, 292.880),
              HarmonicAngle(3, 1, 5, 109.5 * d2r, 292.880), HarmonicAngle(4, 1, 5, 109.5 * d2r, 292.880),
              HarmonicAngle(1, 5, 6, 108.5 * d2r, 460.240)]
    C = (0.94140, 2.82420, 0.0, -3.76560, 0.0, 0.0)
    dihedrals = [RBDihedral(2, 1, 5, 6, C), RBDihedral(3, 1, 5, 6, C), RBDihedral(4, 1, 5, 6, C)]
    bt = BondedTerms(6; bonds=bonds, angles=angles, dihedrals=dihedrals, fudge_lj=0.5, fudge_qq=0.5, nrexcl=3)
    types = [1, 2, 2, 2, 3, 0]
    q = [0.145, 0.040, 0.040, 0.040, -0.683, 0.418]
    return bt, types, q, [0.350, 0.250, 0.312], [0.276144, 0.12552, 0.71128]
end

"GROMACS electric conversion factor ke = e²/(4π ε₀) in kJ mol⁻¹ nm e⁻² (GROMACS manual Table 2.3)."
const KE_KJMOL_NM = 138.935458

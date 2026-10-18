"""
Flexible OPLS-AA molecules from GROMACS topology files (DESIGN §7; JOURNAL entry 53). Reads a molecule `.itp`
(atoms with their opls types and charges, bonds, angles, RB dihedrals; nrexcl, gen-pairs with fudge 0.5/0.5 as in
oplsaa.ff/forcefield.itp) and takes every parameter from the force field's own `ffnonbonded.itp` / `ffbonded.itp`
(bond types via the bond_type column; dihedral types matched on the four bond types, either direction, X wildcards
last), so the template is the GROMACS Hamiltonian of that file — nothing is typed in by hand. A reference geometry
is generated from the equilibrium bond lengths and angles (heavy-atom torsions anti, hydrogens staggered) and
relaxed by steepest descent on the intramolecular energy; it only seeds the vacuum conformer reservoir.
"""

using LinearAlgebra: norm, cross

const OPLS_FFDIR = "/shared/opt/gromacs-2024.3-plumed2.10.1/share/gromacs/top/oplsaa.ff"

_itp_lines(path) = [strip(split(l, ';')[1]) for l in eachline(path)]

function _itp_sections(path)
    secs = Dict{String,Vector{Vector{String}}}(); cur = ""
    for l in _itp_lines(path)
        isempty(l) && continue
        startswith(l, "#") && continue
        if startswith(l, "[")
            cur = strip(l[2:end-1]); haskey(secs, cur) || (secs[cur] = Vector{String}[])
            continue
        end
        push!(get!(secs, cur, Vector{String}[]), split(l))
    end
    return secs
end

# ffbonded.itp has two [ dihedraltypes ] sections (impropers, then propers): collect all rows by section name
function _ffbonded(ffdir)
    bonds = Dict{Tuple{String,String},Tuple{Float64,Float64}}()
    angles = Dict{Tuple{String,String,String},Tuple{Float64,Float64}}()
    rb = Vector{Tuple{NTuple{4,String},NTuple{6,Float64}}}()
    cur = ""
    for l in _itp_lines(joinpath(ffdir, "ffbonded.itp"))
        isempty(l) && continue
        startswith(l, "#") && continue
        if startswith(l, "[")
            cur = strip(l[2:end-1]); continue
        end
        f = split(l)
        if cur == "bondtypes" && length(f) >= 5
            bonds[(String(f[1]), String(f[2]))] = (parse(Float64, f[4]), parse(Float64, f[5]))
        elseif cur == "angletypes" && length(f) >= 6
            angles[(String(f[1]), String(f[2]), String(f[3]))] = (parse(Float64, f[5]), parse(Float64, f[6]))
        elseif cur == "dihedraltypes" && length(f) >= 11 && f[5] == "3"
            push!(rb, ((String(f[1]), String(f[2]), String(f[3]), String(f[4])), ntuple(n -> parse(Float64, f[5+n]), 6)))
        end
    end
    return bonds, angles, rb
end

function _ffnonbonded(ffdir)
    d = Dict{String,NamedTuple}()
    for l in _itp_lines(joinpath(ffdir, "ffnonbonded.itp"))
        f = split(l)
        (length(f) >= 8 && startswith(f[1], "opls_")) || continue
        d[String(f[1])] = (btype=String(f[2]), mass=parse(Float64, f[4]), σ=parse(Float64, f[7]), ϵ=parse(Float64, f[8]))
    end
    return d
end

_lookup2(D, a, b) = haskey(D, (a, b)) ? D[(a, b)] : haskey(D, (b, a)) ? D[(b, a)] : error("OPLS: no bondtype $a-$b")
_lookup3(D, a, b, c) = haskey(D, (a, b, c)) ? D[(a, b, c)] : haskey(D, (c, b, a)) ? D[(c, b, a)] : error("OPLS: no angletype $a-$b-$c")
function _lookup_rb(rb, t::NTuple{4,String})
    best = nothing; nwild = 5
    for (k, C) in rb, key in (t, reverse(t))
        ok = all(k[i] == "X" || k[i] == key[i] for i in 1:4)
        ok || continue
        w = count(==("X"), k)
        w < nwild && (best = C; nwild = w)
    end
    best === nothing && error("OPLS: no RB dihedraltype for $(join(t, '-'))")
    return best
end

"""
    opls_itp_molecule(itp; ffdir=OPLS_FFDIR, name)

Parse a GROMACS OPLS-AA molecule. Returns (bt::BondedTerms in GROMACS units nm/kJ mol⁻¹/rad, types::Vector{Int}
(0 = no LJ), q, mass, lj = [(σ_nm, ϵ_kJmol) per LJ type], X0 (3×n geometry, nm), opls = per-site opls types).
"""
function opls_itp_molecule(itp::AbstractString; ffdir::AbstractString=OPLS_FFDIR)
    S = _itp_sections(itp)
    nb = _ffnonbonded(ffdir); bdb, adb, rb = _ffbonded(ffdir)
    atoms = S["atoms"]; n = length(atoms)
    opls = [String(a[2]) for a in atoms]
    q = [parse(Float64, a[7]) for a in atoms]
    mass = [length(a) >= 8 ? parse(Float64, a[8]) : nb[a[2]].mass for a in atoms]
    bt_of = [String(nb[t].btype) for t in opls]
    nrexcl = parse(Int, S["moleculetype"][1][2])
    bonds = HarmonicBond[]
    for b in S["bonds"]
        i, j = parse(Int, b[1]), parse(Int, b[2])
        b0, kb = _lookup2(bdb, bt_of[i], bt_of[j])
        push!(bonds, HarmonicBond(i, j, b0, kb))
    end
    angles = HarmonicAngle[]
    for a in S["angles"]
        i, j, k = parse.(Int, a[1:3])
        θ0, kθ = _lookup3(adb, bt_of[i], bt_of[j], bt_of[k])
        push!(angles, HarmonicAngle(i, j, k, θ0 * π / 180, kθ))
    end
    dihs = RBDihedral[]
    for d in get(S, "dihedrals", Vector{String}[])
        i, j, k, l = parse.(Int, d[1:4])
        C = _lookup_rb(rb, (bt_of[i], bt_of[j], bt_of[k], bt_of[l]))
        push!(dihs, RBDihedral(i, j, k, l, C))
    end
    bt = BondedTerms(n; bonds=bonds, angles=angles, dihedrals=dihs, fudge_lj=0.5, fudge_qq=0.5, nrexcl=nrexcl)
    # LJ types: one per distinct opls type with ϵ > 0, in order of first appearance
    ljkeys = String[]; types = zeros(Int, n)
    for (s, t) in enumerate(opls)
        nb[t].ϵ > 0 || continue
        idx = findfirst(==(t), ljkeys)
        idx === nothing && (push!(ljkeys, t); idx = length(ljkeys))
        types[s] = idx
    end
    lj = [(nb[t].σ, nb[t].ϵ) for t in ljkeys]
    X0 = _opls_build_geometry(bt, n)
    X0 = _opls_relax(bt, X0, types, q, lj)
    return (bt=bt, types=types, q=q, mass=mass, lj=lj, X0=X0, opls=opls)
end

"""
    ligpargen_itp_molecule(itp, gro) -> NamedTuple (as `opls_itp_molecule`)

Parse a self-contained GROMACS topology as LigParGen writes it (Dodda et al., Nucleic Acids Res. 45 (2017) W331): every
parameter inline — [ atomtypes ] σ/ϵ (comb-rule 3, geometric), [ atoms ] charges and masses, harmonic bonds and angles
(type 1), Ryckaert–Bellemans propers (type 3), periodic dihedrals (types 1/4/9: the impropers), the explicit [ pairs ]
list (fudgeLJ/fudgeQQ from [ defaults ]) and nrexcl. Nothing is looked up or typed in by hand. The geometry is the .gro's
(nm, centred). LJ types: one per distinct (σ, ϵ) with ϵ > 0, in order of first appearance.
"""
function ligpargen_itp_molecule(itp::AbstractString, gro::AbstractString)
    S = _itp_sections(itp)
    d = S["defaults"][1]
    parse(Int, d[2]) == 3 || error("ligpargen_itp_molecule: comb-rule $(d[2]) (only 3, geometric, as OPLS-AA)")
    fudge_lj, fudge_qq = parse(Float64, d[4]), parse(Float64, d[5])
    at = Dict(String(a[1]) => (σ=parse(Float64, a[end-1]), ϵ=parse(Float64, a[end])) for a in S["atomtypes"])
    atoms = S["atoms"]; n = length(atoms)
    all(parse(Int, atoms[s][1]) == s for s in 1:n) || error("ligpargen_itp_molecule: atoms not numbered 1…n")
    opls = [String(a[2]) for a in atoms]
    q = [parse(Float64, a[7]) for a in atoms]
    mass = [parse(Float64, a[8]) for a in atoms]
    nrexcl = parse(Int, S["moleculetype"][1][2])
    bonds = HarmonicBond[]
    for b in S["bonds"]
        b[3] == "1" || error("ligpargen_itp_molecule: bond type $(b[3])")
        push!(bonds, HarmonicBond(parse(Int, b[1]), parse(Int, b[2]), parse(Float64, b[4]), parse(Float64, b[5])))
    end
    angles = HarmonicAngle[]
    for a in S["angles"]
        a[4] == "1" || error("ligpargen_itp_molecule: angle type $(a[4])")
        push!(angles, HarmonicAngle(parse.(Int, a[1:3])..., parse(Float64, a[5]) * π / 180, parse(Float64, a[6])))
    end
    dihs = RBDihedral[]; per = PeriodicDihedral[]
    for x in get(S, "dihedrals", Vector{String}[])
        i, j, k, l = parse.(Int, x[1:4])
        if x[5] == "3"
            push!(dihs, RBDihedral(i, j, k, l, ntuple(m -> parse(Float64, x[5+m]), 6)))
        elseif x[5] in ("1", "4", "9")
            push!(per, PeriodicDihedral(i, j, k, l, parse(Float64, x[6]) * π / 180, parse(Float64, x[7]), parse(Int, x[8])))
        else
            error("ligpargen_itp_molecule: dihedral type $(x[5])")
        end
    end
    pairs = [(parse(Int, p[1]), parse(Int, p[2])) for p in get(S, "pairs", Vector{String}[])]
    bt = BondedTerms(n; bonds=bonds, angles=angles, dihedrals=dihs, periodic=per, fudge_lj=fudge_lj, fudge_qq=fudge_qq,
                     nrexcl=nrexcl, pairs14=pairs)
    ljkeys = Tuple{Float64,Float64}[]; types = zeros(Int, n)
    for (s, t) in enumerate(opls)
        p = (at[t].σ, at[t].ϵ)
        p[2] > 0 || continue
        idx = findfirst(==(p), ljkeys)
        idx === nothing && (push!(ljkeys, p); idx = length(ljkeys))
        types[s] = idx
    end
    gl = readlines(gro)
    parse(Int, strip(gl[2])) == n || error("ligpargen_itp_molecule: $gro has $(strip(gl[2])) atoms, topology $n")
    X0 = zeros(3, n)
    for s in 1:n
        f = split(gl[2+s])
        X0[:, s] .= parse.(Float64, f[end-2:end])
    end
    X0 .-= (X0 * mass) ./ sum(mass)
    return (bt=bt, types=types, q=q, mass=mass, lj=ljkeys, X0=X0, opls=opls)
end

# place atoms along the bond tree with equilibrium lengths/angles; dihedral 180° for the first substituent, ±120° offsets
function _opls_build_geometry(bt::BondedTerms, n::Int)
    adj = [Int[] for _ in 1:n]
    b0 = Dict{Tuple{Int,Int},Float64}()
    for b in bt.bonds
        push!(adj[b.i], b.j); push!(adj[b.j], b.i); b0[(b.i, b.j)] = b.b0; b0[(b.j, b.i)] = b.b0
    end
    θ0(i, j, k) = (a = findfirst(x -> (x.i, x.j, x.k) == (i, j, k) || (x.i, x.j, x.k) == (k, j, i), bt.angles); a === nothing ? 109.5 * π / 180 : bt.angles[a].θ0)
    root = argmax([length(a) for a in adj])
    X = zeros(3, n); placed = falses(n); parent = zeros(Int, n)
    placed[root] = true
    order = [root]; queue = [root]
    while !isempty(queue)
        a = popfirst!(queue)
        kids = [c for c in adj[a] if !placed[c]]
        for (m, c) in enumerate(kids)
            p = parent[a]; g = p > 0 ? parent[p] : 0
            r = b0[(a, c)]
            if p == 0                        # first shell around the root: tetrahedral directions
                dirs = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
                d = collect(Float64, dirs[m]) ./ sqrt(3)
                X[:, c] = X[:, a] .+ r .* d
            else
                # internal coordinates (r, θ, φ) relative to (a, p, ref)
                ref = g > 0 ? g : first([x for x in adj[p] if x != a && placed[x]] ∪ [0])
                θ = θ0(p, a, c); φ = π + (m - 1) * 2π / 3
                A = X[:, a]; B = X[:, p]
                C = ref > 0 ? X[:, ref] : B .+ [0.0, 1.0, 0.3]
                bc = (A .- B) ./ norm(A .- B)
                nrm = cross(B .- C, bc); nrm ./= norm(nrm)
                mvec = cross(nrm, bc)
                d2 = [-r * cos(θ), r * sin(θ) * cos(φ), r * sin(θ) * sin(φ)]
                X[:, c] = A .+ d2[1] .* bc .+ d2[2] .* mvec .+ d2[3] .* nrm
            end
            placed[c] = true; parent[c] = a; push!(queue, c)
        end
    end
    return X
end

function _opls_relax(bt::BondedTerms, X::Matrix{Float64}, types, q, lj; steps::Int=4000)
    K = length(lj)
    σm = [0.5 * (lj[a][1] + lj[b][1]) for a in 1:K, b in 1:K]       # placeholder, replaced by geometric below
    σm = [sqrt(lj[a][1] * lj[b][1]) for a in 1:K, b in 1:K]; ϵm = [sqrt(lj[a][2] * lj[b][2]) for a in 1:K, b in 1:K]
    E(Y) = intra_energy(bt, Y, types, q, σm, ϵm, KE_KJMOL_NM)
    h = 1e-3; e = E(X)
    for _ in 1:steps
        G = zeros(size(X)); δ = 1e-7
        for i in eachindex(X)
            Y = copy(X); Y[i] += δ; G[i] = (E(Y) - e) / δ
        end
        Y = X .- h .* G ./ max(maximum(abs.(G)), 1e-12)
        eY = E(Y)
        if eY < e
            X = Y; e = eY; h *= 1.2
        else
            h *= 0.2
        end
        h < 1e-9 && break
    end
    c = sum(X; dims=2) ./ size(X, 2)
    return X .- c
end

"""
    load_gro!(sys, gro; σ_angstrom) -> sys

Set the box, COMs and conformations of the flexible molecules of `sys` from a GROMACS .gro file (nm; cubic box; atoms in
molecule order, the site order of each template). Each molecule is made whole along its bonds (minimum image of every bond,
breadth-first from site 1), its COM wrapped into the box and its conformation stored as the body (orientation = identity).
`σ_angstrom` converts nm to the reduced length.
"""
function load_gro!(sys, gro::AbstractString; σ_angstrom::Float64)
    lines = readlines(gro)
    nat = parse(Int, strip(lines[2]))
    nat == sys.n_sites_total || error("load_gro!: $gro has $nat atoms, system $(sys.n_sites_total)")
    lu = 10.0 / σ_angstrom
    box = parse.(Float64, split(lines[end])[1:3])
    (box[1] ≈ box[2] ≈ box[3]) || error("load_gro!: cubic boxes only")
    sys.L = box[1] * lu
    X = zeros(3, nat)
    for s in 1:nat
        l = lines[2 + s]
        X[:, s] .= parse.(Float64, (l[21:28], l[29:36], l[37:44])) .* lu
    end
    for i in 1:sys.n_mol
        t = sys.templates[sys.tmpl[i]]
        isflexible(t) || error("load_gro!: molecule $i is rigid (flexible molecules only)")
        s0 = sys.site_start[i]; n = t.n_sites
        Y = X[:, s0+1:s0+n]
        adj = [Int[] for _ in 1:n]
        for b in t.bonded.bonds; push!(adj[b.i], b.j); push!(adj[b.j], b.i); end
        done = falses(n); done[1] = true; queue = [1]
        while !isempty(queue)
            a = popfirst!(queue)
            for b in adj[a]
                done[b] && continue
                for d in 1:3; Y[d, b] = Y[d, a] + _min_image(Y[d, b] - Y[d, a], sys.L); end
                done[b] = true; push!(queue, b)
            end
        end
        all(done) || error("load_gro!: molecule $i is not connected")
        c = (Y * t.site_mass) ./ sum(t.site_mass)
        set_conformation!(sys, i, Y)
        set_molecule!(sys, i, c, sys.quat[:, i])
    end
    return sys
end

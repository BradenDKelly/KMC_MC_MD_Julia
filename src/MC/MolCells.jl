"""
Molecular cell list for the real-space rows of rigid/flexible molecules (DESIGN §13.3 E3b; user 2026-10-17: "go ahead with
neighbour lists").

Linked cells (Allen & Tildesley, Computer Simulation of Liquids, 2nd ed., §5.3; Frenkel & Smit §F.2) over molecular centres
of mass. `pair_energy_virial(i, j)` is exactly zero unless the minimum-image COM distance is below
r_int = rc (molecule cutoff) or rc + ext_i + ext_j (site cutoff), so the row of molecule k needs only the molecules within
r_k = rc + ext_k + ext_max of k. The cells have side ≥ rc + 2 ext_max at construction; a query searches ⌈r_k / side⌉ layers of
cells around k (one layer normally; more if a flexible molecule has grown), and falls back to all molecules when that would
wrap the box. The result is the same row, entry by entry, as the all-pairs loop.

Incremental, as the eKMC bookkeeping: the list holds the committed COMs; `molecule_committed!` (every accepted MC move and
every eKMC event, beside the reciprocal-space commit) moves one molecule between cells in O(1) and raises ext_max if the
molecule grew. A molecule at an uncommitted (candidate or trial) pose is only ever the one whose row is evaluated, and its
current COM is used for the search, so its stale membership is never read. Rebuilt (O(N)) with the eKMC rebuild, after
volume changes, and checked by `cells_check` in the debug rebuilds.
"""

mutable struct MolCells
    L::Float64
    ncell::Int                    # cells per dimension
    side::Float64                 # L / ncell
    rc::Float64
    molcut::Bool                  # ff.cutoff === :molecule
    ext_max::Float64              # ≥ ext of every committed molecule
    head::Vector{Int}             # first molecule in each cell (0 = empty)
    next::Vector{Int}             # doubly linked: O(1) removal
    prev::Vector{Int}
    cell_of::Vector{Int}
    nbr::Vector{Int}              # gather buffer
    n_query::Int
    n_full::Int                   # queries that fell back to all molecules
end

@inline _cell1(x::Float64, L::Float64, n::Int) = mod(floor(Int, x / L * n), n)
@inline _cell_lin(ix::Int, iy::Int, iz::Int, n::Int) = (ix * n + iy) * n + iz + 1

function _cell_of(c::MolCells, sys, i::Int)
    @inbounds _cell_lin(_cell1(sys.com[1, i], c.L, c.ncell), _cell1(sys.com[2, i], c.L, c.ncell),
                        _cell1(sys.com[3, i], c.L, c.ncell), c.ncell)
end

"""
    MolCells(sys, ff)

Cell list for `sys` with the cut-off of `ff` (side ≥ rc + 2 max ext for the site cut-off, ≥ rc for the molecule cut-off).
"""
function MolCells(sys, ff)
    N = sys.n_mol
    c = MolCells(0.0, 0, 0.0, ff.rc, ff.cutoff === :molecule, 0.0, Int[], zeros(Int, N), zeros(Int, N), zeros(Int, N),
                 zeros(Int, N), 0, 0)
    cells_rebuild!(c, sys)
    return c
end

"Rebuild all memberships from the current COMs (and the cell grid if the box length changed)."
function cells_rebuild!(c::MolCells, sys)
    N = sys.n_mol
    length(c.cell_of) == N || error("cells_rebuild!: cell list built for $(length(c.cell_of)) molecules, system has $N")
    c.ext_max = maximum(sys.ext; init=0.0)
    if sys.L != c.L || isempty(c.head)
        c.L = sys.L
        reach = c.molcut ? c.rc : c.rc + 2c.ext_max
        c.ncell = max(1, floor(Int, c.L / reach))
        c.side = c.L / c.ncell
        c.head = zeros(Int, c.ncell^3)
    end
    fill!(c.head, 0)
    @inbounds for i in 1:N
        _cell_link!(c, i, _cell_of(c, sys, i))
    end
    return c
end

@inline function _cell_link!(c::MolCells, i::Int, cell::Int)
    @inbounds begin
        h = c.head[cell]
        c.next[i] = h; c.prev[i] = 0
        h > 0 && (c.prev[h] = i)
        c.head[cell] = i; c.cell_of[i] = cell
    end
end
@inline function _cell_unlink!(c::MolCells, i::Int)
    @inbounds begin
        p = c.prev[i]; n = c.next[i]
        p > 0 ? (c.next[p] = n) : (c.head[c.cell_of[i]] = n)
        n > 0 && (c.prev[n] = p)
    end
end

"Molecule i's committed pose changed: move it to its new cell, raise ext_max if it grew."
function cells_moved!(c::MolCells, sys, i::Int)
    cell = _cell_of(c, sys, i)
    if cell != c.cell_of[i]
        _cell_unlink!(c, i); _cell_link!(c, i, cell)
    end
    @inbounds sys.ext[i] > c.ext_max && (c.ext_max = sys.ext[i])
    return nothing
end

"Molecule i has been moved and accepted: reciprocal-space commit (no-op without Ewald) and cell-list update."
function molecule_committed!(sys, i::Int)
    recip_commit!(sys, i)
    sys.cells === nothing || cells_moved!(sys.cells, sys, i)
    sys.dev === nothing || dp_sync_molecule!(sys.dev.D, sys, i)
    return nothing
end

"The cell list answers rows of `ff` (built for the same cut-off)."
@inline _cells_for(c::MolCells, ff) = c.rc == ff.rc && c.molcut == (ff.cutoff === :molecule)

"""
    cells_gather!(c, sys, k) -> n

Molecules j ≠ k that can interact with k at its current pose into c.nbr[1:n] (every other molecule is beyond the pair
range); n = −1 if the search would cover the whole box (the caller loops over all molecules).
"""
function cells_gather!(c::MolCells, sys, k::Int)::Int
    c.n_query += 1
    r = c.molcut ? c.rc : c.rc + sys.ext[k] + c.ext_max
    nl = ceil(Int, r / c.side)
    n = c.ncell
    if 2nl + 1 > n
        c.n_full += 1
        return -1
    end
    ix = _cell1(sys.com[1, k], c.L, n); iy = _cell1(sys.com[2, k], c.L, n); iz = _cell1(sys.com[3, k], c.L, n)
    m = 0
    @inbounds for dx in -nl:nl, dy in -nl:nl, dz in -nl:nl
        j = c.head[_cell_lin(mod(ix + dx, n), mod(iy + dy, n), mod(iz + dz, n), n)]
        while j > 0
            j != k && (m += 1; c.nbr[m] = j)
            j = c.next[j]
        end
    end
    return m
end

"Number of molecules whose stored cell differs from the cell of their COM, plus 1 if ext_max is below some extent
(0 = the list is current)."
function cells_check(c::MolCells, sys)::Int
    bad = count(i -> c.cell_of[i] != _cell_of(c, sys, i), 1:sys.n_mol)
    maximum(sys.ext; init=0.0) > c.ext_max && (bad += 1)
    return bad
end


# ----------------------------------------------------------------------------
# Site cell list (site cut-off)
# ----------------------------------------------------------------------------
"""
Site-level cell list for the site cut-off (`ff.cutoff === :site`). With large molecules the COM range rc + ext_i + ext_j of
`MolCells` approaches the box (a 30–60 atom drug molecule, ext ≈ 6 Å, rc = 12 Å: ≈ 24 Å in a ≈ 38 Å box), so the cells
hold the interacting sites (LJ type > 0 or q ≠ 0) instead: every site of the row molecule k searches the cells within rc of
it. Cells have side ≥ rc/`subdiv` (search ⌈rc/side⌉ layers; subdiv = 2 searches 5³ cells of side rc/2, (2.5 rc)³ instead of
(3 rc)³ — an efficiency choice). The site–site vector is formed exactly as in `pair_energy_virial` (minimum-image COM vector
plus the two offsets), so each site-pair term, the cap rule (first overlapping pair in (site of k, site of j) order) and the
cap clamp are those of the all-pairs row; only the order of the additions within a molecule pair differs (rounding).
"""
mutable struct SiteCells
    L::Float64
    ncell::Int
    side::Float64
    rc::Float64
    subdiv::Int
    head::Vector{Int}
    next::Vector{Int}             # per site (global site index s0 + n)
    prev::Vector{Int}
    cell_of::Vector{Int}          # 0 = inactive site (no LJ, no charge)
    site_mol::Vector{Int}
    # per-row scratch (length N)
    stamp::Vector{Int}
    nstamp::Int
    touched::Vector{Int}
    Rv::Matrix{Float64}           # 3 × N minimum-image COM vectors k → j of the current row
    capkey::Vector{Int}
    capu::Vector{Float64}
    n_query::Int
    n_full::Int
end

function SiteCells(sys, ff; subdiv::Int=2)
    ns = sys.n_sites_total; N = sys.n_mol
    site_mol = zeros(Int, ns)
    for i in 1:N, n in 1:sys.templates[sys.tmpl[i]].n_sites
        site_mol[sys.site_start[i] + n] = i
    end
    c = SiteCells(0.0, 0, 0.0, ff.rc, subdiv, Int[], zeros(Int, ns), zeros(Int, ns), zeros(Int, ns), site_mol,
                  zeros(Int, N), 0, zeros(Int, N), zeros(3, N), zeros(Int, N), zeros(N), 0, 0)
    cells_rebuild!(c, sys)
    return c
end

@inline function _site_cell(c::SiteCells, sys, i::Int, n::Int)
    s = sys.site_start[i] + n
    @inbounds _cell_lin(_cell1(sys.com[1, i] + sys.site_off[1, s], c.L, c.ncell),
                        _cell1(sys.com[2, i] + sys.site_off[2, s], c.L, c.ncell),
                        _cell1(sys.com[3, i] + sys.site_off[3, s], c.L, c.ncell), c.ncell)
end
@inline _site_active(t, n::Int) = @inbounds (t.site_type[n] > 0 || t.site_q[n] != 0.0)

@inline function _scell_link!(c::SiteCells, s::Int, cell::Int)
    @inbounds begin
        h = c.head[cell]
        c.next[s] = h; c.prev[s] = 0
        h > 0 && (c.prev[h] = s)
        c.head[cell] = s; c.cell_of[s] = cell
    end
end
@inline function _scell_unlink!(c::SiteCells, s::Int)
    @inbounds begin
        p = c.prev[s]; n = c.next[s]
        p > 0 ? (c.next[p] = n) : (c.head[c.cell_of[s]] = n)
        n > 0 && (c.prev[n] = p)
    end
end

function cells_rebuild!(c::SiteCells, sys)
    length(c.stamp) == sys.n_mol || error("cells_rebuild!: site cell list built for $(length(c.stamp)) molecules, system has $(sys.n_mol)")
    if sys.L != c.L || isempty(c.head)
        c.L = sys.L
        c.ncell = max(1, floor(Int, c.L * c.subdiv / c.rc))
        c.side = c.L / c.ncell
        c.head = zeros(Int, c.ncell^3)
    end
    fill!(c.head, 0); fill!(c.cell_of, 0)
    @inbounds for i in 1:sys.n_mol
        t = sys.templates[sys.tmpl[i]]
        for n in 1:t.n_sites
            _site_active(t, n) && _scell_link!(c, sys.site_start[i] + n, _site_cell(c, sys, i, n))
        end
    end
    return c
end

function cells_moved!(c::SiteCells, sys, i::Int)
    t = sys.templates[sys.tmpl[i]]
    @inbounds for n in 1:t.n_sites
        _site_active(t, n) || continue
        s = sys.site_start[i] + n
        cell = _site_cell(c, sys, i, n)
        cell != c.cell_of[s] && (_scell_unlink!(c, s); _scell_link!(c, s, cell))
    end
    return nothing
end

function cells_check(c::SiteCells, sys)::Int
    bad = 0
    for i in 1:sys.n_mol
        t = sys.templates[sys.tmpl[i]]
        for n in 1:t.n_sites
            s = sys.site_start[i] + n
            want = _site_active(t, n) ? _site_cell(c, sys, i, n) : 0
            c.cell_of[s] == want || (bad += 1)
        end
    end
    return bad
end

@inline _cells_for(c::SiteCells, ff) = c.rc == ff.rc && ff.cutoff === :site

"""
    site_cells_row!(row_u, row_w, c, sys, ff, k) -> Bool

Real-space row of molecule k (current pose) from the site cell list; false (row untouched) if the search would cover the box.
"""
function site_cells_row!(row_u::Vector{Float64}, row_w::Vector{Float64}, c::SiteCells, sys, ff, k::Int)::Bool
    c.n_query += 1
    nl = ceil(Int, c.rc / c.side)
    n = c.ncell
    if 2nl + 1 > n
        c.n_full += 1
        return false
    end
    L = sys.L; rc2 = ff.rc2
    tk = sys.templates[sys.tmpl[k]]; sk = sys.site_start[k]
    c.nstamp += 1; st = c.nstamp; nt = 0
    capping = ff.cap_overlaps
    siteimg = _site_images(ff, L)
    @inbounds for a in 1:tk.n_sites
        _site_active(tk, a) || continue
        ox, oy, oz = sys.site_off[1, sk + a], sys.site_off[2, sk + a], sys.site_off[3, sk + a]
        tyn = tk.site_type[a]; qn = tk.site_q[a]
        ix = _cell1(sys.com[1, k] + ox, c.L, n); iy = _cell1(sys.com[2, k] + oy, c.L, n); iz = _cell1(sys.com[3, k] + oz, c.L, n)
        for dx in -nl:nl, dy in -nl:nl, dz in -nl:nl
            s = c.head[_cell_lin(mod(ix + dx, n), mod(iy + dy, n), mod(iz + dz, n), n)]
            while s > 0
                j = c.site_mol[s]
                if j != k
                    if c.stamp[j] != st                  # first site of j in this row: COM vector, reset accumulators
                        c.stamp[j] = st; nt += 1; c.touched[nt] = j
                        c.Rv[1, j] = _min_image(sys.com[1, j] - sys.com[1, k], L)
                        c.Rv[2, j] = _min_image(sys.com[2, j] - sys.com[2, k], L)
                        c.Rv[3, j] = _min_image(sys.com[3, j] - sys.com[3, k], L)
                        row_u[j] = 0.0; row_w[j] = 0.0; c.capkey[j] = typemax(Int)
                    end
                    Rx = c.Rv[1, j]; Ry = c.Rv[2, j]; Rz = c.Rv[3, j]
                    ddx = Rx + sys.site_off[1, s] - ox
                    ddy = Ry + sys.site_off[2, s] - oy
                    ddz = Rz + sys.site_off[3, s] - oz
                    if siteimg && _pair_wraps(Rx, Ry, Rz, sys.ext[k] + sys.ext[j], L)   # as pair_energy_virial
                        kx = L * round(ddx / L); ky = L * round(ddy / L); kz = L * round(ddz / L)
                        ddx -= kx; ddy -= ky; ddz -= kz; Rx -= kx; Ry -= ky; Rz -= kz
                    end
                    r2 = ddx * ddx + ddy * ddy + ddz * ddz
                    if r2 < rc2
                        tj = sys.templates[sys.tmpl[j]]; m = s - sys.site_start[j]
                        tym = tj.site_type[m]
                        if capping && tyn > 0 && tym > 0 && r2 < ff.rcap2_mix[tyn, tym]
                            key = (a - 1) * tj.n_sites + m       # all-pairs order: first (site of k, site of j)
                            key < c.capkey[j] && (c.capkey[j] = key; c.capu[j] = ff.ucap_mix[tyn, tym])
                        else
                            u, fr = _site_pair_uf(ff, tyn, qn, tym, tj.site_q[m], r2)
                            row_u[j] += u
                            row_w[j] += (Rx * ddx + Ry * ddy + Rz * ddz) / r2 * fr
                        end
                    end
                end
                s = c.next[s]
            end
        end
    end
    # every other molecule is beyond rc at every site pair: zero (as in the all-pairs row)
    @inbounds for j in 1:sys.n_mol
        c.stamp[j] == st || (row_u[j] = 0.0; row_w[j] = 0.0)
    end
    if capping
        @inbounds for t in 1:nt
            j = c.touched[t]
            if c.capkey[j] != typemax(Int)
                row_u[j] = c.capu[j]; row_w[j] = 0.0
            elseif row_u[j] > ff.cap_min_u
                row_u[j] = ff.cap_min_u; row_w[j] = 0.0
            end
        end
    end
    return true
end

"""
    attach_cells!(sys, ff; subdiv=2) -> MolCells or SiteCells

Use a cell list for the real-space rows of `sys` (eKMC events, candidate rows): molecular cells for the molecule cut-off,
site cells for the site cut-off. `detach_cells!` removes it.
"""
attach_cells!(sys, ff; subdiv::Int=2) = (sys.cells = ff.cutoff === :molecule ? MolCells(sys, ff) : SiteCells(sys, ff; subdiv=subdiv))
detach_cells!(sys) = (sys.cells = nothing; nothing)

"""
Smooth particle-mesh Ewald inside eKMC and MC of rigid/flexible molecules (DESIGN §13.3 E5, JOURNAL 122): the mesh path.

Hamiltonian (the PGP split of Zhao, Yu & MacKerell, J. Chem. Phys. 165 (2026) 084104, applied to every molecule):

    E = Σ_{m<n} [real_mn + ψ_mn] + Σ_m e_m,     ψ_mn = Q_m · Φ_n  (the SPME cross term of molecules m and n),
    e_m = ½ Σ_k c_k |S_m(k)|² − ke α/√π Σ_a q_a² − ke Σ_{a<b ∈ m} q_a q_b erf(α r_ab)/r_ab   (exact Ewald image term).

Q_m is molecule m's charge spread on the K³ mesh with cardinal B-splines of order p (Essmann et al., J. Chem. Phys. 103
(1995) 8577, Eq. 4.1), Φ_n = brfft(2G ∘ rfft(Q_n)) its mesh potential with the SPME influence function G = `pme_influence`
(E_rec = Σ_m G(m)|F[Q](m)|² = ½ Q·Φ), so ψ_mn is exactly the m–n term of the SPME reciprocal energy (PGP Eqs. 14–20: the
cross term on the mesh is algebraically the SPME one). The per-molecule term is the exact Ewald image energy e_m of
EwaldRecip.jl (k-vectors from ewald-rtol), not the mesh self-energy ½ Q_m·Φ_m: PGP omits the mobile molecule's mesh
self-term because it carries the mesh artefact (translation- and orientation-dependent; ≤ 1.8e-3 kJ/mol for rigid water at
64³, order 4, PGP §III). E therefore equals the GROMACS PME energy (same α, K, p) minus Σ_m (½ Q_m·Φ_m − ½ Σ c |S_m|²), the
mesh self-artefact, which vanishes as the mesh is refined (checked against GROMACS PME reruns in test_pme_ekmc.jl).

eKMC bookkeeping (every rate changes after every event, only through the moved molecule): committed mesh Q = Σ_m Q_m and
Φ = G⋆Q; per site its stencil (base index, B-spline weights). When k moves (`_recip_commit!`): ΔQ_k is spread (old stencil
with −, new with +), ΔΦ = G⋆ΔQ_k (one rfft/brfft pair) is added to Φ and kept in `dΦ` for the candidate updates, and
Φ_k = G⋆Q_k(new) (a second pair) gives the new row ψ_kj = Q_j·Φ_k by interpolation at every committed stencil
(O(N n p³)). A stored candidate pose c of molecule i has reciprocal energy Q_c·(Φ − Φ_i) + e_c; after a move of k it changes
by Q_c·ΔΦ_k (interpolation at the candidate's stored stencils, O(n p³) per candidate) — the O(N_cand · N_k) structure-factor
product of the Ewald path becomes O(N_cand · n p³) plus two FFT pairs per event. Everything is Float64 (`precision = :double`);
the debug rebuild is the check.
"""

using FFTW

mutable struct PMERecip
    L::Float64
    α::Float64
    ke::Float64
    K::Int
    p::Int
    ew::EwaldRecip                    # exact Ewald image terms (recip_sf!, recip_corr, recip_half_norm only)
    Gh::Array{Float64,3}              # 2G on the rfft half grid: Φ = brfft(Gh ∘ rfft(Q), K)
    fplan::Any
    bplan::Any
    Q::Array{Float64,3}               # committed total charge mesh
    Φ::Array{Float64,3}               # its potential
    Qw::Array{Float64,3}              # work mesh
    Fw::Array{ComplexF64,3}           # work half spectrum
    Φw::Array{Float64,3}              # potential of a molecule at an uncommitted pose
    Φk::Array{Float64,3}              # potential of molecule mk alone at its committed pose
    mk::Int                           # 0 = Φk invalid
    dΦ::Array{Float64,3}              # potential of ΔQ of the last commit
    Φpin::Array{Float64,3}            # potential of the pinned molecule (the ladder's fractional one), kept current
    pin::Int                          # 0 = none
    base::Matrix{Int}                 # 3 × n_sites_total committed stencil base (0-based mesh index of the lowest node)
    θ::Array{Float64,3}               # p × 3 × n_sites_total committed B-spline weights
    wbase::Matrix{Int}                # 3 × n_max stencils of one molecule at an uncommitted pose
    wθ::Array{Float64,3}
    θs::Vector{Float64}               # p scratch
    eimg::Vector{Float64}             # committed exact image energies e_m
    corr::Vector{Float64}             # committed ke α/√π Σq² + ke Σ_{a<b} q_a q_b erf(α r)/r
    mself::Vector{Float64}            # committed mesh self energies ½ Q_m·Φ_m (for the total energy only)
    last::Int                         # molecule of the last commit (0 after a rebuild)
    n_commit::Int
    precision::Symbol                 # :double
    backend::Symbol                   # :cpu, or the device of the gather engine (`dev`)
    dev::Any                          # PMEDevice: stencil gathers on a KernelAbstractions device (nothing = host loops)
    Gvh::Array{Float64,3}             # virial kernel G(1 − k²/2α²) on the rfft half grid, with the half-spectrum weights
    wself::Vector{Float64}            # committed per-molecule mesh self-virial (−d(½Q_m·Φ_m)/d ln L)
    wimg::Vector{Float64}             # committed per-molecule image-term virial (recip_virial_molecule)
    gθ::Matrix{Float64}; gdθ::Matrix{Float64}; gθ1::Vector{Float64}   # scratch of _pme_grad
end

"Candidate stencils (sites of stored candidate poses): column c holds up to `nmax` sites."
mutable struct PMEStencils
    nmax::Int
    base::Array{Int,3}                # 3 × nmax × ncol
    θ::Array{Float64,4}               # p × 3 × nmax × ncol
    q::Matrix{Float64}                # nmax × ncol (0 = unused site)
    dev::Any                          # device mirror (PMEColumns) for the gather engine; nothing = host only
end
PMEStencils(p::Int, nmax::Int, ncol::Int) = PMEStencils(nmax, zeros(Int, 3, nmax, ncol), zeros(p, 3, nmax, ncol), zeros(nmax, ncol), nothing)

n_kvectors(R::PMERecip) = n_kvectors(R.ew)
pme_mesh_size(R::PMERecip) = R.K

function _pme_halfG(L::Float64, α::Float64, ke::Float64, K::Int, p::Int)
    G = pme_influence(PMEParams(L, α, ke, K, p))
    return 2.0 .* G[1:K÷2+1, :, :]
end

"""
    PMERecip(sys; α, ke, K, p, n2max)

Mesh state for `sys` (committed with `recip_rebuild!`). K: mesh points per edge; p: B-spline order; n2max: the k-vector set
of the exact image terms (`ewald_n2max_for`).
"""
function PMERecip(sys; α::Float64, ke::Float64, K::Int, p::Int, n2max::Int)
    ew = EwaldRecip(sys; α=α, ke=ke, n2max=n2max)
    nst = sys.n_sites_total
    nmax = maximum(t.n_sites for t in sys.templates)
    Q = zeros(K, K, K)
    Fw = zeros(ComplexF64, K ÷ 2 + 1, K, K)
    fplan = plan_rfft(Q); bplan = plan_brfft(Fw, K)
    N = sys.n_mol
    R = PMERecip(sys.L, α, ke, K, p, ew, _pme_halfG(sys.L, α, ke, K, p), fplan, bplan, Q, zeros(K, K, K), zeros(K, K, K),
                 Fw, zeros(K, K, K), zeros(K, K, K), 0, zeros(K, K, K), zeros(K, K, K), 0,
                 zeros(Int, 3, nst), zeros(p, 3, nst), zeros(Int, 3, nmax), zeros(p, 3, nmax), zeros(p),
                 zeros(N), zeros(N), zeros(N), 0, 0, :double, :cpu, nothing,
                 _pme_virial_halfkernel(sys.L, α, ke, K, p), zeros(N), zeros(N), zeros(p, 3), zeros(p, 3), zeros(p))
    return R
end

# stencil of a point (Essmann: u = K x/L wrapped into [0, K), base = floor(u), weights M_p(u − base + j), j = 0…p−1;
# node of weight j is base − j (mod K), as `pme_spread!`)
@inline function _pme_stencil!(base::AbstractMatrix{Int}, θ::AbstractArray{Float64,3}, s::Int, x::Float64, y::Float64,
                               z::Float64, R::PMERecip)
    K = R.K
    @inbounds for (d, c) in ((1, x), (2, y), (3, z))
        u = K * c / R.L
        u -= K * floor(u / K)
        b = floor(Int, u)
        b >= K && (b -= K)
        base[d, s] = b
        _bspline!(R.θs, u - b, R.p)
        for j in 1:R.p; θ[j, d, s] = R.θs[j]; end
    end
    return nothing
end

# Σ over the stencil of site s of w·M (spread with factor f into M when `add`, else the interpolation of M)
@inline function _pme_site!(M::Array{Float64,3}, base::AbstractMatrix{Int}, θ::AbstractArray{Float64,3}, s::Int,
                            f::Float64, R::PMERecip, add::Bool)::Float64
    K = R.K; p = R.p
    bx = base[1, s]; by = base[2, s]; bz = base[3, s]
    acc = 0.0
    @inbounds for jz in 0:p-1
        gz = bz - jz; gz < 0 && (gz += K)
        wz = θ[jz+1, 3, s]
        for jy in 0:p-1
            gy = by - jy; gy < 0 && (gy += K)
            wyz = wz * θ[jy+1, 2, s]
            for jx in 0:p-1
                gx = bx - jx; gx < 0 && (gx += K)
                w = wyz * θ[jx+1, 1, s]
                if add
                    M[gx+1, gy+1, gz+1] += f * w
                else
                    acc += w * M[gx+1, gy+1, gz+1]
                end
            end
        end
    end
    return acc
end

"Stencils of molecule i at its current pose (COM + site_off) into (base, θ) columns off+1 … off+n."
function _pme_mol_stencils!(base, θ, off::Int, R::PMERecip, sys, i::Int)
    t = sys.templates[sys.tmpl[i]]; s0 = sys.site_start[i]
    @inbounds for a in 1:t.n_sites
        _pme_stencil!(base, θ, off + a, sys.com[1, i] + sys.site_off[1, s0 + a], sys.com[2, i] + sys.site_off[2, s0 + a],
                      sys.com[3, i] + sys.site_off[3, s0 + a], R)
    end
    return nothing
end

"Σ_a q_a Φ(r_a) over the sites off+1 … off+n with charges of molecule i's template."
function _pme_gather(Φ::Array{Float64,3}, base, θ, off::Int, R::PMERecip, t)::Float64
    s = 0.0
    @inbounds for a in 1:t.n_sites
        q = t.site_q[a]
        q == 0.0 && continue
        s += q * _pme_site!(Φ, base, θ, off + a, 0.0, R, false)
    end
    return s
end

function _pme_spread_mol!(M::Array{Float64,3}, base, θ, off::Int, R::PMERecip, t, f::Float64)
    @inbounds for a in 1:t.n_sites
        q = t.site_q[a]
        q == 0.0 && continue
        _pme_site!(M, base, θ, off + a, f * q, R, true)
    end
    return nothing
end

"out = G⋆M (mesh potential of the charge mesh M)."
function _pme_potential!(out::Array{Float64,3}, R::PMERecip, M::Array{Float64,3})
    mul!(R.Fw, R.fplan, M)
    R.Fw .*= R.Gh
    mul!(out, R.bplan, R.Fw)
    return out
end

"Φ_i of molecule i alone at its committed pose (cached in Φk / Φpin)."
function pme_mol_potential!(R::PMERecip, sys, i::Int)
    i == R.pin && return R.Φpin
    i == R.mk && return R.Φk
    t = sys.templates[sys.tmpl[i]]
    fill!(R.Qw, 0.0)
    _pme_spread_mol!(R.Qw, R.base, R.θ, sys.site_start[i], R, t, 1.0)
    _pme_potential!(R.Φk, R, R.Qw)
    R.mk = i
    return R.Φk
end

"Keep Φ_s of molecule s (the ladder's fractional one) current in R.Φpin."
function pme_pin!(R::PMERecip, sys, s::Int)
    s == R.pin && return R.Φpin
    R.pin = 0
    if s > 0
        Φs = pme_mol_potential!(R, sys, s)
        R.Φpin === Φs || copyto!(R.Φpin, Φs)
        R.pin = s
    end
    return R.Φpin
end

"Exact Ewald image term and its correction of molecule i at its current pose."
function _pme_image_now(R::PMERecip, sys, i::Int)
    ew = R.ew
    recip_sf!(ew.work, ew, sys, i)
    c = recip_corr(ew, sys, i)
    return recip_half_norm(ew, ew.work) - c, c
end
recip_image_now(R::PMERecip, sys, i::Int)::Float64 = first(_pme_image_now(R, sys, i))

function recip_rebuild!(R::PMERecip, sys)
    if sys.L != R.L
        R.L = sys.L
        R.Gh = _pme_halfG(R.L, R.α, R.ke, R.K, R.p)
        R.Gvh = _pme_virial_halfkernel(R.L, R.α, R.ke, R.K, R.p)
        R.ew.L = sys.L
        R.ew.c = _recip_weights(R.ew.L, R.ew.α, R.ew.ke, R.ew.nx, R.ew.ny, R.ew.nz); R.ew.cd = R.ew.c
    end
    N = sys.n_mol
    length(R.eimg) == N || error("recip_rebuild!: PMERecip built for $(length(R.eimg)) molecules, system has $N")
    fill!(R.Q, 0.0)
    for i in 1:N
        t = sys.templates[sys.tmpl[i]]; off = sys.site_start[i]
        _pme_mol_stencils!(R.base, R.θ, off, R, sys, i)
        _pme_spread_mol!(R.Q, R.base, R.θ, off, R, t, 1.0)
        R.eimg[i], R.corr[i] = _pme_image_now(R, sys, i)
    end
    _pme_potential!(R.Φ, R, R.Q)
    R.mk = 0
    pin = R.pin; R.pin = 0
    for i in 1:N                                       # mesh self energies and self-virials: one FFT pair per molecule
        t = sys.templates[sys.tmpl[i]]
        fill!(R.Qw, 0.0)
        _pme_spread_mol!(R.Qw, R.base, R.θ, sys.site_start[i], R, t, 1.0)
        _pme_selfvirial_potential!(R, sys, i)            # R.Φk = Φ_i, R.wself[i]
        R.mk = i
        R.mself[i] = 0.5 * _pme_gather(R.Φk, R.base, R.θ, sys.site_start[i], R, t)
        R.wimg[i] = recip_virial_molecule(R.ew, sys, i)
    end
    R.mk = 0
    pin > 0 && pme_pin!(R, sys, pin)
    R.last = 0
    if R.dev !== nothing
        for i in 1:N; _pme_dev_molecule!(R, sys, i); end
        R.dev.mk_up = -1; R.dev.n_up = -1
    end
    return nothing
end

"Molecule i has been moved: ΔQ_i, ΔΦ (kept in R.dΦ), Φ, Φ_i (in R.Φk), stencils, image energy."
function _recip_commit!(R::PMERecip, sys, i::Int)
    t = sys.templates[sys.tmpl[i]]; off = sys.site_start[i]
    fill!(R.Qw, 0.0)
    _pme_spread_mol!(R.Qw, R.base, R.θ, off, R, t, -1.0)          # old stencils
    _pme_mol_stencils!(R.base, R.θ, off, R, sys, i)                 # new stencils
    _pme_spread_mol!(R.Qw, R.base, R.θ, off, R, t, 1.0)
    R.Q .+= R.Qw
    _pme_potential!(R.dΦ, R, R.Qw)
    R.Φ .+= R.dΦ
    fill!(R.Qw, 0.0)
    _pme_spread_mol!(R.Qw, R.base, R.θ, off, R, t, 1.0)
    _pme_selfvirial_potential!(R, sys, i)                # R.Φk = Φ_i and its mesh self-virial (E6)
    R.wimg[i] = recip_virial_molecule(R.ew, sys, i)
    R.mk = i
    i == R.pin && copyto!(R.Φpin, R.Φk)
    R.mself[i] = 0.5 * _pme_gather(R.Φk, R.base, R.θ, off, R, t)
    R.eimg[i], R.corr[i] = _pme_image_now(R, sys, i)
    R.last = i
    R.n_commit += 1
    R.dev === nothing || _pme_dev_molecule!(R, sys, i)
    return nothing
end

"row[j] += ψ_kj = Q_j·Φ_k for every j, Φ_k from the potential mesh `Φk` of molecule k."
function _pme_row!(row::AbstractVector{Float64}, R::PMERecip, sys, Φk::Array{Float64,3})
    if R.dev !== nothing && Φk === R.Φk && R.mk > 0               # device gather over the N committed molecule columns
        g = _pme_dev_gather!(R.dev, R.dev.mol, _pme_dev_sync_k!(R), sys.n_mol, R)
        @inbounds for j in 1:sys.n_mol; row[j] += g[j]; end
        return nothing
    end
    @inbounds for j in 1:sys.n_mol
        row[j] += _pme_gather(Φk, R.base, R.θ, sys.site_start[j], R, sys.templates[sys.tmpl[j]])
    end
    return nothing
end

"As `recip_row_add!` (EwaldRecip.jl): k at its current (possibly uncommitted) pose, the others committed; row[k] meaningless."
function recip_row_add!(row::AbstractVector{Float64}, R::PMERecip, sys, k::Int)
    t = sys.templates[sys.tmpl[k]]
    _pme_mol_stencils!(R.wbase, R.wθ, 0, R, sys, k)
    fill!(R.Qw, 0.0)
    _pme_spread_mol!(R.Qw, R.wbase, R.wθ, 0, R, t, 1.0)
    _pme_potential!(R.Φw, R, R.Qw)
    _pme_row!(row, R, sys, R.Φw)
end

recip_row_add_committed!(row::AbstractVector{Float64}, R::PMERecip, k::Int) =
    error("recip_row_add_committed!(row, R::PMERecip, k) needs sys; use recip_row_add_committed!(row, R, sys, k)")
function recip_row_add_committed!(row::AbstractVector{Float64}, R::PMERecip, sys, k::Int)
    _pme_row!(row, R, sys, pme_mol_potential!(R, sys, k))
end

"Σ_{j≠i} ψ_ij + e_i of molecule i at its current pose, the rest committed: Q_i(now)·(Φ − Φ_i(committed)) + e_i(now)."
function recip_removal(R::PMERecip, sys, i::Int)::Float64
    t = sys.templates[sys.tmpl[i]]
    Φi = pme_mol_potential!(R, sys, i)
    _pme_mol_stencils!(R.wbase, R.wθ, 0, R, sys, i)
    s = _pme_gather(R.Φ, R.wbase, R.wθ, 0, R, t) - _pme_gather(Φi, R.wbase, R.wθ, 0, R, t)
    return s + recip_image_now(R, sys, i)
end

"Committed Σ_{m<n} ψ_mn + Σ_m e_m = ½ Q·Φ − Σ_m ½ Q_m·Φ_m + Σ_m e_m (needs the stencils of every site: `sys` passed)."
recip_energy_total(R::PMERecip) = error("recip_energy_total(R::PMERecip) needs sys")
function recip_energy_total(R::PMERecip, sys)
    E = 0.0
    for i in 1:sys.n_mol
        E += 0.5 * _pme_gather(R.Φ, R.base, R.θ, sys.site_start[i], R, sys.templates[sys.tmpl[i]])
    end
    return E - sum(R.mself) + sum(R.eimg)
end

"N × N matrix of the committed cross terms ψ_mn (diagonal 0): one FFT pair per molecule (rebuilds only)."
function recip_pair_matrix(R::PMERecip, sys)
    N = sys.n_mol
    Ψ = zeros(N, N)
    pin = R.pin; R.pin = 0
    for n in 1:N
        Φn = pme_mol_potential!(R, sys, n)
        for m in 1:N
            m == n && continue
            Ψ[m, n] = _pme_gather(Φn, R.base, R.θ, sys.site_start[m], R, sys.templates[sys.tmpl[m]])
        end
    end
    R.mk = 0
    pin > 0 && pme_pin!(R, sys, pin)
    return Ψ
end

# --- candidate poses (LocalMoves, SolventLocalMoves) ---

"Store the stencils of molecule i at its current (placed candidate) pose in column `col` of `P`."
function pme_cand_store!(P::PMEStencils, col::Int, R::PMERecip, sys, i::Int)
    t = sys.templates[sys.tmpl[i]]; s0 = sys.site_start[i]
    bview = view(P.base, :, :, col); θview = view(P.θ, :, :, :, col)
    @inbounds for a in 1:P.nmax
        P.q[a, col] = a <= t.n_sites ? t.site_q[a] : 0.0
    end
    @inbounds for a in 1:t.n_sites
        _pme_stencil!(bview, θview, a, sys.com[1, i] + sys.site_off[1, s0 + a], sys.com[2, i] + sys.site_off[2, s0 + a],
                      sys.com[3, i] + sys.site_off[3, s0 + a], R)
    end
    return nothing
end

"Σ_a q_a Φ(r_a) over the stored candidate column `col`."
function pme_cand_gather(P::PMEStencils, col::Int, Φ::Array{Float64,3}, R::PMERecip)::Float64
    bview = view(P.base, :, :, col); θview = view(P.θ, :, :, :, col)
    s = 0.0
    @inbounds for a in 1:P.nmax
        q = P.q[a, col]
        q == 0.0 && continue
        s += q * _pme_site!(Φ, bview, θview, a, 0.0, R, false)
    end
    return s
end

pme_cand_clear!(P::PMEStencils, col::Int) = (@inbounds for a in 1:P.nmax; P.q[a, col] = 0.0; end; nothing)

"""
    attach_pme!(sys, ff; rtol=1e-5, spacing, K=0, order=4, n2max=0)

Give `sys` the SPME mesh path for a force field with `coulomb = :ewald` (α = ff.wolf.α, ke = ff.wolf.ke; the real-space
erfc terms are the Ewald ones). Mesh: K points per edge, or the smallest FFT-friendly K with L/K ≤ `spacing` (reduced
length; GROMACS fourier-spacing, default 0.12 nm); B-spline order `order` (GROMACS pme-order, default 4). The exact image
terms use the k-vectors of `rtol` (`ewald_n2max_for`) unless `n2max` is given. Returns the PMERecip.
"""
function attach_pme!(sys, ff; rtol::Float64=1e-5, spacing::Float64=NaN, K::Int=0, order::Int=4, n2max::Int=0)
    ff.coulomb === :ewald || error("attach_pme!: force field coulomb = $(ff.coulomb), need :ewald")
    K > 0 || spacing > 0 || error("attach_pme!: give K or spacing")
    Km = K > 0 ? K : pme_grid_size(sys.L / spacing)
    α = ff.wolf.α
    n2 = n2max > 0 ? n2max : ewald_n2max_for(sys.L, α, rtol)
    R = PMERecip(sys; α=α, ke=ff.wolf.ke, K=Km, p=order, n2max=n2)
    sys.recip = R
    recip_rebuild!(R, sys)
    return R
end

# --- gather engine on a KernelAbstractions device (DESIGN §16, JOURNAL 123) ---
# The event logic, the spreads and the FFTs stay on the host (mesh sizes of the target systems: K ≲ 40); the stencil
# interpolations — the row of the moved molecule over all N committed molecules and the update of every stored solvent
# candidate by ΔΦ — run as one launch each on the device, over device mirrors of the committed molecule stencils and of the
# candidate stencils, with Φ_k and ΔΦ uploaded once per commit. Float64 throughout (:double; the host rebuild is the check).

using KernelAbstractions
const PKA = KernelAbstractions

"Device mirror of a set of stencil columns (base 3 × nmax × ncol Int32, θ p × 3 × nmax × ncol, q nmax × ncol)."
mutable struct PMEColumns
    base::Any; θ::Any; q::Any; out::Any
    hout::Vector{Float64}
end

mutable struct PMEDevice
    backend::Symbol
    kab::Any
    Φk::Any; dΦ::Any                  # device copies of R.Φk (molecule mk_up) and R.dΦ (commit n_up)
    mk_up::Int; n_up::Int
    mol::PMEColumns                   # committed stencils, one column per molecule
    nmax::Int
    hb::Array{Int32,3}; hθ::Array{Float64,4}; hq::Matrix{Float64}   # host staging of one column
end

function _pme_columns(backend::Symbol, p::Int, nmax::Int, ncol::Int)
    PMEColumns(_dp_array(backend, zeros(Int32, 3, nmax, ncol)), _dp_array(backend, zeros(p, 3, nmax, ncol)),
               _dp_array(backend, zeros(nmax, ncol)), _dp_array(backend, zeros(ncol)), zeros(ncol))
end

@kernel function _pme_gather_kernel!(out, @Const(Φ), @Const(base), @Const(θ), @Const(q), K::Int32, p::Int32, nmax::Int32)
    c = @index(Global, Linear)
    acc = 0.0
    @inbounds for a in 1:Int(nmax)
        qa = q[a, c]
        if qa != 0.0
            bx = Int(base[1, a, c]); by = Int(base[2, a, c]); bz = Int(base[3, a, c]); KK = Int(K)
            s = 0.0
            for jz in 0:Int(p)-1
                gz = bz - jz; gz < 0 && (gz += KK)
                wz = θ[jz+1, 3, a, c]
                for jy in 0:Int(p)-1
                    gy = by - jy; gy < 0 && (gy += KK)
                    wyz = wz * θ[jy+1, 2, a, c]
                    for jx in 0:Int(p)-1
                        gx = bx - jx; gx < 0 && (gx += KK)
                        s += wyz * θ[jx+1, 1, a, c] * Φ[gx+1, gy+1, gz+1]
                    end
                end
            end
            acc += qa * s
        end
    end
    @inbounds out[c] = acc
end

"Gather Φ (device) at columns 1…ncol of C; returns the host vector (first ncol entries valid)."
function _pme_dev_gather!(D::PMEDevice, C::PMEColumns, Φ, ncol::Int, R::PMERecip)
    ncol == 0 && return C.hout
    k = _pme_gather_kernel!(D.kab, 64)
    k(C.out, Φ, C.base, C.θ, C.q, Int32(R.K), Int32(R.p), Int32(size(C.q, 1)); ndrange=ncol)
    PKA.synchronize(D.kab)
    copyto!(C.hout, 1, C.out, 1, ncol)
    return C.hout
end

"Upload column `col` of C from host stencils (base, θ) sites off+1 … off+n and charges qv (length ≤ nmax)."
function _pme_dev_column!(D::PMEDevice, C::PMEColumns, col::Int, base, θ, off::Int, n::Int, qv, R::PMERecip)
    fill!(D.hb, Int32(0)); fill!(D.hθ, 0.0); fill!(D.hq, 0.0)
    @inbounds for a in 1:n
        for d in 1:3
            D.hb[d, a, 1] = Int32(base[d, off + a])
            for j in 1:R.p; D.hθ[j, d, a, 1] = θ[j, d, off + a]; end
        end
        D.hq[a, 1] = qv[a]
    end
    nm = D.nmax
    copyto!(C.base, 3 * nm * (col - 1) + 1, D.hb, 1, 3 * nm)
    copyto!(C.θ, R.p * 3 * nm * (col - 1) + 1, D.hθ, 1, R.p * 3 * nm)
    copyto!(C.q, nm * (col - 1) + 1, D.hq, 1, nm)
    return nothing
end

function _pme_dev_molecule!(R::PMERecip, sys, i::Int)
    t = sys.templates[sys.tmpl[i]]
    _pme_dev_column!(R.dev, R.dev.mol, i, R.base, R.θ, sys.site_start[i], t.n_sites, t.site_q, R)
end

"Make the device copies of Φk (molecule R.mk) and dΦ (commit R.n_commit) current."
function _pme_dev_sync_k!(R::PMERecip)
    D = R.dev
    tag = R.mk + R.n_commit * (length(R.eimg) + 1)   # Φk content = (molecule, commits so far): unique per content
    D.mk_up == tag || (copyto!(D.Φk, R.Φk); D.mk_up = tag)
    return D.Φk
end
function _pme_dev_sync_d!(R::PMERecip)
    D = R.dev
    D.n_up == R.n_commit || (copyto!(D.dΦ, R.dΦ); D.n_up = R.n_commit)
    return D.dΦ
end

"""
    pme_attach_device!(R, sys; backend=:cuda)

Run the stencil gathers of the SPME mesh path (rows, solvent-candidate updates) on `backend` (:cpu = the KernelAbstractions CPU
backend, the test oracle; :cuda with `using CUDA`).
"""
function pme_attach_device!(R::PMERecip, sys; backend::Symbol=:cuda)
    nmax = maximum(t.n_sites for t in sys.templates); K = R.K
    D = PMEDevice(backend, device_backend(backend), _dp_array(backend, zeros(K, K, K)), _dp_array(backend, zeros(K, K, K)),
                  -1, -1, _pme_columns(backend, R.p, nmax, sys.n_mol), nmax,
                  zeros(Int32, 3, nmax, 1), zeros(R.p, 3, nmax, 1), zeros(nmax, 1))
    R.dev = D; R.backend = backend
    for i in 1:sys.n_mol; _pme_dev_molecule!(R, sys, i); end
    return D
end

"Device mirror of candidate stencils P (all columns uploaded)."
function _pme_cand_dev!(P::PMEStencils, R::PMERecip)
    D = R.dev
    if P.dev === nothing || size(P.dev.q, 2) != size(P.q, 2) || size(P.dev.q, 1) != D.nmax
        P.dev = _pme_columns(D.backend, R.p, D.nmax, size(P.q, 2))
        for c in 1:size(P.q, 2); _pme_cand_dev_column!(P, c, R); end
    end
    return P.dev
end
function _pme_cand_dev_column!(P::PMEStencils, col::Int, R::PMERecip)
    D = R.dev
    n = min(P.nmax, D.nmax)
    fill!(D.hb, Int32(0)); fill!(D.hθ, 0.0); fill!(D.hq, 0.0)
    @inbounds for a in 1:n
        for d in 1:3
            D.hb[d, a, 1] = Int32(P.base[d, a, col])
            for j in 1:R.p; D.hθ[j, d, a, 1] = P.θ[j, d, a, col]; end
        end
        D.hq[a, 1] = P.q[a, col]
    end
    C = P.dev; nm = D.nmax
    copyto!(C.base, 3 * nm * (col - 1) + 1, D.hb, 1, 3 * nm)
    copyto!(C.θ, R.p * 3 * nm * (col - 1) + 1, D.hθ, 1, R.p * 3 * nm)
    copyto!(C.q, nm * (col - 1) + 1, D.hq, 1, nm)
    return nothing
end

# --- reciprocal molecular virial of the mesh path (E6 for PME, JOURNAL 125) ---
# W = −dH_rec/d ln L under the scaling of the molecular centres (orientations and internal coordinates fixed), for
# H_rec = E(Q) − Σ_m E(Q_m) + Σ_m e_m with E(Q) = Σ_m G(m)|F[Q](m)|² (Essmann 1995). For each mesh energy:
#   −dE/d ln L = Σ G (1 − k²/2α²) |F[Q]|²  (d ln G/d ln L = −1 + k²/2α², B(m) independent of L)
#              + Σ_a q_a ∇_u Φ̃(u_a) · (K d_a / L)   (u_a = K r_a/L, du_a/d ln L = −K d_a/L, ∂E/∂u_a = q_a ∇_u Φ̃, Φ = ∂E/∂Q)
# with the B-spline derivative dM_p(x)/dx = M_{p−1}(x) − M_{p−1}(x − 1) (Essmann Eq. 4.1); e_m: `recip_virial_molecule`.

"Virial kernel G(1 − k²/2α²) on the rfft half grid, times the half-spectrum weight (1 for i1 = 0 and i1 = K/2, else 2)."
function _pme_virial_halfkernel(L::Float64, α::Float64, ke::Float64, K::Int, p::Int)
    G = pme_influence(PMEParams(L, α, ke, K, p))
    H = zeros(K ÷ 2 + 1, K, K); tp = 2π / L
    @inbounds for i3 in 0:K-1, i2 in 0:K-1, i1 in 0:K÷2
        m1 = i1; m2 = i2 <= K ÷ 2 ? i2 : i2 - K; m3 = i3 <= K ÷ 2 ? i3 : i3 - K
        k2 = tp^2 * (m1^2 + m2^2 + m3^2)
        w = (i1 == 0 || 2i1 == K) ? 1.0 : 2.0
        H[i1+1, i2+1, i3+1] = w * G[i1+1, i2+1, i3+1] * (1 - k2 / (2α^2))
    end
    return H
end

"∇_u of the B-spline interpolation of Φ at lab point (x, y, z) (u = K r/L); allocation-free (scratch in R)."
function _pme_grad(Φ::Array{Float64,3}, R::PMERecip, x::Float64, y::Float64, z::Float64)
    K = R.K; p = R.p; θ = R.gθ; dθ = R.gdθ; θ1 = R.gθ1
    b1 = 0; b2 = 0; b3 = 0
    @inbounds for d in 1:3
        c = d == 1 ? x : (d == 2 ? y : z)
        u = K * c / R.L; u -= K * floor(u / K); bb = floor(Int, u); bb >= K && (bb -= K)
        d == 1 ? (b1 = bb) : (d == 2 ? (b2 = bb) : (b3 = bb))
        t = u - bb
        _bspline!(view(θ, :, d), t, p)
        fill!(θ1, 0.0); _bspline!(view(θ1, 1:p-1), t, p - 1)      # θ1[j+1] = M_{p−1}(t + j), j = 0 … p−2
        for j in 0:p-1
            dθ[j+1, d] = (j <= p - 2 ? θ1[j+1] : 0.0) - (j >= 1 ? θ1[j] : 0.0)
        end
    end
    g1 = 0.0; g2 = 0.0; g3 = 0.0
    @inbounds for jz in 0:p-1
        gz = b3 - jz; gz < 0 && (gz += K)
        for jy in 0:p-1
            gy = b2 - jy; gy < 0 && (gy += K)
            for jx in 0:p-1
                gx = b1 - jx; gx < 0 && (gx += K)
                v = Φ[gx+1, gy+1, gz+1]
                g1 += dθ[jx+1, 1] * θ[jy+1, 2] * θ[jz+1, 3] * v
                g2 += θ[jx+1, 1] * dθ[jy+1, 2] * θ[jz+1, 3] * v
                g3 += θ[jx+1, 1] * θ[jy+1, 2] * dθ[jz+1, 3] * v
            end
        end
    end
    return g1, g2, g3
end

"Σ_{sites of molecules mols} q_a ∇Φ̃(u_a)·(K d_a/L) (the stencil part of −dE/d ln L)."
function _pme_grad_virial(R::PMERecip, Φ::Array{Float64,3}, sys, mols)
    W = 0.0; s = R.K / R.L
    for i in mols
        t = sys.templates[sys.tmpl[i]]; s0 = sys.site_start[i]
        @inbounds for a in 1:t.n_sites
            q = t.site_q[a]
            q == 0.0 && continue
            dx = sys.site_off[1, s0 + a]; dy = sys.site_off[2, s0 + a]; dz = sys.site_off[3, s0 + a]
            g1, g2, g3 = _pme_grad(Φ, R, sys.com[1, i] + dx, sys.com[2, i] + dy, sys.com[3, i] + dz)
            W += q * s * (g1 * dx + g2 * dy + g3 * dz)
        end
    end
    return W
end

"Φ_i (into R.Φk) of the charge mesh in R.Qw (molecule i alone) and its mesh self-virial into R.wself[i] (one FFT pair)."
function _pme_selfvirial_potential!(R::PMERecip, sys, i::Int)
    mul!(R.Fw, R.fplan, R.Qw)
    w = 0.0
    @inbounds for j in eachindex(R.Fw); w += R.Gvh[j] * abs2(R.Fw[j]); end
    R.Fw .*= R.Gh
    mul!(R.Φk, R.bplan, R.Fw)
    R.wself[i] = w + _pme_grad_virial(R, R.Φk, sys, i:i)
    return R.Φk
end

"""
    recip_virial(R::PMERecip, sys) -> W_rec

Molecular reciprocal virial of the mesh-path Hamiltonian (cross terms on the mesh + exact Ewald image terms): the full mesh
energy's virial minus every molecule's own mesh virial plus the image-term virials; the per-molecule parts are maintained at
each commit (from the FFT of Φ_k) and rebuild, so a pressure costs one rfft and the stencil gradients at every site.
"""
function recip_virial(R::PMERecip, sys)::Float64
    mul!(R.Fw, R.fplan, R.Q)                              # full mesh: one rfft + the stencil gradients at every site
    W = 0.0
    @inbounds for j in eachindex(R.Fw); W += R.Gvh[j] * abs2(R.Fw[j]); end
    W += _pme_grad_virial(R, R.Φ, sys, 1:sys.n_mol)
    return W - sum(R.wself) + sum(R.wimg)                 # per-molecule parts maintained at every commit / rebuild
end

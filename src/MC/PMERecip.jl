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
    backend::Symbol                   # :cpu
end

"Candidate stencils (sites of stored candidate poses): column c holds up to `nmax` sites."
mutable struct PMEStencils
    nmax::Int
    base::Array{Int,3}                # 3 × nmax × ncol
    θ::Array{Float64,4}               # p × 3 × nmax × ncol
    q::Matrix{Float64}                # nmax × ncol (0 = unused site)
end
PMEStencils(p::Int, nmax::Int, ncol::Int) = PMEStencils(nmax, zeros(Int, 3, nmax, ncol), zeros(p, 3, nmax, ncol), zeros(nmax, ncol))

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
                 zeros(N), zeros(N), zeros(N), 0, 0, :double, :cpu)
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
    for i in 1:N                                       # mesh self energies (total energy only): one FFT pair per molecule
        Φi = pme_mol_potential!(R, sys, i)
        R.mself[i] = 0.5 * _pme_gather(Φi, R.base, R.θ, sys.site_start[i], R, sys.templates[sys.tmpl[i]])
    end
    R.mk = 0
    pin > 0 && pme_pin!(R, sys, pin)
    R.last = 0
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
    _pme_potential!(R.Φk, R, R.Qw)
    R.mk = i
    i == R.pin && copyto!(R.Φpin, R.Φk)
    R.mself[i] = 0.5 * _pme_gather(R.Φk, R.base, R.θ, off, R, t)
    R.eimg[i], R.corr[i] = _pme_image_now(R, sys, i)
    R.last = i
    R.n_commit += 1
    return nothing
end

"row[j] += ψ_kj = Q_j·Φ_k for every j, Φ_k from the potential mesh `Φk` of molecule k."
function _pme_row!(row::AbstractVector{Float64}, R::PMERecip, sys, Φk::Array{Float64,3})
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

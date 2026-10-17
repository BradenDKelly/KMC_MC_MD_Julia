"""
Reciprocal-space Ewald bookkeeping for eKMC and MC of rigid/flexible molecules (DESIGN §13, JOURNAL entry 96).

Ewald (tin-foil, cubic box, half-space wave vectors k = 2πn/L, 0 < |n|² < n2max), as in `ewald_energy`
(Electrostatics.jl, validated against the NIST SPC/E reference energies):

    E_rec = ke (4π/V) Σ_{k ∈ half} e^{−k²/4α²}/k² |S(k)|² = ½ Σ_k c_k |S(k)|²,   c_k = ke (8π/V) e^{−k²/4α²}/k²,
    S(k) = Σ_m S_m(k),   S_m(k) = Σ_{a ∈ m} q_a e^{i k·r_a}.

Split by molecules, |S|² = Σ_m |S_m|² + Σ_{m ≠ n} S_m S_n^*, so the Fourier energy is pairwise in molecules plus one
term per molecule:

    pair (m, n):   ψ_mn = Σ_k c_k Re[S_m^* S_n]                                     (added to the real-space pair energy)
    molecule m:    e_m  = ½ Σ_k c_k |S_m|² − ke α/√π Σ_a q_a² − ke Σ_{a<b ∈ m} q_a q_b erf(α r_ab)/r_ab

e_m is the energy of m with its own periodic images (Ewald energy of m alone in the box minus its vacuum Coulomb energy,
independent of the exclusion list); the vacuum intramolecular Coulomb energy of a flexible molecule stays in U_intra.
Total intermolecular Coulomb energy = Σ_{m<n} (real_mn + ψ_mn) + Σ_m e_m = `ewald_energy(...).total` (the NIST sum).

Backends (DESIGN §13.3 E2): the products run on `R.Sd`, `R.cd`, … which are the host arrays themselves (`:cpu`) or device
copies (`:cuda`, in the package extension MolSimCUDAExt, loaded with `using CUDA`; the CPU-only build never needs CUDA).
The per-molecule work (structure factor of the moved molecule, image energy) stays on the host; per event one column
(N_k complex) goes up and one row (N complex) or one candidate vector (K·M complex) comes back.
e_m depends on the pose of m only (orientation; conformation if flexible), so it is part of m's removal energy and of
every candidate energy of m.

Incremental update (the eKMC requirement: every rate changes after every event, but only through the moved molecule):
the structure factors S_m are stored ("committed"). When molecule k moves, `recip_commit!` computes S_k at the new pose
(O(n_sites · N_k)), ΔS_k = S_k^new − S_k^old, Stot += ΔS_k, and e_k. The new row ψ_kj for all j is one complex
matrix–vector product Re[Sᴴ (c ∘ S_k)] (O(N · N_k)); every stored candidate energy changes by Σ_k c_k Re[S_cand^* ΔS_k]
(one product over the stored candidate structure factors). Nothing else is recomputed. Molecule structure factors are
built with the e^{i 2π n x/L} power tables (one sincos per site and direction).
"""

using SpecialFunctions: erf, erfc
using LinearAlgebra: mul!, dot

mutable struct EwaldRecip
    L::Float64
    α::Float64
    ke::Float64
    n2max::Int
    nmax::Int
    nx::Vector{Int}
    ny::Vector{Int}
    nz::Vector{Int}
    c::Vector{Float64}            # c_k = ke (8π/V) exp(−k²/4α²)/k²
    S::Matrix{ComplexF64}         # N_k × N committed molecular structure factors
    Stot::Vector{ComplexF64}
    eimg::Vector{Float64}         # committed image energies e_m
    corr::Vector{Float64}         # committed ke α/√π Σq² + ke Σ_{a<b} q_a q_b erf(α r_ab)/r_ab
    work::Vector{ComplexF64}      # S of a molecule at its current (possibly uncommitted) pose
    cw::Vector{ComplexF64}        # c ∘ (a structure factor)
    dS::Vector{ComplexF64}        # ΔS of the last commit
    last::Int                     # molecule of the last commit (0 after a rebuild)
    tmpN::Vector{ComplexF64}
    ex::Vector{ComplexF64}        # power tables e^{i 2π n x/L}, n = −nmax…nmax (one site)
    ey::Vector{ComplexF64}
    ez::Vector{ComplexF64}
    n_commit::Int
    backend::Symbol               # :cpu or :cuda
    Sd::AbstractMatrix{ComplexF64}   # S on the backend (=== S on the CPU)
    cd::AbstractVector{Float64}      # c on the backend
    vd::AbstractVector{ComplexF64}   # an N_k vector on the backend (S_k or ΔS_k)
    cwd::AbstractVector{ComplexF64}  # c ∘ vd on the backend
    tmpNd::AbstractVector{ComplexF64}
end

"Device array for `backend` (identity on the CPU; `:cuda` is provided by MolSimCUDAExt)."
backend_array(::Val{:cpu}, A::AbstractArray) = A
backend_array(::Val{B}, A::AbstractArray) where {B} =
    error("reciprocal-space backend :$B is not available (for :cuda load CUDA.jl first: `using CUDA`)")
backend_array(b::Symbol, A::AbstractArray) = backend_array(Val(b), A)
"Copy a backend vector into a host vector (no-op when they are the same array)."
to_host!(dst::Array, src::AbstractArray) = (dst === src || copyto!(dst, src); dst)

"""
    ewald_alpha_for(rc, rtol) -> α

Ewald splitting parameter with erfc(α r_c) = rtol (the GROMACS `ewald-rtol` definition, default 1e-5; solved by
bisection as GROMACS `calc_ewaldcoeff_q` does).
"""
function ewald_alpha_for(rc::Float64, rtol::Float64)::Float64
    lo, hi = 0.0, 5.0 / rc
    while erfc(hi * rc) > rtol; hi *= 2; end
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        erfc(mid * rc) > rtol ? (lo = mid) : (hi = mid)
    end
    return 0.5 * (lo + hi)
end

"""
    ewald_n2max_for(L, α, rtol) -> n2max

Spherical reciprocal cut-off balanced with the real-space one (Kolafa & Perram, Mol. Simul. 9 (1992) 351; Frenkel & Smit
§12.1.5): the Gaussian factor exp(−k²/4α²) = exp(−π²|n|²/(α²L²)) is below `rtol` for every omitted |n|, i.e.
k_max = ⌈α L √(−ln rtol)/π⌉ and all |n|² ≤ k_max² are kept (n2max = k_max² + 1, exclusive bound).
"""
function ewald_n2max_for(L::Float64, α::Float64, rtol::Float64)::Int
    kmax = ceil(Int, α * L * sqrt(-log(rtol)) / π)
    return kmax * kmax + 1
end

function _recip_weights(L::Float64, α::Float64, ke::Float64, nx, ny, nz)
    V = L^3
    tp = 2π / L
    c = similar(nx, Float64)
    @inbounds for t in eachindex(nx)
        k2 = tp^2 * (nx[t]^2 + ny[t]^2 + nz[t]^2)
        c[t] = ke * (8π / V) * exp(-k2 / (4α^2)) / k2
    end
    return c
end

"""
    EwaldRecip(sys; α, ke, n2max)

Reciprocal-space state for `sys` (attach with `attach_ewald!`). Wave vectors as `ewald_kvectors`.
"""
function EwaldRecip(sys; α::Float64, ke::Float64, n2max::Int)
    ks = ewald_kvectors(EwaldParams(sys.L, α, 0.0, n2max, ke))
    nx = [k[1] for k in ks]; ny = [k[2] for k in ks]; nz = [k[3] for k in ks]
    nmax = floor(Int, sqrt(n2max - 1))
    Nk = length(ks); N = sys.n_mol
    c = _recip_weights(sys.L, α, ke, nx, ny, nz)
    R = EwaldRecip(sys.L, α, ke, n2max, nmax, nx, ny, nz, c, zeros(ComplexF64, Nk, N), zeros(ComplexF64, Nk),
                   zeros(N), zeros(N), zeros(ComplexF64, Nk), zeros(ComplexF64, Nk), zeros(ComplexF64, Nk), 0,
                   zeros(ComplexF64, N), zeros(ComplexF64, 2nmax + 1), zeros(ComplexF64, 2nmax + 1),
                   zeros(ComplexF64, 2nmax + 1), 0, :cpu, zeros(ComplexF64, 0, 0), Float64[], ComplexF64[], ComplexF64[],
                   ComplexF64[])
    _bind_backend!(R, :cpu)
    return R
end

"(Re)create the backend arrays of R from the host state (`backend` = :cpu or :cuda)."
function _bind_backend!(R::EwaldRecip, backend::Symbol)
    R.backend = backend
    if backend === :cpu
        R.Sd = R.S; R.cd = R.c; R.vd = R.work; R.cwd = R.cw; R.tmpNd = R.tmpN
    else
        R.Sd = backend_array(backend, R.S); R.cd = backend_array(backend, R.c)
        R.vd = backend_array(backend, R.work); R.cwd = backend_array(backend, R.cw)
        R.tmpNd = backend_array(backend, R.tmpN)
    end
    return R
end

"Move the reciprocal-space products of R to `backend` (:cpu or :cuda)."
set_recip_backend!(R::EwaldRecip, backend::Symbol) = _bind_backend!(R, backend)

n_kvectors(R::EwaldRecip) = length(R.c)

"Reciprocal-space Ewald active: `sys` has EwaldRecip state and the force field uses `coulomb = :ewald` (a force field
without charges, e.g. an LJ-only copy, sees the real-space terms only)."
@inline _has_recip(sys, ff) = sys.recip !== nothing && ff.coulomb === :ewald

# powers e^{i θ n}, n = −nmax…nmax, stored at index n + nmax + 1
@inline function _powers!(e::Vector{ComplexF64}, θ::Float64, nmax::Int)
    s, c = sincos(θ)
    z = complex(c, s)
    o = nmax + 1
    @inbounds begin
        e[o] = 1.0 + 0.0im
        for n in 1:nmax
            e[o + n] = e[o + n - 1] * z
            e[o - n] = conj(e[o + n])
        end
    end
    return nothing
end

"""
    recip_sf!(out, R, sys, i)

Structure factor of molecule i at its current pose (sites COM + site_off) into `out` (length N_k).
"""
function recip_sf!(out::AbstractVector{ComplexF64}, R::EwaldRecip, sys, i::Int)
    fill!(out, 0.0 + 0.0im)
    t = sys.templates[sys.tmpl[i]]
    s0 = sys.site_start[i]
    tp = 2π / R.L
    o = R.nmax + 1
    @inbounds for a in 1:t.n_sites
        q = t.site_q[a]
        q == 0.0 && continue
        x = sys.com[1, i] + sys.site_off[1, s0 + a]
        y = sys.com[2, i] + sys.site_off[2, s0 + a]
        z = sys.com[3, i] + sys.site_off[3, s0 + a]
        _powers!(R.ex, tp * x, R.nmax); _powers!(R.ey, tp * y, R.nmax); _powers!(R.ez, tp * z, R.nmax)
        for kk in eachindex(out)
            out[kk] += q * (R.ex[o + R.nx[kk]] * R.ey[o + R.ny[kk]] * R.ez[o + R.nz[kk]])
        end
    end
    return out
end

"ke α/√π Σ_a q_a² + ke Σ_{a<b} q_a q_b erf(α r_ab)/r_ab of molecule i at its current pose."
function recip_corr(R::EwaldRecip, sys, i::Int)::Float64
    t = sys.templates[sys.tmpl[i]]
    s0 = sys.site_start[i]
    E = 0.0
    @inbounds for a in 1:t.n_sites
        qa = t.site_q[a]
        qa == 0.0 && continue
        E += R.ke * R.α / SQRT_PI * qa * qa
        for b in a+1:t.n_sites
            qb = t.site_q[b]
            qb == 0.0 && continue
            dx = sys.site_off[1, s0 + b] - sys.site_off[1, s0 + a]
            dy = sys.site_off[2, s0 + b] - sys.site_off[2, s0 + a]
            dz = sys.site_off[3, s0 + b] - sys.site_off[3, s0 + a]
            r = sqrt(dx * dx + dy * dy + dz * dz)
            E += R.ke * qa * qb * erf(R.α * r) / r
        end
    end
    return E
end

"Σ_k c_k Re[a_k^* b_k]."
@inline function recip_cross(R::EwaldRecip, a::AbstractVector{ComplexF64}, b::AbstractVector{ComplexF64})::Float64
    s = 0.0
    @inbounds @simd for kk in eachindex(R.c)
        s += R.c[kk] * (real(a[kk]) * real(b[kk]) + imag(a[kk]) * imag(b[kk]))
    end
    return s
end

"½ Σ_k c_k |a_k|²."
@inline function recip_half_norm(R::EwaldRecip, a::AbstractVector{ComplexF64})::Float64
    s = 0.0
    @inbounds @simd for kk in eachindex(R.c)
        s += R.c[kk] * abs2(a[kk])
    end
    return 0.5 * s
end

"""
    recip_rebuild!(R, sys)

Recompute every committed structure factor, Stot and the image energies (after a volume change the weights c_k follow
the new L; the integer wave-vector set is kept).
"""
function recip_rebuild!(R::EwaldRecip, sys)
    if sys.L != R.L
        R.L = sys.L
        R.c = _recip_weights(R.L, R.α, R.ke, R.nx, R.ny, R.nz)
        R.backend === :cpu ? (R.cd = R.c) : copyto!(R.cd, R.c)
    end
    N = sys.n_mol
    size(R.S, 2) == N || error("recip_rebuild!: EwaldRecip built for $(size(R.S, 2)) molecules, system has $N")
    fill!(R.Stot, 0.0 + 0.0im)
    for i in 1:N
        col = view(R.S, :, i)
        recip_sf!(col, R, sys, i)
        R.Stot .+= col
        R.corr[i] = recip_corr(R, sys, i)
        R.eimg[i] = recip_half_norm(R, col) - R.corr[i]
    end
    R.Sd === R.S || copyto!(R.Sd, R.S)
    R.last = 0
    return nothing
end

"""
    recip_commit!(sys, i)

Molecule i has been moved: store its structure factor at the new pose, ΔS (kept in R.dS for the candidate updates),
Stot and its image energy. No-op without reciprocal space.
"""
function recip_commit!(sys, i::Int)
    R = sys.recip
    R === nothing && return nothing
    recip_sf!(R.work, R, sys, i)
    col = view(R.S, :, i)
    @inbounds for kk in eachindex(R.work)
        d = R.work[kk] - col[kk]
        R.dS[kk] = d
        R.Stot[kk] += d
        col[kk] = R.work[kk]
    end
    R.corr[i] = recip_corr(R, sys, i)
    R.eimg[i] = recip_half_norm(R, col) - R.corr[i]
    R.Sd === R.S || copyto!(view(R.Sd, :, i), R.work)   # R.work == the new column (a host Array)
    R.last = i
    R.n_commit += 1
    return nothing
end

"Image energy e_i of molecule i at its current (possibly uncommitted) pose; leaves S_i(now) in R.work."
function recip_image_now(R::EwaldRecip, sys, i::Int)::Float64
    recip_sf!(R.work, R, sys, i)
    return recip_half_norm(R, R.work) - recip_corr(R, sys, i)
end

"""
    recip_row_add!(row, R, sys, k)

row[j] += ψ_kj = Σ_k c_k Re[S_k(now)^* S_j] for every j (one matrix–vector product; S_k at its current pose, the other
molecules committed). row[k] is meaningless (callers reset it).
"""
function recip_row_add!(row::AbstractVector{Float64}, R::EwaldRecip, sys, k::Int)
    recip_sf!(R.work, R, sys, k)
    R.vd === R.work || copyto!(R.vd, R.work)
    R.cwd .= R.cd .* R.vd
    mul!(R.tmpNd, R.Sd', R.cwd)                     # Σ_k conj(S_jk) c_k S_k
    to_host!(R.tmpN, R.tmpNd)
    @inbounds for j in eachindex(row); row[j] += real(R.tmpN[j]); end
    return nothing
end

"""
    recip_removal(R, sys, i) -> Σ_{j≠i} ψ_ij + e_i

Reciprocal part of the removal energy of molecule i at its current pose, the rest committed.
"""
function recip_removal(R::EwaldRecip, sys, i::Int)::Float64
    e = recip_image_now(R, sys, i)                  # R.work = S_i(now)
    s = 0.0
    @inbounds @simd for kk in eachindex(R.c)
        r = R.Stot[kk] - R.S[kk, i]
        s += R.c[kk] * (real(R.work[kk]) * real(r) + imag(R.work[kk]) * imag(r))
    end
    return s + e
end

"Committed total reciprocal-space contribution Σ_{m<n} ψ_mn + Σ_m e_m = ½ Σ c |Stot|² − Σ_m corr_m."
recip_energy_total(R::EwaldRecip) = recip_half_norm(R, R.Stot) - sum(R.corr)

"""
    attach_ewald!(sys, ff; rtol=1e-5, n2max=0)

Give `sys` reciprocal-space Ewald state for a force field with `coulomb = :ewald` (α = ff.wolf.α, ke = ff.wolf.ke).
`n2max = 0` chooses the reciprocal cut-off from `rtol` (`ewald_n2max_for`). Returns the EwaldRecip.
"""
function attach_ewald!(sys, ff; rtol::Float64=1e-5, n2max::Int=0)
    ff.coulomb === :ewald || error("attach_ewald!: force field coulomb = $(ff.coulomb), need :ewald")
    α = ff.wolf.α
    n2 = n2max > 0 ? n2max : ewald_n2max_for(sys.L, α, rtol)
    R = EwaldRecip(sys; α=α, ke=ff.wolf.ke, n2max=n2)
    sys.recip = R
    recip_rebuild!(R, sys)
    return R
end

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

Backends (DESIGN §13.3 E2): the products run on `R.Sd`, `R.cd`, … which are the host arrays themselves (`:cpu`, `:double`)
or copies (`:cuda`, in the package extension MolSimCUDAExt, loaded with `using CUDA`; the CPU-only build never needs CUDA).
Precision (user, 2026-10-17: "keep the precision that gromacs does"): `:mixed` (default) is the GROMACS mixed-precision
convention — state-like arrays in single precision, critical accumulations in double (GROMACS reference manual,
"Definitions": coordinates, velocities and forces single, "critical variables" such as the virial double). Here the
backend copies of the structure factors, weights and candidate structure factors and the products are ComplexF32/Float32;
the host keeps the committed S, Stot, image energies in Float64 and every result (pair energies, φ, candidate energies,
rates, totals) is accumulated in Float64. The single-precision increments drift by O(u_Float32) per event; the eKMC loop
removes it with an exact Float64 rebuild every `RECIP_RESYNC_EVENTS` events. `:double` keeps everything in Float64.
The per-molecule work (structure factor of the moved molecule, image energy) stays on the host; per event one column
(N_k complex) goes up and one row (N real) or one candidate vector (K·M real) comes back; the products are real
matrix–vector products over the interleaved (Re, Im) storage (`_realview`), since only Re[Sᴴ v] is needed.
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
    gstart::Vector{Int}           # k-vector groups of equal (nx, ny): k = gstart[g] … gstart[g+1]−1 have nz = gnz0[g], gnz0[g]+1, …
    gnx::Vector{Int}
    gny::Vector{Int}
    gnz0::Vector{Int}
    rowh::Vector{Float64}         # N: host copy of a product row
    ex::Vector{ComplexF64}        # power tables e^{i 2π n x/L}, n = −nmax…nmax (one site)
    ey::Vector{ComplexF64}
    ez::Vector{ComplexF64}
    n_commit::Int
    backend::Symbol               # :cpu or :cuda
    precision::Symbol             # :mixed (GROMACS convention, default) or :double
    Sd::AbstractMatrix            # S on the backend (=== S for :cpu/:double; ComplexF32 copy for :mixed)
    cd::AbstractVector            # c on the backend
    vd::AbstractVector            # an N_k vector on the backend (S_k or ΔS_k)
    cwd::AbstractVector           # c ∘ vd on the backend
    Sdr::AbstractMatrix           # Sd as a real 2N_k × N matrix (interleaved Re, Im; same memory)
    cwdr::AbstractVector          # cwd as a real 2N_k vector (same memory)
    rowd::AbstractVector          # N real: Re[Sdᴴ cwd] = Sdrᵀ cwdr
    stage::Vector                 # N_k host staging vector in the backend element type (allocation-free uploads)
    stage_row::Vector             # N host staging vector in the backend real type (allocation-free downloads)
end

"Events between exact Float64 rebuilds of the eKMC bookkeeping with `:mixed` precision (a frequency choice: the
rebuild costs ~N²N_k, i.e. ≲ 2 % of 10⁴ events for N = 500 SPC/E; it removes the O(u_Float32)-per-event drift)."
const RECIP_RESYNC_EVENTS = 10_000

"Backend array for `backend` (the array itself on the CPU; `:cuda` is provided by MolSimCUDAExt)."
backend_array(::Val{:cpu}, A::AbstractArray) = A
backend_array(::Val{B}, A::AbstractArray) where {B} =
    error("reciprocal-space backend :$B is not available (for :cuda load CUDA.jl first: `using CUDA`)")
backend_array(b::Symbol, A::AbstractArray) = backend_array(Val(b), A)
"Copy a backend array into a host array, converting the element type if needed (no-op when they are the same array)."
function to_host!(dst::Array, src::AbstractArray)
    dst === src && return dst
    eltype(dst) == eltype(src) ? copyto!(dst, src) : (dst .= Array(src))
    return dst
end
"Copy a host array into a backend array (or a contiguous view of one), converting the element type if needed."
_upload!(dst::AbstractArray, src::AbstractArray) =
    eltype(dst) == eltype(src) ? copyto!(dst, src) : copyto!(dst, convert(Array{eltype(dst)}, src))
"As `_upload!` without allocating: host destinations convert in place, others go through the host buffer `stage`
(eltype(dst), same length as src)."
function _upload!(dst::AbstractArray, src::AbstractArray, stage::Array)
    if dst isa Array || (dst isa SubArray && parent(dst) isa Array)
        dst .= src
    else
        # a device destination takes one block copy from a dense host array: a host view (e.g. a column range) or
        # another element type is gathered into `stage` first (copying a view straight into a device array falls back
        # to element-wise setindex!, which CUDA forbids)
        s = src isa Array && eltype(src) == eltype(dst) ? src : copyto!(stage, src)
        _device_copy!(dst, s, length(src))
    end
    return dst
end
# n elements of the dense host array s into a device array or a contiguous view of one (offset copy into the parent)
_device_copy!(dst::AbstractArray, s::Array, n::Int) = copyto!(dst, 1, s, 1, n)
function _device_copy!(dst::SubArray, s::Array, n::Int)
    n == 0 && return dst
    p = parent(dst); li = LinearIndices(p)
    off = li[first.(parentindices(dst))...]
    li[last.(parentindices(dst))...] - off + 1 == n || error("_upload!: device view is not contiguous")
    copyto!(p, off, s, 1, n)
    return dst
end
"As `to_host!` without allocating on the host side (`stage`: host array of eltype(src), same length)."
function to_host!(dst::Array, src::AbstractArray, stage::Array)
    dst === src && return dst
    if src isa Array || eltype(dst) == eltype(src)
        src isa Array ? (dst .= src) : copyto!(dst, src)
    else
        copyto!(stage, src); dst .= stage
    end
    return dst
end

"""
    _realview(A)

A complex array seen as a real one over the same memory: a vector of length n → 2n, a matrix n × m → 2n × m, with
(Re, Im) interleaved. Re[aᴴ b] = Σ (Re a Re b + Im a Im b) = (real a)ᵀ (real b), so Re[Aᴴ v] is one real matrix–vector
product (half the arithmetic of the complex one, which also computes the unused imaginary part).
"""
_realview(A::AbstractVector{<:Complex}) = reinterpret(real(eltype(A)), A)
_realview(A::AbstractMatrix{<:Complex}) = reshape(reinterpret(real(eltype(A)), A), 2 * size(A, 1), size(A, 2))

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
    gstart, gnx, gny, gnz0 = _kvector_groups(nx, ny, nz)
    R = EwaldRecip(sys.L, α, ke, n2max, nmax, nx, ny, nz, c, zeros(ComplexF64, Nk, N), zeros(ComplexF64, Nk),
                   zeros(N), zeros(N), zeros(ComplexF64, Nk), zeros(ComplexF64, Nk), zeros(ComplexF64, Nk), 0,
                   gstart, gnx, gny, gnz0, zeros(N), zeros(ComplexF64, 2nmax + 1), zeros(ComplexF64, 2nmax + 1),
                   zeros(ComplexF64, 2nmax + 1), 0, :cpu, :double, zeros(ComplexF64, 0, 0), Float64[], ComplexF64[],
                   ComplexF64[], zeros(0, 0), Float64[], Float64[], ComplexF64[], Float64[])
    _bind_backend!(R, :cpu; precision=:double)
    return R
end

"""
    _kvector_groups(nx, ny, nz) -> (gstart, gnx, gny, gnz0)

Runs of consecutive wave vectors with equal (nx, ny) and nz increasing by one (`ewald_kvectors` loops nz innermost, and
the kept nz of a given (nx, ny) form one interval: a sphere intersected with the half space). Lets the structure factor
form e^{i2π(nx x + ny y)/L} once per run.
"""
function _kvector_groups(nx::Vector{Int}, ny::Vector{Int}, nz::Vector{Int})
    gstart = Int[]; gnx = Int[]; gny = Int[]; gnz0 = Int[]
    for t in eachindex(nx)
        if t == 1 || nx[t] != nx[t-1] || ny[t] != ny[t-1] || nz[t] != nz[t-1] + 1
            push!(gstart, t); push!(gnx, nx[t]); push!(gny, ny[t]); push!(gnz0, nz[t])
        end
    end
    push!(gstart, length(nx) + 1)
    return gstart, gnx, gny, gnz0
end

"(Re)create the backend arrays of R from the host state (`backend` = :cpu or :cuda, `precision` = :mixed or :double)."
function _bind_backend!(R::EwaldRecip, backend::Symbol; precision::Symbol=R.precision)
    precision in (:mixed, :double) || error("EwaldRecip: precision must be :mixed or :double")
    R.backend = backend; R.precision = precision
    if backend === :cpu && precision === :double
        R.Sd = R.S; R.cd = R.c; R.vd = R.work; R.cwd = R.cw; R.rowd = R.rowh
        R.stage = ComplexF64[]; R.stage_row = Float64[]
    else
        C = precision === :mixed ? ComplexF32 : ComplexF64
        F = precision === :mixed ? Float32 : Float64
        R.Sd = backend_array(backend, convert(Matrix{C}, R.S)); R.cd = backend_array(backend, convert(Vector{F}, R.c))
        R.vd = backend_array(backend, zeros(C, length(R.c))); R.cwd = backend_array(backend, zeros(C, length(R.c)))
        R.rowd = backend_array(backend, zeros(F, length(R.rowh)))
        R.stage = zeros(C, length(R.c)); R.stage_row = zeros(F, length(R.rowh))
    end
    R.Sdr = _realview(R.Sd); R.cwdr = _realview(R.cwd)
    return R
end

"Set the backend (:cpu or :cuda) and precision (:mixed, the GROMACS convention, or :double) of R's products."
set_recip_backend!(R::EwaldRecip, backend::Symbol; precision::Symbol=R.precision) = _bind_backend!(R, backend; precision=precision)

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
        for g in eachindex(R.gnx)                    # one run of equal (nx, ny): e^{i(kx x + ky y)} formed once
            qxy = q * (R.ex[o + R.gnx[g]] * R.ey[o + R.gny[g]])
            z0 = o + R.gnz0[g] - R.gstart[g]
            @simd for kk in R.gstart[g]:R.gstart[g+1]-1
                out[kk] += qxy * R.ez[z0 + kk]
            end
        end
    end
    return out
end

"""
    recip_sf_translate!(out, R, src, d)

out = e^{i k·d} ∘ src: the structure factor of a molecule with structure factor `src` translated rigidly by d
(S(k) = Σ_a q_a e^{i k·(r_a + d)} = e^{i k·d} Σ_a q_a e^{i k·r_a}); cost independent of the number of sites.
"""
function recip_sf_translate!(out::AbstractVector{ComplexF64}, R::EwaldRecip, src::AbstractVector{ComplexF64},
                             d::NTuple{3,Float64})
    tp = 2π / R.L
    o = R.nmax + 1
    _powers!(R.ex, tp * d[1], R.nmax); _powers!(R.ey, tp * d[2], R.nmax); _powers!(R.ez, tp * d[3], R.nmax)
    @inbounds for g in eachindex(R.gnx)
        pxy = R.ex[o + R.gnx[g]] * R.ey[o + R.gny[g]]
        z0 = o + R.gnz0[g] - R.gstart[g]
        @simd for kk in R.gstart[g]:R.gstart[g+1]-1
            out[kk] = (pxy * R.ez[z0 + kk]) * src[kk]
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
        R.Sd === R.S ? (R.cd = R.c) : _upload!(R.cd, R.c)      # aliased on :cpu/:double, a copy otherwise
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
    R.Sd === R.S || _upload!(R.Sd, R.S)
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
    R.Sd === R.S || _upload!(view(R.Sd, :, i), R.work, R.stage)   # R.work == the new column (a host Array)
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
    R.vd === R.work || _upload!(R.vd, R.work, R.stage)
    R.cwd .= R.cd .* R.vd
    _recip_row_product!(row, R)
end

"Row from the backend vector R.cwd: row[j] += Re[Σ_k conj(S_jk) cwd_k], one real matrix–vector product (`_realview`)."
function _recip_row_product!(row::AbstractVector{Float64}, R::EwaldRecip)
    mul!(R.rowd, transpose(R.Sdr), R.cwdr)
    to_host!(R.rowh, R.rowd, R.stage_row)
    @inbounds for j in eachindex(row); row[j] += R.rowh[j]; end
    return nothing
end

"""
    recip_row_add_committed!(row, R, k)

As `recip_row_add!` for molecule k at its committed pose (right after `recip_commit!`): the committed backend column
S_k is used directly (no structure factor, no upload).
"""
function recip_row_add_committed!(row::AbstractVector{Float64}, R::EwaldRecip, k::Int)
    R.cwd .= R.cd .* view(R.Sd, :, k)
    _recip_row_product!(row, R)
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
`n2max = 0` chooses the reciprocal cut-off from `rtol` (`ewald_n2max_for`). `precision = :mixed` (default, the GROMACS
convention) or `:double`; `backend = :cpu` or `:cuda`. Returns the EwaldRecip.
"""
function attach_ewald!(sys, ff; rtol::Float64=1e-5, n2max::Int=0, backend::Symbol=:cpu, precision::Symbol=:mixed)
    ff.coulomb === :ewald || error("attach_ewald!: force field coulomb = $(ff.coulomb), need :ewald")
    α = ff.wolf.α
    n2 = n2max > 0 ? n2max : ewald_n2max_for(sys.L, α, rtol)
    R = EwaldRecip(sys; α=α, ke=ff.wolf.ke, n2max=n2)
    sys.recip = R
    recip_rebuild!(R, sys)
    set_recip_backend!(R, backend; precision=precision)
    return R
end

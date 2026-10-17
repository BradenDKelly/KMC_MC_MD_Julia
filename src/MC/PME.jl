"""
Smooth particle-mesh Ewald (SPME; Essmann, Perera, Berkowitz, Darden, Lee, Pedersen, J. Chem. Phys. 103 (1995) 8577),
reciprocal-space energy for a cubic box (DESIGN §13.3 E4). Charges are spread on a K³ mesh with cardinal B-splines of
order p (Essmann Eq. 4.1, recursion 4.2); the structure factor is S(m) ≈ b(m) F[Q](m) with the Euler exponential spline
factors |b_i(m_i)|² = 1/|Σ_{k=0}^{p−2} M_p(k+1) e^{2πi m_i k/K}|² (Essmann Eq. 4.4), so

    E_rec ≈ ke (2π/V) Σ_{m ≠ 0} e^{−k²/4α²}/k² B(m) |F[Q](m)|²,   k = 2π m/L,   B = Π_i |b_i|²

— the Ewald Fourier sum over every charge pair (self and intramolecular pairs included, as `ewald_recip_energy`); the
self and intramolecular erf corrections are the analytic Ewald ones. Grid size: GROMACS `fourier-spacing` (default
0.12 nm) rounded up to a product of small primes, order 4 (`pme-order`), α from `ewald-rtol` — the GROMACS defaults.
"""

using FFTW

"Smallest n ≥ x whose prime factors are 2, 3, 5, 7 (FFT-friendly grid size, as GROMACS chooses PME grids)."
function pme_grid_size(x::Real)::Int
    n = max(ceil(Int, x - 1e-9), 1)
    while true
        m = n
        for f in (2, 3, 5, 7)
            while m % f == 0; m ÷= f; end
        end
        m == 1 && return n
        n += 1
    end
end

"""
    _bspline!(θ, t, p)

θ[j+1] = M_p(t + j), j = 0…p−1, for t ∈ [0, 1) (cardinal B-spline of order p, Essmann Eq. 4.1–4.2).
"""
@inline function _bspline!(θ::AbstractVector{Float64}, t::Float64, p::Int)
    @inbounds begin
        θ[1] = t; θ[2] = 1.0 - t                         # M_2(t), M_2(t+1)
        for j in 3:p; θ[j] = 0.0; end
        for n in 3:p                                     # order n from order n−1
            for j in n:-1:1
                x = t + (j - 1)
                a = x * θ[j]
                b = j > 1 ? (n - x) * θ[j-1] : 0.0
                θ[j] = (a + b) / (n - 1)
            end
        end
    end
    return θ
end

"M_p(x) for integer x = 1 … p−1 (the values in the b(m) denominator)."
function _bspline_integer_values(p::Int)
    θ = zeros(p)
    _bspline!(θ, 0.0, p)                                 # θ[j+1] = M_p(j)
    return [θ[k+1] for k in 1:p-1]
end

"|b(m)|², m = 0…K−1 (Essmann Eq. 4.4)."
function pme_bmod2(K::Int, p::Int)
    Mi = _bspline_integer_values(p)                      # M_p(1) … M_p(p−1)
    b2 = zeros(K)
    for m in 0:K-1
        s = 0.0 + 0.0im
        for k in 0:p-2
            s += Mi[k+1] * cis(2π * m * k / K)
        end
        b2[m+1] = 1.0 / abs2(s)
    end
    return b2
end

struct PMEParams
    L::Float64
    α::Float64
    ke::Float64
    K::Int
    p::Int
end

"""
    pme_influence(pp) -> G (K × K × K)

G(m) = ke (2π/V) e^{−k²/4α²}/k² B(m), G(0) = 0, so that E_rec = Σ_m G(m) |F[Q](m)|².
"""
function pme_influence(pp::PMEParams)
    K = pp.K; L = pp.L; V = L^3
    b2 = pme_bmod2(K, pp.p)
    G = zeros(K, K, K)
    tp = 2π / L
    @inbounds for i3 in 0:K-1, i2 in 0:K-1, i1 in 0:K-1
        (i1 | i2 | i3) == 0 && continue
        m1 = i1 <= K ÷ 2 ? i1 : i1 - K; m2 = i2 <= K ÷ 2 ? i2 : i2 - K; m3 = i3 <= K ÷ 2 ? i3 : i3 - K
        k2 = tp^2 * (m1^2 + m2^2 + m3^2)
        G[i1+1, i2+1, i3+1] = pp.ke * (2π / V) * exp(-k2 / (4pp.α^2)) / k2 * b2[i1+1] * b2[i2+1] * b2[i3+1]
    end
    return G
end

"""
    pme_spread!(Q, pos, q, pp)

Q ← Σ_a q_a Π_d M_p(u_{a,d} − k_d) on the K³ mesh (u = K x/L wrapped into [0, K)).
"""
function pme_spread!(Q::Array{Float64,3}, pos::AbstractMatrix{Float64}, q::AbstractVector{Float64}, pp::PMEParams)
    fill!(Q, 0.0)
    K = pp.K; p = pp.p
    θx = zeros(p); θy = zeros(p); θz = zeros(p)
    @inbounds for a in eachindex(q)
        qa = q[a]
        qa == 0.0 && continue
        ux = K * pos[1, a] / pp.L; uy = K * pos[2, a] / pp.L; uz = K * pos[3, a] / pp.L
        ux -= K * floor(ux / K); uy -= K * floor(uy / K); uz -= K * floor(uz / K)
        ix = floor(Int, ux); iy = floor(Int, uy); iz = floor(Int, uz)
        _bspline!(θx, ux - ix, p); _bspline!(θy, uy - iy, p); _bspline!(θz, uz - iz, p)
        for jz in 0:p-1
            gz = mod(iz - jz, K) + 1
            for jy in 0:p-1
                gy = mod(iy - jy, K) + 1
                w = qa * θy[jy+1] * θz[jz+1]
                for jx in 0:p-1
                    Q[mod(ix - jx, K) + 1, gy, gz] += w * θx[jx+1]
                end
            end
        end
    end
    return Q
end

"""
    pme_recip_energy(pos, q, pp; G=pme_influence(pp)) -> E_rec

SPME reciprocal-space energy of all charges (compare `ewald_recip_energy`).
"""
function pme_recip_energy(pos::AbstractMatrix{Float64}, q::AbstractVector{Float64}, pp::PMEParams;
                          G::Array{Float64,3}=pme_influence(pp))
    Q = zeros(pp.K, pp.K, pp.K)
    pme_spread!(Q, pos, q, pp)
    F = fft(Q)
    E = 0.0
    @inbounds for i in eachindex(F)
        E += G[i] * abs2(F[i])
    end
    return E
end

"""
    pme_energy(pos, q, mol, ep::EwaldParams, K, p) -> NamedTuple (real, fourier, self, intra, total)

As `ewald_energy` with the Fourier term from SPME on a K³ mesh of order p.
"""
function pme_energy(pos::AbstractMatrix{Float64}, q::AbstractVector{Float64}, mol::AbstractVector{Int},
                    ep::EwaldParams, K::Int, p::Int)
    pp = PMEParams(ep.L, ep.α, ep.ke, K, p)
    Er = ewald_real_energy(pos, q, mol, ep)
    Ef = pme_recip_energy(pos, q, pp)
    Es = ewald_self_energy(q, ep)
    Ei = ewald_intra_energy(pos, q, mol, ep)
    return (real=Er, fourier=Ef, self=Es, intra=Ei, total=Er + Ef + Es + Ei)
end

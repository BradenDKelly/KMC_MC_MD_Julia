"""
Electrostatics for site-based rigid molecules (additive module; nothing else depends on it).

All routines take flat site arrays:
  pos :: 3 × n matrix of site positions (any units of length; need not be wrapped)
  q   :: n-vector of site charges
  mol :: n-vector of molecule ids (sites with equal id are intramolecular and are excluded from
         all intermolecular pair sums)
and a Coulomb constant `ke` so that the bare pair energy is `ke * q_i * q_j / r`. With ke = 1 and
reduced charges q* = q / sqrt(4π ε0 σ ε) the energies come out in units of ε, which is how the
rest of MolSim works; with ke = e²/(4π ε0 k_B) in K·Å the energies are E/k_B in kelvin, which is
the convention of the NIST SPC/E reference calculations used by `test/test_electrostatics_nist.jl`.

Three Coulomb models:

1. Ewald summation, tin-foil boundary conditions, cubic box (reference method). The decomposition
   follows the NIST Standard Reference Simulation Website "SPC/E Water Reference Calculations":
     E_real    = ke Σ_{i<j, mol_i≠mol_j, r_ij<rc} q_i q_j erfc(α r_ij)/r_ij        (minimum image)
     E_fourier = ke (2π/V) Σ_{k≠0} exp(-k²/4α²)/k² |S(k)|²,  S(k) = Σ_j q_j exp(i k·r_j)
     E_self    = -ke (α/√π) Σ_j q_j²
     E_intra   = -ke Σ_{i<j, mol_i=mol_j} q_i q_j erf(α r_ij)/r_ij                   (minimum image)
   k = 2π n / L for integer n with 0 < |n|² < n2max (NIST: kmax = 5, |n|² < 27).
   The Fourier term contains the intramolecular pairs; E_intra removes them so the total is the
   intermolecular Coulomb energy of rigid molecules (Allen & Tildesley, Computer Simulation of
   Liquids, §5.5; Frenkel & Smit, Understanding Molecular Simulation, §12.1).

2. Wolf summation (D. Wolf, P. Keblinski, S.R. Phillpot, J. Eggebrecht, J. Chem. Phys. 110 (1999)
   8254), damped shifted potential with charge neutralisation, as used in Monte Carlo by
   R. Hens & T.J.H. Vlugt, J. Chem. Eng. Data 63 (2018) 1096, and in B. Kelly's 2020 Julia code:
     u_ij  = ke q_i q_j [erfc(α r)/r − erfc(α rc)/rc]           for r < rc, intermolecular pairs
     E_self = -ke [erfc(α rc)/(2 rc) + α/√π] Σ_j q_j²            (configuration independent)
     E_intra = -Σ_{intramolecular pairs} (ke q_i q_j / r − u_ij(r))  (exclusion correction, constant
               for rigid molecules; see `wolf_intra_energy`)

3. Damped shifted force (DSF) energy of C.J. Fennell & J.D. Gezelter, J. Chem. Phys. 124 (2006)
   234104, whose derivative is the shifted Wolf force of Hansen, Schrøder & Dyre, J. Phys. Chem. B
   116 (2012) 5738, Eq. (1):
     u_ij = ke q_i q_j [erfc(α r)/r − erfc(α rc)/rc
                        + (erfc(α rc)/rc² + (2α/√π) exp(−α² rc²)/rc) (r − rc)]   for r < rc
   with the same self term as Wolf.  **Experimental**: the energy form is transcribed from memory of
   Fennell & Gezelter Eq. (10) and the self term has not been checked against that paper. Hansen et
   al. (2012, introduction) note that a shifted-force potential "does not match the original potential
   for r < rc", and the scan in dev/wolf_parameter_scan.jl confirms that the DSF *energy* deviates
   from Ewald by up to +100 % at small α rc while the Wolf (DSP) energy stays within ~2 %. For MC and
   eKMC, which sample energies, use `:wolf`; `:dsf` is kept for force-based comparisons only.

The pair functions are plain functions of (r, q_i, q_j) so that they can be dropped into the same
pair-energy bookkeeping that the LJ eKMC uses (pair matrix, removal energies, mobilities).
"""

using SpecialFunctions: erfc, erf

const SQRT_PI = sqrt(π)

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------

"""
    EwaldParams(L, α, rc, n2max, ke)

Cubic-box Ewald parameters. `n2max` is the exclusive bound on the squared integer wave-vector index
(NIST reference: kmax = 5 with |n|² < 27 → n2max = 27).
"""
struct EwaldParams
    L::Float64
    α::Float64
    rc::Float64
    n2max::Int
    ke::Float64
end

"""
    WolfParams(α, rc, ke; model=:wolf)

Pairwise damped-Coulomb parameters. `model` is `:wolf` (damped shifted potential) or `:dsf`.
"""
struct WolfParams
    α::Float64
    rc::Float64
    ke::Float64
    model::Symbol
    function WolfParams(α::Float64, rc::Float64, ke::Float64; model::Symbol=:wolf)
        model in (:wolf, :dsf) || error("WolfParams: model must be :wolf or :dsf, got $model")
        new(α, rc, ke, model)
    end
end

# ----------------------------------------------------------------------------
# Pair functions
# ----------------------------------------------------------------------------

"""
    ewald_real_pair_u(r, qi, qj, α, ke)

Real-space Ewald pair energy `ke qi qj erfc(α r)/r` (no cutoff test; caller applies rc).
"""
@inline ewald_real_pair_u(r::Float64, qi::Float64, qj::Float64, α::Float64, ke::Float64) =
    ke * qi * qj * erfc(α * r) / r

"""
    wolf_pair_u(r, qi, qj, wp::WolfParams)

Damped shifted (Wolf) or damped shifted force (DSF) pair energy for r < rc; zero otherwise.
"""
@inline function wolf_pair_u(r::Float64, qi::Float64, qj::Float64, wp::WolfParams)::Float64
    rc = wp.rc
    if r >= rc
        return 0.0
    end
    α = wp.α
    erfc_rc = erfc(α * rc)
    u = erfc(α * r) / r - erfc_rc / rc
    if wp.model === :dsf
        u += (erfc_rc / (rc * rc) + (2.0 * α / SQRT_PI) * exp(-α * α * rc * rc) / rc) * (r - rc)
    end
    return wp.ke * qi * qj * u
end

"""
    wolf_self_energy(q, wp::WolfParams)

Configuration-independent Wolf self term `-ke [erfc(α rc)/(2 rc) + α/√π] Σ q²`.
"""
function wolf_self_energy(q::AbstractVector{Float64}, wp::WolfParams)::Float64
    s = 0.0
    @inbounds for qi in q
        s += qi * qi
    end
    return -wp.ke * (erfc(wp.α * wp.rc) / (2.0 * wp.rc) + wp.α / SQRT_PI) * s
end

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

@inline function _min_image_r2(dx::Float64, dy::Float64, dz::Float64, L::Float64)
    dx -= L * round(dx / L)
    dy -= L * round(dy / L)
    dz -= L * round(dz / L)
    return dx * dx + dy * dy + dz * dz
end

# ----------------------------------------------------------------------------
# Wolf / DSF total energies
# ----------------------------------------------------------------------------

"""
    wolf_energy(pos, q, mol, L, wp::WolfParams) -> (E_pair, E_self, E_intra)

Intermolecular pair sum (minimum image, site-based spherical cutoff), the self term and the
intramolecular exclusion correction. The total Wolf electrostatic energy is `E_pair + E_self + E_intra`;
only `E_pair` depends on the configuration for rigid molecules.
"""
function wolf_energy(pos::AbstractMatrix{Float64}, q::AbstractVector{Float64},
                     mol::AbstractVector{Int}, L::Float64, wp::WolfParams)
    n = length(q)
    rc2 = wp.rc * wp.rc
    E = 0.0
    @inbounds for i in 1:n-1
        qi = q[i]
        qi == 0.0 && continue
        xi, yi, zi = pos[1, i], pos[2, i], pos[3, i]
        mi = mol[i]
        for j in i+1:n
            (mol[j] == mi || q[j] == 0.0) && continue
            r2 = _min_image_r2(pos[1, j] - xi, pos[2, j] - yi, pos[3, j] - zi, L)
            r2 < rc2 || continue
            E += wolf_pair_u(sqrt(r2), qi, q[j], wp)
        end
    end
    return E, wolf_self_energy(q, wp), wolf_intra_energy(pos, q, mol, L, wp)
end

"""
    wolf_intra_energy(pos, q, mol, L, wp::WolfParams)

Intramolecular exclusion correction for Wolf-type sums with rigid molecules,
`-Σ_{i<j same molecule} ke q_i q_j [1/r − u_pair(r)/(ke q_i q_j)]`, i.e. minus the part of the bare
intramolecular Coulomb energy that the damped shifted pair function does not carry. Derivation: the
Wolf self term is obtained from charge neutrality over *all* other charges (Wolf et al. 1999,
Eqs. 5.9–5.13), so when the intramolecular pairs are excluded from the pair sum their smooth
remainder has to be removed explicitly, exactly as the Ewald intramolecular term does (Allen &
Tildesley §5.5). For α → Ewald α and rc → ∞ this reduces to `-ke Σ q_i q_j erf(α r)/r`, the NIST
E_intra. Constant for rigid molecules (minimum-image separations).
"""
function wolf_intra_energy(pos::AbstractMatrix{Float64}, q::AbstractVector{Float64},
                           mol::AbstractVector{Int}, L::Float64, wp::WolfParams)::Float64
    n = length(q)
    E = 0.0
    @inbounds for i in 1:n-1
        qi = q[i]
        qi == 0.0 && continue
        mi = mol[i]
        for j in i+1:n
            (mol[j] != mi || q[j] == 0.0) && continue
            r = sqrt(_min_image_r2(pos[1, j] - pos[1, i], pos[2, j] - pos[2, i],
                                   pos[3, j] - pos[3, i], L))
            E -= wp.ke * qi * q[j] / r - wolf_pair_u(r, qi, q[j], wp)
        end
    end
    return E
end

# ----------------------------------------------------------------------------
# Ewald
# ----------------------------------------------------------------------------

"""
    ewald_real_energy(pos, q, mol, ep::EwaldParams)

Real-space sum over intermolecular site pairs within rc (minimum image).
"""
function ewald_real_energy(pos::AbstractMatrix{Float64}, q::AbstractVector{Float64},
                           mol::AbstractVector{Int}, ep::EwaldParams)::Float64
    n = length(q)
    L = ep.L
    rc2 = ep.rc * ep.rc
    E = 0.0
    @inbounds for i in 1:n-1
        qi = q[i]
        qi == 0.0 && continue
        xi, yi, zi = pos[1, i], pos[2, i], pos[3, i]
        mi = mol[i]
        for j in i+1:n
            (mol[j] == mi || q[j] == 0.0) && continue
            r2 = _min_image_r2(pos[1, j] - xi, pos[2, j] - yi, pos[3, j] - zi, L)
            r2 < rc2 || continue
            E += ewald_real_pair_u(sqrt(r2), qi, q[j], ep.α, ep.ke)
        end
    end
    return E
end

"""
    ewald_kvectors(ep::EwaldParams) -> Vector{NTuple{3,Int}}

Integer index vectors n with 0 < |n|² < n2max, one representative per ±n pair (the |S(k)|² sum
is symmetric under k → −k, so each representative is counted twice in `ewald_recip_energy`).
"""
function ewald_kvectors(ep::EwaldParams)
    nmax = floor(Int, sqrt(ep.n2max - 1))
    ks = NTuple{3,Int}[]
    for nx in 0:nmax, ny in -nmax:nmax, nz in -nmax:nmax
        n2 = nx * nx + ny * ny + nz * nz
        (n2 == 0 || n2 >= ep.n2max) && continue
        # keep one of ±n: first nonzero component positive
        if nx > 0 || (nx == 0 && (ny > 0 || (ny == 0 && nz > 0)))
            push!(ks, (nx, ny, nz))
        end
    end
    return ks
end

"""
    ewald_recip_energy(pos, q, ep::EwaldParams)

Fourier-space energy `ke (2π/V) Σ_{k≠0} exp(-k²/4α²)/k² |S(k)|²` over all sites (intramolecular
pairs included; subtract `ewald_intra_energy`).
"""
function ewald_recip_energy(pos::AbstractMatrix{Float64}, q::AbstractVector{Float64},
                            ep::EwaldParams)::Float64
    n = length(q)
    L = ep.L
    V = L * L * L
    twopi_L = 2π / L
    inv4α2 = 1.0 / (4.0 * ep.α * ep.α)
    E = 0.0
    for (nx, ny, nz) in ewald_kvectors(ep)
        kx, ky, kz = twopi_L * nx, twopi_L * ny, twopi_L * nz
        k2 = kx * kx + ky * ky + kz * kz
        sre = 0.0
        sim = 0.0
        @inbounds for j in 1:n
            qj = q[j]
            qj == 0.0 && continue
            ph = kx * pos[1, j] + ky * pos[2, j] + kz * pos[3, j]
            s, c = sincos(ph)
            sre += qj * c
            sim += qj * s
        end
        # factor 2: representative covers both k and −k
        E += 2.0 * exp(-k2 * inv4α2) / k2 * (sre * sre + sim * sim)
    end
    return ep.ke * (2π / V) * E
end

"""
    ewald_self_energy(q, ep::EwaldParams)

`-ke (α/√π) Σ q²`.
"""
function ewald_self_energy(q::AbstractVector{Float64}, ep::EwaldParams)::Float64
    s = 0.0
    @inbounds for qi in q
        s += qi * qi
    end
    return -ep.ke * ep.α / SQRT_PI * s
end

"""
    ewald_intra_energy(pos, q, mol, ep::EwaldParams)

Intramolecular correction `-ke Σ_{i<j same molecule} q_i q_j erf(α r_ij)/r_ij` with minimum-image
site separations (a rigid molecule is always shorter than L/2); constant for rigid molecules.
"""
function ewald_intra_energy(pos::AbstractMatrix{Float64}, q::AbstractVector{Float64},
                            mol::AbstractVector{Int}, ep::EwaldParams)::Float64
    n = length(q)
    E = 0.0
    @inbounds for i in 1:n-1
        qi = q[i]
        qi == 0.0 && continue
        mi = mol[i]
        for j in i+1:n
            (mol[j] != mi || q[j] == 0.0) && continue
            # minimum image: reference configurations may store sites of one molecule wrapped
            # into different periodic images (NIST configurations 3 and 4 do)
            r = sqrt(_min_image_r2(pos[1, j] - pos[1, i], pos[2, j] - pos[2, i],
                                   pos[3, j] - pos[3, i], ep.L))
            E -= ep.ke * qi * q[j] * erf(ep.α * r) / r
        end
    end
    return E
end

"""
    ewald_energy(pos, q, mol, ep::EwaldParams) -> NamedTuple (real, fourier, self, intra, total)

All four Ewald terms and their sum (the intermolecular Coulomb energy of rigid molecules).
"""
function ewald_energy(pos::AbstractMatrix{Float64}, q::AbstractVector{Float64},
                      mol::AbstractVector{Int}, ep::EwaldParams)
    er = ewald_real_energy(pos, q, mol, ep)
    ef = ewald_recip_energy(pos, q, ep)
    es = ewald_self_energy(q, ep)
    ei = ewald_intra_energy(pos, q, mol, ep)
    return (real=er, fourier=ef, self=es, intra=ei, total=er + ef + es + ei)
end

# ----------------------------------------------------------------------------
# Site-based Lennard-Jones (for validating full force fields on reference configurations)
# ----------------------------------------------------------------------------

"""
    site_lj_energy(pos, σ, ε, mol, L, rc) -> (E_pair, E_lrc)

Intermolecular LJ pair sum over sites with ε > 0 (Lorentz–Berthelot mixing, truncated at rc,
minimum image) plus the standard tail correction applied per LJ-site species
(Allen & Tildesley Eq. 2.136; `compute_lrc_energy_per_particle` in reduced units). Sites with
ε = 0 (e.g. SPC/E hydrogens) carry no LJ interaction.
"""
function site_lj_energy(pos::AbstractMatrix{Float64}, σ::AbstractVector{Float64},
                        ε::AbstractVector{Float64}, mol::AbstractVector{Int}, L::Float64,
                        rc::Float64)
    n = length(σ)
    rc2 = rc * rc
    E = 0.0
    @inbounds for i in 1:n-1
        ε[i] > 0.0 || continue
        xi, yi, zi = pos[1, i], pos[2, i], pos[3, i]
        mi = mol[i]
        for j in i+1:n
            (mol[j] == mi || ε[j] <= 0.0) && continue
            r2 = _min_image_r2(pos[1, j] - xi, pos[2, j] - yi, pos[3, j] - zi, L)
            r2 < rc2 || continue
            σij = 0.5 * (σ[i] + σ[j])
            εij = sqrt(ε[i] * ε[j])
            s2 = σij * σij / r2
            s6 = s2 * s2 * s2
            E += 4.0 * εij * (s6 * s6 - s6)
        end
    end
    # Tail correction: for each LJ species a, N_a * ε_aa * u_tail(ρ_a σ_aa³, rc/σ_aa) with the
    # mean-field g(r)=1 assumption; cross terms are included via the same formula when all LJ
    # sites share (σ, ε), which is the case for the water models used here.
    V = L * L * L
    E_lrc = 0.0
    seen = Dict{Tuple{Float64,Float64},Int}()
    @inbounds for i in 1:n
        ε[i] > 0.0 || continue
        key = (σ[i], ε[i])
        seen[key] = get(seen, key, 0) + 1
    end
    length(seen) <= 1 || error("site_lj_energy: tail correction implemented for a single LJ site species only")
    for ((σa, εa), Na) in seen
        ρa = Na / V * σa^3
        E_lrc += Na * εa * compute_lrc_energy_per_particle(ρa, rc / σa)
    end
    return E, E_lrc
end

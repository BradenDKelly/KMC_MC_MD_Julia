"""
RigidMolecules.jl — rigid, site-based molecules (LJ sites + partial charges) for Metropolis MC and
for the equilibrium kinetic Monte Carlo (eKMC) scheme of Ustinov & Do / Tan et al.

Additive module: `Molecules.jl` (CBMC/GCMC) is untouched; this file reuses its quaternion helpers
(`quaternion_to_rotation_matrix`, `uniform_random_quaternion`) and the accumulators of `eKMC.jl`.

Physics and provenance
- Energy: U = Σ_{i<j} u_ij over molecule pairs, u_ij = Σ_{n∈i, m∈j} [LJ(r^{nm}) + Coulomb(r^{nm})] with a
  site-based spherical cutoff r_c on every site pair; intramolecular terms are excluded (rigid).
  Site vectors are built from the minimum-image COM separation plus body offsets, so a molecule is
  never split between images (Allen & Tildesley, Computer Simulation of Liquids, §1.5/§5.2).
- Coulomb: Wolf damped shifted potential with self and intramolecular-exclusion terms
  (`Electrostatics.jl`; parameters by the Hens & Vlugt 2018 protocol, see dev/wolf_parameter_scan.jl).
- Overlap rule (molecule-pair level, Tan et al. CEJ 316 (2017) Appendix 2; Kelly thesis 2020 §3.5):
  if any LJ site pair is closer than r* = `OVERLAP_R_REDUCED`·σ_ab the whole molecule pair energy is
  set to `OVERLAP_U_REDUCED`·ε_ab (the LJ eKMC uses the same constants).
- Tail corrections: per site-species pair, U_tail = Σ_ab (8π/3)(N_a N_b/V) ε_ab σ_ab³ [(σ_ab/r_c)⁹/3 −
  (σ_ab/r_c)³], P_tail = Σ_ab (16π/3)(N_a N_b/V²) ε_ab σ_ab³ [2(σ_ab/r_c)⁹/3 − (σ_ab/r_c)³]
  (Allen & Tildesley Eqs. 2.136–2.137 summed over species; reduces to `LJLongRange.jl`).
- Pressure: molecular virial, W = Σ_{i<j} R_ij · F_ij = Σ_{i<j} Σ_{nm} (R_ij · r^{nm}/|r^{nm}|)(−du/dr)
  (Tan 2017 Eq. A3.3b; Allen & Tildesley Eq. 2.63), P = N_mol k T / V + W/(3V) + P_tail.
- eKMC: mobility v_i = exp(β u_i) with u_i the removal energy of molecule i, total rate R = Σ v_i,
  residence time Δt = ln(1/ξ)/R, Rosenbluth selection, the selected molecule is placed at a uniform
  random position with a uniform random orientation (Shoemake quaternion) — Tan et al. CEJ 316 (2017)
  §2.1–2.2 and Appendix 1; μ_ex = kT ln⟨R/N⟩ (Eq. 8); NPT via the auxiliary pressure (Eq. 12) and the
  NPT reverse-Widom μ of Kelly's thesis Eq. (A.132).
- Metropolis MC counterpart (for cross-validation): small translations, small rotations about a
  random axis (symmetric proposal, Allen & Tildesley §4.7), uniform reorientations, ln V volume moves
  (Frenkel & Smit §5.4.2), Widom insertion of a rigid molecule with uniform orientation (F&S §7.2.1).
"""

using Random
using StaticArrays
using LinearAlgebra: norm, cross, dot

# ============================================================================
# Templates and force field
# ============================================================================

"""
    RigidTemplate(name, sites_body, site_type, site_q, site_mass)

Rigid molecule: body-frame site coordinates (3 × n, relative to the centre of mass), LJ type per site
(0 = no LJ site), partial charge and mass per site. The constructor re-centres the sites on the COM.
"""
struct RigidTemplate
    name::String
    n_sites::Int
    sites_body::Matrix{Float64}
    site_type::Vector{Int}
    site_q::Vector{Float64}
    site_mass::Vector{Float64}
    extent::Float64             # largest site distance from the COM (interacting sites only)
    function RigidTemplate(name::String, sites_body::Matrix{Float64}, site_type::Vector{Int},
                           site_q::Vector{Float64}, site_mass::Vector{Float64})
        n = size(sites_body, 2)
        length(site_type) == n && length(site_q) == n && length(site_mass) == n ||
            error("RigidTemplate: inconsistent site counts")
        mtot = sum(site_mass)
        mtot > 0.0 || error("RigidTemplate: total mass must be positive")
        com = sites_body * site_mass ./ mtot
        sb = sites_body .- com
        ext = 0.0
        for k in 1:n
            (site_type[k] > 0 || site_q[k] != 0.0) && (ext = max(ext, sqrt(sum(abs2, view(sb, :, k)))))
        end
        new(name, n, sb, copy(site_type), copy(site_q), copy(site_mass), ext)
    end
end

"""
    SiteForceField(σ_types, ϵ_types; rc, coulomb=:none, α=0.0, ke=1.0, use_lrc=true,
                   cap_overlaps=true)

LJ parameters per site type (Lorentz–Berthelot mixing, same tables as `LJParams`), site-based cutoff
`rc` shared by LJ and Coulomb, and the Coulomb model (`:none` or `:wolf` with damping `α`).
`shift_lj=true` uses the cut-and-shifted LJ, u(r) − u(rc), which is continuous at rc so that the
virial pressure needs no impulsive correction (the `:shifted` model of `LJParams`); the tail
corrections are those of the truncated potential in both cases. `cap_factor` sets the molecule-pair overlap cap
U_cap = cap_factor·ϵ_ab (default `OVERLAP_U_REDUCED` = 100, the LJ value of Kelly's thesis §3.5). **For
charged molecules the cap must exceed any binding energy a molecule can gain by overlapping**: with
100 ϵ the SPC/E liquid at 298 K keeps ~1 % of its molecules in O–O overlaps (a capped pair costs
+100 ϵ = 65 kJ/mol while a water molecule's bonds are worth −140 ϵ), which lowers the energy, raises
the density and hides +8000 ϵ of real Coulomb repulsion per overlapped pair (2026-10-12 diagnostic);
`spce_model` therefore uses cap_factor = 1000 (mobilities are clamped at exp(MOBILITY_LOG_MAX) so
multiply-capped molecules cannot overflow R). `cutoff=:molecule` includes all site
pairs of a molecule pair iff the COM separation is below rc (group-based cutoff, Allen & Tildesley
§5.2.1): the Wolf shift terms −q_i q_j erfc(α rc)/rc then cancel exactly for neutral molecules, which
removes the artificial ±|q_i q_j| erfc(α rc)/rc energies that a site-based cutoff gives to molecule
pairs straddling rc (−68 kJ/mol for an O–O pair inside rc = 4.5 Å with the H's outside; ±1.2 kJ/mol
per site pair at 10 Å).
Energies are in units of ϵ when the charges are reduced (q* = q/√(4πε₀ σ ϵ), ke = 1).
"""
struct SiteForceField
    n_types::Int
    σ_mix::Matrix{Float64}
    ϵ_mix::Matrix{Float64}
    rcap2_mix::Matrix{Float64}
    ucap_mix::Matrix{Float64}
    rc::Float64
    rc2::Float64
    coulomb::Symbol
    wolf::WolfParams
    use_lrc::Bool
    cap_overlaps::Bool
    shift_lj::Bool
    u_rc_mix::Matrix{Float64}   # LJ value at rc per type pair (subtracted when shift_lj)
    cutoff::Symbol              # :site (each site pair tested against rc) or :molecule (COM distance)
end

function SiteForceField(σ_types::Vector{Float64}, ϵ_types::Vector{Float64}; rc::Float64,
                        coulomb::Symbol=:none, α::Float64=0.0, ke::Float64=1.0,
                        use_lrc::Bool=true, cap_overlaps::Bool=true, shift_lj::Bool=false,
                        cutoff::Symbol=:site, cap_factor::Float64=Float64(OVERLAP_U_REDUCED))
    coulomb in (:none, :wolf) || error("SiteForceField: coulomb must be :none or :wolf")
    cutoff in (:site, :molecule) || error("SiteForceField: cutoff must be :site or :molecule")
    σ_mix, ϵ_mix = _compute_mixing_tables(σ_types, ϵ_types)
    rcap2 = (OVERLAP_R_REDUCED .* σ_mix) .^ 2
    ucap = cap_factor .* ϵ_mix
    wp = WolfParams(α, rc, ke; model=:wolf)
    u_rc = [4.0 * ϵ_mix[a, b] * ((σ_mix[a, b] / rc)^12 - (σ_mix[a, b] / rc)^6)
            for a in 1:length(σ_types), b in 1:length(σ_types)]
    return SiteForceField(length(σ_types), σ_mix, ϵ_mix, rcap2, ucap, rc, rc * rc, coulomb, wp,
                          use_lrc, cap_overlaps, shift_lj, u_rc, cutoff)
end

# ============================================================================
# SPC/E in reduced units
# ============================================================================

"Coulomb constant e²/(4π ε₀ k_B) in K·Å (CODATA 2010, as on the NIST SPC/E pages)."
const KE_K_ANGSTROM_CODATA2010 = (1.602176565e-19)^2 / (4π * 8.854187817e-12 * 1.3806488e-23) * 1e10

"""
    spce_model(; α_per_angstrom=0.16, rc_angstrom=10.0)

SPC/E water (Berendsen, Grigera, Straatsma, J. Phys. Chem. 91 (1987) 6269; NIST parameter values) in
reduced units with σ_OO = ϵ_OO = k_B = 1. Returns a NamedTuple with the template, the force field,
and the conversion constants (`σ_angstrom`, `ϵ_kelvin`, `q_scale`): T* = T/ϵ_kelvin,
ρ* = ρ[Å⁻³] σ_angstrom³, q* = q[e]·q_scale. Default Wolf parameters from dev/wolf_parameter_scan.jl.
"""
function spce_model(; α_per_angstrom::Float64=0.16, rc_angstrom::Float64=10.0,
                    use_lrc::Bool=true, cap_overlaps::Bool=true, shift_lj::Bool=false,
                    cutoff::Symbol=:site, cap_factor::Float64=1000.0)
    σ_A = 3.16555789
    ϵ_K = 78.19743111
    q_H = 0.42380
    q_scale = sqrt(KE_K_ANGSTROM_CODATA2010 / (ϵ_K * σ_A))   # q* = q · q_scale, so ke* = 1
    θ = 109.47 * π / 180
    r_OH = 1.0 / σ_A
    sites = zeros(Float64, 3, 3)
    sites[:, 1] .= (0.0, 0.0, 0.0)                                   # O
    sites[:, 2] .= (r_OH * sin(θ / 2), r_OH * cos(θ / 2), 0.0)       # H1
    sites[:, 3] .= (-r_OH * sin(θ / 2), r_OH * cos(θ / 2), 0.0)      # H2
    tmpl = RigidTemplate("SPC/E", sites, [1, 0, 0],
                         [-2 * q_H * q_scale, q_H * q_scale, q_H * q_scale],
                         [15.9994, 1.00794, 1.00794])
    ff = SiteForceField([1.0], [1.0]; rc=rc_angstrom / σ_A, coulomb=:wolf,
                        α=α_per_angstrom * σ_A, ke=1.0, use_lrc=use_lrc, cap_overlaps=cap_overlaps,
                        shift_lj=shift_lj, cutoff=cutoff, cap_factor=cap_factor)
    return (template=tmpl, ff=ff, σ_angstrom=σ_A, ϵ_kelvin=ϵ_K, q_scale=q_scale)
end

# ============================================================================
# System state
# ============================================================================

"""
    RigidSystem

`n_mol` rigid molecules in a cubic box of side `L`: COM positions (wrapped to [0, L)), orientation
quaternions (w, x, y, z), and the rotated body offsets of every site (`site_off`, 3 × n_sites).
`site_start[i]` is the index of the site before the first site of molecule i.
"""
mutable struct RigidSystem
    n_mol::Int
    L::Float64
    β::Float64
    templates::Vector{RigidTemplate}
    tmpl::Vector{Int}
    com::Matrix{Float64}
    quat::Matrix{Float64}
    site_off::Matrix{Float64}
    site_start::Vector{Int}
    n_sites_total::Int
    rng::Xoshiro
end

function RigidSystem(templates::Vector{RigidTemplate}, tmpl::Vector{Int}, L::Float64, β::Float64,
                     rng::Xoshiro)
    n_mol = length(tmpl)
    site_start = zeros(Int, n_mol)
    n = 0
    for i in 1:n_mol
        site_start[i] = n
        n += templates[tmpl[i]].n_sites
    end
    sys = RigidSystem(n_mol, L, β, templates, copy(tmpl), zeros(3, n_mol), zeros(4, n_mol),
                      zeros(3, n), site_start, n, rng)
    for i in 1:n_mol
        sys.quat[1, i] = 1.0
        update_site_offsets!(sys, i)
    end
    return sys
end

"Rotate the body-frame sites of molecule i into the lab frame (offsets from the COM)."
function update_site_offsets!(sys::RigidSystem, i::Int)
    t = sys.templates[sys.tmpl[i]]
    R = quaternion_to_rotation_matrix(view(sys.quat, :, i))
    s0 = sys.site_start[i]
    @inbounds for n in 1:t.n_sites
        bx, by, bz = t.sites_body[1, n], t.sites_body[2, n], t.sites_body[3, n]
        sys.site_off[1, s0 + n] = R[1, 1] * bx + R[1, 2] * by + R[1, 3] * bz
        sys.site_off[2, s0 + n] = R[2, 1] * bx + R[2, 2] * by + R[2, 3] * bz
        sys.site_off[3, s0 + n] = R[3, 1] * bx + R[3, 2] * by + R[3, 3] * bz
    end
    return nothing
end

@inline function _wrap01(x::Float64, L::Float64)
    x -= L * floor(x / L)
    return x >= L ? x - L : x
end

"Set COM (wrapped) and orientation of molecule i."
function set_molecule!(sys::RigidSystem, i::Int, com::AbstractVector{Float64}, q::AbstractVector{Float64})
    L = sys.L
    @inbounds begin
        sys.com[1, i] = _wrap01(com[1], L)
        sys.com[2, i] = _wrap01(com[2], L)
        sys.com[3, i] = _wrap01(com[3], L)
        nq = sqrt(q[1]^2 + q[2]^2 + q[3]^2 + q[4]^2)
        sys.quat[1, i] = q[1] / nq; sys.quat[2, i] = q[2] / nq
        sys.quat[3, i] = q[3] / nq; sys.quat[4, i] = q[4] / nq
    end
    update_site_offsets!(sys, i)
    return nothing
end

"Lab-frame site positions (3 × n_sites_total), COM + offset, not wrapped."
function site_positions(sys::RigidSystem)
    pos = similar(sys.site_off)
    @inbounds for i in 1:sys.n_mol
        t = sys.templates[sys.tmpl[i]]
        for n in 1:t.n_sites
            s = sys.site_start[i] + n
            pos[1, s] = sys.com[1, i] + sys.site_off[1, s]
            pos[2, s] = sys.com[2, i] + sys.site_off[2, s]
            pos[3, s] = sys.com[3, i] + sys.site_off[3, s]
        end
    end
    return pos
end

"Flat site arrays (pos, q, mol, σ, ϵ) for the routines in Electrostatics.jl (single LJ type per site)."
function site_arrays(sys::RigidSystem, ff::SiteForceField)
    pos = site_positions(sys)
    n = sys.n_sites_total
    q = zeros(n); mol = zeros(Int, n); σ = zeros(n); ϵ = zeros(n)
    for i in 1:sys.n_mol
        t = sys.templates[sys.tmpl[i]]
        for k in 1:t.n_sites
            s = sys.site_start[i] + k
            q[s] = t.site_q[k]; mol[s] = i
            ty = t.site_type[k]
            if ty > 0
                σ[s] = ff.σ_mix[ty, ty]; ϵ[s] = ff.ϵ_mix[ty, ty]
            end
        end
    end
    return pos, q, mol, σ, ϵ
end

"""
    quaternion_from_rotation_matrix(R)

Unit quaternion (w, x, y, z) of a proper rotation matrix (Shepperd's method).
"""
function quaternion_from_rotation_matrix(R::AbstractMatrix{Float64})
    tr = R[1, 1] + R[2, 2] + R[3, 3]
    if tr > 0
        s = sqrt(tr + 1.0) * 2
        w = 0.25 * s; x = (R[3, 2] - R[2, 3]) / s; y = (R[1, 3] - R[3, 1]) / s; z = (R[2, 1] - R[1, 2]) / s
    elseif R[1, 1] > R[2, 2] && R[1, 1] > R[3, 3]
        s = sqrt(1.0 + R[1, 1] - R[2, 2] - R[3, 3]) * 2
        w = (R[3, 2] - R[2, 3]) / s; x = 0.25 * s; y = (R[1, 2] + R[2, 1]) / s; z = (R[1, 3] + R[3, 1]) / s
    elseif R[2, 2] > R[3, 3]
        s = sqrt(1.0 + R[2, 2] - R[1, 1] - R[3, 3]) * 2
        w = (R[1, 3] - R[3, 1]) / s; x = (R[1, 2] + R[2, 1]) / s; y = 0.25 * s; z = (R[2, 3] + R[3, 2]) / s
    else
        s = sqrt(1.0 + R[3, 3] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = (R[1, 3] + R[3, 1]) / s; y = (R[2, 3] + R[3, 2]) / s; z = 0.25 * s
    end
    q = MVector{4,Float64}(w, x, y, z)
    q ./= norm(q)
    return q
end

"""
    molecule_from_sites!(sys, i, lab_sites)

Set molecule i so that its sites coincide with the given lab-frame positions (3 × n_sites, in a
consistent image). Fits the rotation by Kabsch/SVD-free orthogonal Procrustes using three sites or
the COM frame; requires a non-linear molecule for a unique orientation.
"""
function molecule_from_sites!(sys::RigidSystem, i::Int, lab::AbstractMatrix{Float64})
    t = sys.templates[sys.tmpl[i]]
    m = t.site_mass
    com = lab * m ./ sum(m)
    A = lab .- com                      # lab offsets
    B = t.sites_body                    # body offsets
    # rotation R with A ≈ R B : build orthonormal frames from the first two independent offsets
    function frame(M)
        e1 = M[:, 2] - M[:, 1]; e1 /= norm(e1)
        v = M[:, 3] - M[:, 1]; e2 = v - dot(v, e1) * e1; e2 /= norm(e2)
        e3 = cross(e1, e2)
        return hcat(e1, e2, e3)
    end
    t.n_sites >= 3 || error("molecule_from_sites!: need ≥ 3 sites for a unique orientation")
    R = frame(A) * frame(B)'
    set_molecule!(sys, i, com, quaternion_from_rotation_matrix(R))
    return nothing
end

# ============================================================================
# Pair energies
# ============================================================================

@inline function _min_image(d::Float64, L::Float64)
    return d - L * round(d / L)
end

"""
    pair_energy(sys, ff, i, j) -> u_ij

Intermolecular energy of molecules i and j (site-based cutoff, overlap cap at the pair level).
"""
function pair_energy(sys::RigidSystem, ff::SiteForceField, i::Int, j::Int)::Float64
    L = sys.L
    @inbounds begin
        Rx = _min_image(sys.com[1, j] - sys.com[1, i], L)
        Ry = _min_image(sys.com[2, j] - sys.com[2, i], L)
        Rz = _min_image(sys.com[3, j] - sys.com[3, i], L)
    end
    ti = sys.templates[sys.tmpl[i]]; tj = sys.templates[sys.tmpl[j]]
    R2 = Rx * Rx + Ry * Ry + Rz * Rz
    molcut = ff.cutoff === :molecule
    if molcut
        R2 >= ff.rc2 && return 0.0
    else
        # no site pair can be within rc if the COMs are farther apart than rc + both extents
        rmax = ff.rc + ti.extent + tj.extent
        R2 >= rmax * rmax && return 0.0
    end
    si = sys.site_start[i]; sj = sys.site_start[j]
    rc2 = ff.rc2
    u = 0.0
    @inbounds for n in 1:ti.n_sites
        ox, oy, oz = sys.site_off[1, si + n], sys.site_off[2, si + n], sys.site_off[3, si + n]
        tyn = ti.site_type[n]; qn = ti.site_q[n]
        for m in 1:tj.n_sites
            dx = Rx + sys.site_off[1, sj + m] - ox
            dy = Ry + sys.site_off[2, sj + m] - oy
            dz = Rz + sys.site_off[3, sj + m] - oz
            r2 = dx * dx + dy * dy + dz * dz
            (molcut || r2 < rc2) || continue
            tym = tj.site_type[m]
            if tyn > 0 && tym > 0
                if ff.cap_overlaps && r2 < ff.rcap2_mix[tyn, tym]
                    return ff.ucap_mix[tyn, tym]
                end
                σ2 = ff.σ_mix[tyn, tym]^2
                s6 = (σ2 / r2)^3
                u += 4.0 * ff.ϵ_mix[tyn, tym] * (s6 * s6 - s6)
                ff.shift_lj && (u -= ff.u_rc_mix[tyn, tym])
            end
            qm = tj.site_q[m]
            if ff.coulomb === :wolf && qn != 0.0 && qm != 0.0
                u += molcut ? wolf_pair_u_nocut(sqrt(r2), qn, qm, ff.wolf) : wolf_pair_u(sqrt(r2), qn, qm, ff.wolf)
            end
        end
    end
    return u
end

"""
    pair_energy_virial(sys, ff, i, j) -> (u_ij, w_ij)

As `pair_energy`, also returning the molecular virial contribution R_ij·F_ij (zero for capped pairs).
"""
function pair_energy_virial(sys::RigidSystem, ff::SiteForceField, i::Int, j::Int)
    L = sys.L
    @inbounds begin
        Rx = _min_image(sys.com[1, j] - sys.com[1, i], L)
        Ry = _min_image(sys.com[2, j] - sys.com[2, i], L)
        Rz = _min_image(sys.com[3, j] - sys.com[3, i], L)
    end
    ti = sys.templates[sys.tmpl[i]]; tj = sys.templates[sys.tmpl[j]]
    R2 = Rx * Rx + Ry * Ry + Rz * Rz
    molcut = ff.cutoff === :molecule
    if molcut
        R2 >= ff.rc2 && return 0.0, 0.0
    else
        rmax = ff.rc + ti.extent + tj.extent
        R2 >= rmax * rmax && return 0.0, 0.0
    end
    si = sys.site_start[i]; sj = sys.site_start[j]
    rc2 = ff.rc2
    u = 0.0; w = 0.0
    α = ff.wolf.α; ke = ff.wolf.ke
    @inbounds for n in 1:ti.n_sites
        ox, oy, oz = sys.site_off[1, si + n], sys.site_off[2, si + n], sys.site_off[3, si + n]
        tyn = ti.site_type[n]; qn = ti.site_q[n]
        for m in 1:tj.n_sites
            dx = Rx + sys.site_off[1, sj + m] - ox
            dy = Ry + sys.site_off[2, sj + m] - oy
            dz = Rz + sys.site_off[3, sj + m] - oz
            r2 = dx * dx + dy * dy + dz * dz
            (molcut || r2 < rc2) || continue
            tym = tj.site_type[m]
            fr = 0.0                       # −du/dr · r  (site pair)
            if tyn > 0 && tym > 0
                if ff.cap_overlaps && r2 < ff.rcap2_mix[tyn, tym]
                    return ff.ucap_mix[tyn, tym], 0.0
                end
                σ2 = ff.σ_mix[tyn, tym]^2
                s6 = (σ2 / r2)^3
                ϵ = ff.ϵ_mix[tyn, tym]
                u += 4.0 * ϵ * (s6 * s6 - s6)
                ff.shift_lj && (u -= ff.u_rc_mix[tyn, tym])
                fr += 24.0 * ϵ * (2.0 * s6 * s6 - s6)
            end
            qm = tj.site_q[m]
            if ff.coulomb === :wolf && qn != 0.0 && qm != 0.0
                r = sqrt(r2)
                u += molcut ? wolf_pair_u_nocut(r, qn, qm, ff.wolf) : wolf_pair_u(r, qn, qm, ff.wolf)
                # −r du/dr of ke q q [erfc(αr)/r − erfc(α rc)/rc]
                fr += ke * qn * qm * (erfc(α * r) / r + (2.0 * α / SQRT_PI) * exp(-α * α * r2))
            end
            # R_ij · r̂^{nm} (−du/dr) = (R·r)/r² · (−r du/dr)
            w += (Rx * dx + Ry * dy + Rz * dz) / r2 * fr
        end
    end
    return u, w
end

# ----------------------------------------------------------------------------
# Threaded evaluation of pair rows and totals
# ----------------------------------------------------------------------------
# All the cost of MC trials and eKMC events is the row {u_kj, w_kj}_{j≠k}, whose terms are independent;
# it is split over Julia threads (`julia -t N`) when N_mol ≥ PARALLEL_MIN_MOL. Results are identical
# to the serial path up to floating-point summation order (the reductions below sum per-thread
# partials in a fixed order, so a given thread count is deterministic). Random numbers are only drawn
# on the calling thread.

"""
Upper bound on β·φ_i in the mobility exp(β φ_i). A molecule with k capped pairs has β φ_i ≈ k·β·U_cap;
with U_cap = 1000 ϵ at T* ≈ 4 that is ≈ 260 k, so k ≥ 3 would overflow Float64. Clamping changes only
the rates of states whose time weight is below exp(−600) — negligible by any measure — and keeps the
selection of the overlapping molecule as the next mover certain.
"""
const MOBILITY_LOG_MAX = 600.0
@inline _mobility(β::Float64, φ::Float64) = exp(min(β * φ, MOBILITY_LOG_MAX))

"Smallest number of molecules for which the pair-row loops are threaded."
const PARALLEL_MIN_MOL = 96

@inline _use_threads(N::Int) = Threads.nthreads() > 1 && N >= PARALLEL_MIN_MOL

"Fill row_u[j], row_w[j] = pair energy/virial of molecule k with every j ≠ k."
function eval_row!(row_u::Vector{Float64}, row_w::Vector{Float64}, sys::RigidSystem,
                   ff::SiteForceField, k::Int)
    N = sys.n_mol
    if _use_threads(N)
        Threads.@threads :static for j in 1:N
            if j != k
                u, w = pair_energy_virial(sys, ff, k, j)
                @inbounds row_u[j] = u; @inbounds row_w[j] = w
            end
        end
    else
        @inbounds for j in 1:N
            j == k && continue
            u, w = pair_energy_virial(sys, ff, k, j)
            row_u[j] = u; row_w[j] = w
        end
    end
    @inbounds row_u[k] = 0.0; @inbounds row_w[k] = 0.0
    return nothing
end

"Σ_{j≠k} pair_energy(k, j) with a chunked threaded reduction (fixed summation order per chunk count)."
function _row_energy_sum(sys::RigidSystem, ff::SiteForceField, k::Int)::Float64
    N = sys.n_mol
    if _use_threads(N)
        nt = Threads.nthreads()
        partial = zeros(Float64, nt)
        Threads.@threads :static for t in 1:nt
            lo = (N * (t - 1)) ÷ nt + 1; hi = (N * t) ÷ nt
            acc = 0.0
            for j in lo:hi
                j == k && continue
                acc += pair_energy(sys, ff, k, j)
            end
            @inbounds partial[t] = acc
        end
        return sum(partial)
    else
        u = 0.0
        for j in 1:N
            j == k && continue
            u += pair_energy(sys, ff, k, j)
        end
        return u
    end
end

"Removal energy u_k = Σ_{j≠k} u_kj (the eKMC molecular energy, Tan 2017 Eq. 1a)."
removal_energy(sys::RigidSystem, ff::SiteForceField, k::Int)::Float64 = _row_energy_sum(sys, ff, k)

# Per-task partial sums are indexed by the task number t (rows i ≡ t mod nt, which also balances the
# triangular loop), never by Threads.threadid(): with Julia ≥ 1.12 the default pool's thread ids start
# at 2 when an interactive thread exists, so threadid()-indexed arrays overflow.

"Total intermolecular pair energy (no tail, no Wolf constants)."
function total_pair_energy(sys::RigidSystem, ff::SiteForceField)::Float64
    N = sys.n_mol
    if _use_threads(N)
        nt = Threads.nthreads()
        partial = zeros(Float64, nt)
        Threads.@threads :static for t in 1:nt
            acc = 0.0
            for i in t:nt:N-1, j in i+1:N
                acc += pair_energy(sys, ff, i, j)
            end
            @inbounds partial[t] = acc
        end
        return sum(partial)
    end
    U = 0.0
    for i in 1:N-1, j in i+1:N
        U += pair_energy(sys, ff, i, j)
    end
    return U
end

"Total intermolecular pair energy and molecular virial."
function total_pair_energy_virial(sys::RigidSystem, ff::SiteForceField)
    N = sys.n_mol
    if _use_threads(N)
        nt = Threads.nthreads()
        pU = zeros(Float64, nt); pW = zeros(Float64, nt)
        Threads.@threads :static for t in 1:nt
            aU = 0.0; aW = 0.0
            for i in t:nt:N-1, j in i+1:N
                u, w = pair_energy_virial(sys, ff, i, j)
                aU += u; aW += w
            end
            @inbounds pU[t] = aU; @inbounds pW[t] = aW
        end
        return sum(pU), sum(pW)
    end
    U = 0.0; W = 0.0
    for i in 1:N-1, j in i+1:N
        u, w = pair_energy_virial(sys, ff, i, j)
        U += u; W += w
    end
    return U, W
end

"Number of LJ sites of each type."
function count_lj_sites(sys::RigidSystem, ff::SiteForceField)
    N = zeros(Int, ff.n_types)
    for i in 1:sys.n_mol
        t = sys.templates[sys.tmpl[i]]
        for ty in t.site_type
            ty > 0 && (N[ty] += 1)
        end
    end
    return N
end

"LJ tail correction to the energy for the current volume."
function lrc_energy(sys::RigidSystem, ff::SiteForceField)::Float64
    ff.use_lrc || return 0.0
    N = count_lj_sites(sys, ff)
    V = sys.L^3
    E = 0.0
    for a in 1:ff.n_types, b in 1:ff.n_types
        σ = ff.σ_mix[a, b]; x3 = (σ / ff.rc)^3; x9 = x3^3
        E += (8π / 3) * (N[a] * N[b] / V) * ff.ϵ_mix[a, b] * σ^3 * (x9 / 3 - x3)
    end
    return E
end

"LJ tail correction to the pressure for the current volume."
function lrc_pressure(sys::RigidSystem, ff::SiteForceField)::Float64
    ff.use_lrc || return 0.0
    N = count_lj_sites(sys, ff)
    V = sys.L^3
    P = 0.0
    for a in 1:ff.n_types, b in 1:ff.n_types
        σ = ff.σ_mix[a, b]; x3 = (σ / ff.rc)^3; x9 = x3^3
        P += (16π / 3) * (N[a] * N[b] / V^2) * ff.ϵ_mix[a, b] * σ^3 * (2 * x9 / 3 - x3)
    end
    return P
end

"""
    wolf_constant_energy(sys, ff)

Configuration-independent Wolf terms (self + intramolecular exclusion) of all molecules; add to the
pair energy to compare absolute energies with Ewald.
"""
function wolf_constant_energy(sys::RigidSystem, ff::SiteForceField)::Float64
    ff.coulomb === :wolf || return 0.0
    E = 0.0
    wp = ff.wolf
    for i in 1:sys.n_mol
        t = sys.templates[sys.tmpl[i]]
        for n in 1:t.n_sites
            E -= wp.ke * (erfc(wp.α * wp.rc) / (2 * wp.rc) + wp.α / SQRT_PI) * t.site_q[n]^2
            for m in n+1:t.n_sites
                (t.site_q[n] == 0.0 || t.site_q[m] == 0.0) && continue
                r = norm(t.sites_body[:, n] - t.sites_body[:, m])
                E -= wp.ke * t.site_q[n] * t.site_q[m] / r - wolf_pair_u(r, t.site_q[n], t.site_q[m], wp)
            end
        end
    end
    return E
end

"Total configurational energy U = pair sum + tail correction (+ Wolf constants if `with_constants`)."
function total_energy(sys::RigidSystem, ff::SiteForceField; with_constants::Bool=false)::Float64
    U = total_pair_energy(sys, ff) + lrc_energy(sys, ff)
    with_constants && (U += wolf_constant_energy(sys, ff))
    return U
end

"""
    pressure(sys, ff, W; tail=:virial)

Pressure from the molecular virial, N kT/V + W/(3V) + P_tail. `tail=:virial` uses the virial-route
tail (Allen & Tildesley Eq. 2.137, what LAMMPS/GROMACS add and what the NIST SPC/E EOS used);
`tail=:thermodynamic` uses −∂U_tail/∂V = U_tail/V, the tail pressure that is consistent with
U_tail(V) entering an NPT Metropolis acceptance (the two differ by ≈ (8π/3)ρ²εσ³(σ/r_c)³, ~100 atm
for water at r_c = 10 Å). Use `:thermodynamic` when comparing with `run_rigid_mc!` NPT densities.
"""
function pressure(sys::RigidSystem, ff::SiteForceField, W::Float64; tail::Symbol=:virial)::Float64
    V = sys.L^3
    tail in (:virial, :thermodynamic) || error("pressure: tail must be :virial or :thermodynamic")
    P_tail = tail === :virial ? lrc_pressure(sys, ff) : lrc_energy(sys, ff) / V
    return sys.n_mol / (sys.β * V) + W / (3.0 * V) + P_tail
end

function pressure(sys::RigidSystem, ff::SiteForceField; tail::Symbol=:virial)::Float64
    _, W = total_pair_energy_virial(sys, ff)
    return pressure(sys, ff, W; tail=tail)
end

# ============================================================================
# Random configurations
# ============================================================================

"""
    random_rigid_system!(sys, ff; max_tries=10_000)

Place molecules one by one at uniform random positions/orientations, rejecting placements whose
pair energy with the molecules already placed is capped (overlapping). Equilibrate afterwards.
"""
function random_rigid_system!(sys::RigidSystem, ff::SiteForceField; max_tries::Int=10_000)
    L = sys.L
    for i in 1:sys.n_mol
        placed = false
        for _ in 1:max_tries
            com = MVector{3,Float64}(rand(sys.rng) * L, rand(sys.rng) * L, rand(sys.rng) * L)
            set_molecule!(sys, i, com, uniform_random_quaternion(sys.rng))
            ok = true
            for j in 1:i-1
                u = pair_energy(sys, ff, i, j)
                if u >= minimum(ff.ucap_mix) * 0.999
                    ok = false; break
                end
            end
            if ok
                placed = true; break
            end
        end
        placed || error("random_rigid_system!: could not place molecule $i without overlap")
    end
    return sys
end

# ============================================================================
# Metropolis MC
# ============================================================================

"Hamilton product q1 ⊗ q2 (both (w, x, y, z))."
@inline function quat_mul(q1::AbstractVector{Float64}, q2::AbstractVector{Float64})
    w1, x1, y1, z1 = q1[1], q1[2], q1[3], q1[4]
    w2, x2, y2, z2 = q2[1], q2[2], q2[3], q2[4]
    return MVector{4,Float64}(w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                              w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                              w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                              w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
end

"Random rotation quaternion: uniform random axis, angle uniform in (−δθ, δθ)."
function small_rotation_quaternion(rng::AbstractRNG, δθ::Float64)
    # uniform axis on the sphere
    z = 2.0 * rand(rng) - 1.0
    φ = 2π * rand(rng)
    s = sqrt(1.0 - z * z)
    ax, ay, az = s * cos(φ), s * sin(φ), z
    θ = (2.0 * rand(rng) - 1.0) * δθ
    h = 0.5 * θ
    return MVector{4,Float64}(cos(h), sin(h) * ax, sin(h) * ay, sin(h) * az)
end

mutable struct RigidMCStats
    trans_att::Int; trans_acc::Int
    rot_att::Int; rot_acc::Int
    reor_att::Int; reor_acc::Int
    vol_att::Int; vol_acc::Int
end
RigidMCStats() = RigidMCStats(0, 0, 0, 0, 0, 0, 0, 0)

"Metropolis translation of molecule i by a uniform random displacement in a cube of half-side δ."
function mc_translate!(sys::RigidSystem, ff::SiteForceField, i::Int, δ::Float64, st::RigidMCStats)::Bool
    st.trans_att += 1
    u_old = removal_energy(sys, ff, i)
    com_old = MVector{3,Float64}(sys.com[1, i], sys.com[2, i], sys.com[3, i])
    q = MVector{4,Float64}(sys.quat[1, i], sys.quat[2, i], sys.quat[3, i], sys.quat[4, i])
    com_new = MVector{3,Float64}(com_old[1] + (2rand(sys.rng) - 1) * δ,
                                 com_old[2] + (2rand(sys.rng) - 1) * δ,
                                 com_old[3] + (2rand(sys.rng) - 1) * δ)
    set_molecule!(sys, i, com_new, q)
    ΔU = removal_energy(sys, ff, i) - u_old
    if ΔU <= 0.0 || rand(sys.rng) < exp(-sys.β * ΔU)
        st.trans_acc += 1
        return true
    end
    set_molecule!(sys, i, com_old, q)
    return false
end

"Metropolis rotation of molecule i about a random axis by an angle uniform in (−δθ, δθ)."
function mc_rotate!(sys::RigidSystem, ff::SiteForceField, i::Int, δθ::Float64, st::RigidMCStats)::Bool
    st.rot_att += 1
    u_old = removal_energy(sys, ff, i)
    com = MVector{3,Float64}(sys.com[1, i], sys.com[2, i], sys.com[3, i])
    q_old = MVector{4,Float64}(sys.quat[1, i], sys.quat[2, i], sys.quat[3, i], sys.quat[4, i])
    q_new = quat_mul(small_rotation_quaternion(sys.rng, δθ), q_old)
    set_molecule!(sys, i, com, q_new)
    ΔU = removal_energy(sys, ff, i) - u_old
    if ΔU <= 0.0 || rand(sys.rng) < exp(-sys.β * ΔU)
        st.rot_acc += 1
        return true
    end
    set_molecule!(sys, i, com, q_old)
    return false
end

"Metropolis uniform reorientation of molecule i (Shoemake quaternion)."
function mc_reorient!(sys::RigidSystem, ff::SiteForceField, i::Int, st::RigidMCStats)::Bool
    st.reor_att += 1
    u_old = removal_energy(sys, ff, i)
    com = MVector{3,Float64}(sys.com[1, i], sys.com[2, i], sys.com[3, i])
    q_old = MVector{4,Float64}(sys.quat[1, i], sys.quat[2, i], sys.quat[3, i], sys.quat[4, i])
    set_molecule!(sys, i, com, uniform_random_quaternion(sys.rng))
    ΔU = removal_energy(sys, ff, i) - u_old
    if ΔU <= 0.0 || rand(sys.rng) < exp(-sys.β * ΔU)
        st.reor_acc += 1
        return true
    end
    set_molecule!(sys, i, com, q_old)
    return false
end

"""
    mc_volume_trial!(sys, ff, Pext, max_dlnV, st; U_current) -> (accepted, U_new)

Random walk in ln V with COM scaling (Frenkel & Smit, Understanding Molecular Simulation, §5.4.2,
Eq. 5.4.11): acc = min{1, exp(−β[ΔU + P ΔV] + (N+1) ln(V'/V))}. Returns the new total energy
(pair + tail) so callers need not recompute it.
"""
function mc_volume_trial!(sys::RigidSystem, ff::SiteForceField, Pext::Float64, max_dlnV::Float64,
                          st::RigidMCStats; U_current::Float64)
    st.vol_att += 1
    L_old = sys.L; V_old = L_old^3
    lnV_new = log(V_old) + (2rand(sys.rng) - 1) * max_dlnV
    V_new = exp(lnV_new); L_new = cbrt(V_new)
    s = L_new / L_old
    com_old = copy(sys.com)
    sys.L = L_new
    sys.com .*= s
    U_new = total_energy(sys, ff)
    ΔU = U_new - U_current
    arg = -sys.β * (ΔU + Pext * (V_new - V_old)) + (sys.n_mol + 1) * log(V_new / V_old)
    if arg >= 0.0 || rand(sys.rng) < exp(arg)
        st.vol_acc += 1
        return true, U_new
    end
    sys.L = L_old
    sys.com .= com_old
    return false, U_current
end

"""
    widom_rigid(sys, ff, template_idx; ninsert) -> mean of exp(−β ΔU)

Widom insertion of a rigid molecule at a uniform random position and orientation (Frenkel & Smit
§7.2.1); the excess chemical potential is −kT ln of the returned average (intermolecular part; the
tail contribution of the inserted molecule, if any, is added by `widom_lrc_shift`).
"""
function widom_rigid(sys::RigidSystem, ff::SiteForceField, template_idx::Int; ninsert::Int=1000)
    # temporarily append a ghost molecule
    ghost = RigidSystem(sys.templates, vcat(sys.tmpl, template_idx), sys.L, sys.β, sys.rng)
    ghost.com[:, 1:sys.n_mol] .= sys.com
    ghost.quat[:, 1:sys.n_mol] .= sys.quat
    ghost.site_off[:, 1:sys.n_sites_total] .= sys.site_off
    g = ghost.n_mol
    acc = 0.0
    L = sys.L
    for _ in 1:ninsert
        com = MVector{3,Float64}(rand(sys.rng) * L, rand(sys.rng) * L, rand(sys.rng) * L)
        set_molecule!(ghost, g, com, uniform_random_quaternion(sys.rng))
        ΔU = removal_energy(ghost, ff, g)
        acc += exp(-sys.β * ΔU)
    end
    return acc / ninsert
end

"Change of the LJ tail energy on adding one molecule of `template_idx` at the current volume."
function widom_lrc_shift(sys::RigidSystem, ff::SiteForceField, template_idx::Int)::Float64
    ff.use_lrc || return 0.0
    ghost = RigidSystem(sys.templates, vcat(sys.tmpl, template_idx), sys.L, sys.β, sys.rng)
    return lrc_energy(ghost, ff) - lrc_energy(sys, ff)
end

"""
    run_rigid_mc!(sys, ff; sweeps, δ, δθ, Pext=nothing, max_dlnV=0.1, vol_every=1,
                  sample_every=1, block_size=10, widom_template=0, widom_per_sample=0)

NVT (Pext === nothing) or NPT Metropolis MC. One sweep = n_mol translation + n_mol rotation
attempts (+ one volume trial every `vol_every` sweeps in NPT). Returns block averages of U/N
(pair + tail), P, ρ, V and, if requested, the Widom ⟨exp(−βΔU)⟩ (`widom`) and ⟨V exp(−βΔU)⟩
(`widomV`; in NPT βμ_ex = −ln(⟨V e^{−βΔU}⟩/⟨V⟩), Frenkel & Smit §7.2.1, Kelly thesis Eq. A.133).
"""
function run_rigid_mc!(sys::RigidSystem, ff::SiteForceField; sweeps::Int, δ::Float64, δθ::Float64,
                       Pext::Union{Nothing,Float64}=nothing, max_dlnV::Float64=0.1,
                       vol_every::Int=1, sample_every::Int=1, block_size::Int=10,
                       widom_template::Int=0, widom_per_sample::Int=0, tail_route::Symbol=:virial)
    st = RigidMCStats()
    N = sys.n_mol
    U = total_energy(sys, ff)
    ba_U = BlockAverager(block_size); ba_P = BlockAverager(block_size)
    ba_rho = BlockAverager(block_size); ba_w = BlockAverager(block_size)
    ba_V = BlockAverager(block_size); ba_wV = BlockAverager(block_size)   # NPT Widom: <V e^{-βΔU}>/<V>
    for sweep in 1:sweeps
        for _ in 1:N
            i = rand(sys.rng, 1:N)
            mc_translate!(sys, ff, i, δ, st)
            i = rand(sys.rng, 1:N)
            mc_rotate!(sys, ff, i, δθ, st)
        end
        if Pext !== nothing && sweep % vol_every == 0
            U = total_energy(sys, ff)
            _, U = mc_volume_trial!(sys, ff, Pext, max_dlnV, st; U_current=U)
        end
        if sweep % sample_every == 0
            Up, W = total_pair_energy_virial(sys, ff)
            push!(ba_U, (Up + lrc_energy(sys, ff)) / N)
            push!(ba_P, pressure(sys, ff, W; tail=tail_route))
            V = sys.L^3
            push!(ba_rho, N / V)
            push!(ba_V, V)
            if widom_template > 0 && widom_per_sample > 0
                w = widom_rigid(sys, ff, widom_template; ninsert=widom_per_sample)
                push!(ba_w, w)
                push!(ba_wV, V * w)
            end
        end
    end
    return (U=ba_U, P=ba_P, rho=ba_rho, V=ba_V, widom=ba_w, widomV=ba_wV, stats=st)
end

# ============================================================================
# Equilibrium kinetic Monte Carlo (Tan / Ustinov scheme) for rigid molecules
# ============================================================================

"""
    RigidEKMCState

Pair-energy matrix (n_mol × n_mol), molecular energies φ_i = Σ_j u_ij, mobilities m_i = exp(β φ_i)
and the total rate R, kept consistent after every event (Tan 2017 Appendix 1).
"""
mutable struct RigidEKMCState
    pair_u::Matrix{Float64}
    pair_w::Matrix{Float64}     # molecular pair virials R_ij·F_ij, kept alongside the energies
    phi::Vector{Float64}
    wsum::Vector{Float64}       # Σ_j w_ij; total virial W = ½ Σ_i wsum_i
    m::Vector{Float64}
    R::Float64
    debug_check_every::Int
    events::Int
    row_u::Vector{Float64}      # scratch: candidate row for Metropolis trials inside eKMC
    row_w::Vector{Float64}
    mc_trials::Int              # Metropolis trials made by the MC-kernel events
    mc_accepted::Int
    mc_events::Int              # number of MC-kernel events
end

function init_rigid_ekmc(sys::RigidSystem, ff::SiteForceField; debug_check_every::Int=0)
    N = sys.n_mol
    ek = RigidEKMCState(zeros(N, N), zeros(N, N), zeros(N), zeros(N), zeros(N), 0.0, debug_check_every, 0,
                        zeros(N), zeros(N), 0, 0, 0)
    rebuild_rigid_ekmc!(ek, sys, ff)
    return ek
end

"""
    ekmc_mc_trial!(ek, sys, ff, i, δ, δθ; rotate) -> accepted::Bool

One Metropolis trial (translation by a uniform displacement in a cube of half-side δ, or rotation
about a random axis by an angle uniform in ±δθ) on molecule i, evaluated and committed through the
eKMC pair bookkeeping (row i of energies and virials, φ_j, Σ_j w_ij, mobilities, R). The proposal is
symmetric, so the kernel preserves the canonical distribution (Metropolis et al. 1953).
"""
function ekmc_mc_trial!(ek::RigidEKMCState, sys::RigidSystem, ff::SiteForceField, i::Int,
                        δ::Float64, δθ::Float64; rotate::Bool)::Bool
    N = sys.n_mol
    ek.mc_trials += 1
    com_old = MVector{3,Float64}(sys.com[1, i], sys.com[2, i], sys.com[3, i])
    q_old = MVector{4,Float64}(sys.quat[1, i], sys.quat[2, i], sys.quat[3, i], sys.quat[4, i])
    if rotate
        set_molecule!(sys, i, com_old, quat_mul(small_rotation_quaternion(sys.rng, δθ), q_old))
    else
        set_molecule!(sys, i, MVector{3,Float64}(com_old[1] + (2rand(sys.rng) - 1) * δ,
                                                 com_old[2] + (2rand(sys.rng) - 1) * δ,
                                                 com_old[3] + (2rand(sys.rng) - 1) * δ), q_old)
    end
    eval_row!(ek.row_u, ek.row_w, sys, ff, i)
    ΔU = 0.0
    @inbounds for j in 1:N
        j == i && continue
        ΔU += ek.row_u[j] - ek.pair_u[i, j]
    end
    if ΔU <= 0.0 || rand(sys.rng) < exp(-sys.β * ΔU)
        β = sys.β
        phii = 0.0; wi = 0.0
        @inbounds for j in 1:N
            j == i && continue
            du = ek.row_u[j] - ek.pair_u[i, j]; dw = ek.row_w[j] - ek.pair_w[i, j]
            ek.pair_u[i, j] = ek.row_u[j]; ek.pair_u[j, i] = ek.row_u[j]
            ek.pair_w[i, j] = ek.row_w[j]; ek.pair_w[j, i] = ek.row_w[j]
            if du != 0.0
                ek.phi[j] += du
                ek.m[j] = _mobility(β, ek.phi[j])
            end
            ek.wsum[j] += dw
            phii += ek.row_u[j]; wi += ek.row_w[j]
        end
        ek.phi[i] = phii; ek.wsum[i] = wi
        ek.m[i] = _mobility(β, phii)
        ek.R = sum(ek.m)                      # never incremental (see update_rigid_ekmc_after_move!)
        ek.mc_accepted += 1
        return true
    end
    set_molecule!(sys, i, com_old, q_old)
    return false
end

"""
    ekmc_mc_sweep!(ek, sys, ff, δ, δθ)

The MC-kernel event: N translation and N rotation trials on randomly chosen molecules, then R is
recomputed from the mobilities to prevent round-off drift.
"""
function ekmc_mc_sweep!(ek::RigidEKMCState, sys::RigidSystem, ff::SiteForceField, δ::Float64, δθ::Float64)
    N = sys.n_mol
    for _ in 1:N
        ekmc_mc_trial!(ek, sys, ff, rand(sys.rng, 1:N), δ, δθ; rotate=false)
        ekmc_mc_trial!(ek, sys, ff, rand(sys.rng, 1:N), δ, δθ; rotate=true)
    end
    ek.R = sum(ek.m)
    ek.mc_events += 1
    return nothing
end

"Total molecular virial from the maintained pair virials."
ekmc_virial(ek::RigidEKMCState) = 0.5 * sum(ek.wsum)

"Recompute all pair energies and virials, φ, mobilities and R from scratch."
function rebuild_rigid_ekmc!(ek::RigidEKMCState, sys::RigidSystem, ff::SiteForceField)
    N = sys.n_mol
    fill!(ek.pair_u, 0.0); fill!(ek.phi, 0.0); fill!(ek.pair_w, 0.0); fill!(ek.wsum, 0.0)
    if _use_threads(N)
        nt = Threads.nthreads()
        Threads.@threads :static for t in 1:nt            # rows i ≡ t (mod nt); distinct cells: no races
            for i in t:nt:N-1, j in i+1:N
                u, w = pair_energy_virial(sys, ff, i, j)
                @inbounds begin
                    ek.pair_u[i, j] = u; ek.pair_u[j, i] = u
                    ek.pair_w[i, j] = w; ek.pair_w[j, i] = w
                end
            end
        end
        @inbounds for i in 1:N
            ek.phi[i] = sum(view(ek.pair_u, :, i)); ek.wsum[i] = sum(view(ek.pair_w, :, i))
        end
    else
        for i in 1:N-1, j in i+1:N
            u, w = pair_energy_virial(sys, ff, i, j)
            ek.pair_u[i, j] = u; ek.pair_u[j, i] = u
            ek.pair_w[i, j] = w; ek.pair_w[j, i] = w
            ek.phi[i] += u; ek.phi[j] += u
            ek.wsum[i] += w; ek.wsum[j] += w
        end
    end
    R = 0.0
    for i in 1:N
        ek.m[i] = _mobility(sys.β, ek.phi[i]); R += ek.m[i]
    end
    ek.R = R
    (R > 0.0 && isfinite(R)) || error("rigid eKMC: total rate R = $R")
    return nothing
end

"Appendix-1 update after molecule k has been moved: row/column k of energies and virials, all φ_j, Σ_j w_ij, mobilities, R."
function update_rigid_ekmc_after_move!(ek::RigidEKMCState, sys::RigidSystem, ff::SiteForceField, k::Int)
    N = sys.n_mol
    β = sys.β
    eval_row!(ek.row_u, ek.row_w, sys, ff, k)          # threaded when N ≥ PARALLEL_MIN_MOL
    phik = 0.0; wk = 0.0
    @inbounds for j in 1:N
        j == k && continue
        u_old = ek.pair_u[k, j]; w_old = ek.pair_w[k, j]
        u_new = ek.row_u[j]; w_new = ek.row_w[j]
        if u_new != u_old
            ek.pair_u[k, j] = u_new; ek.pair_u[j, k] = u_new
            ek.phi[j] += u_new - u_old
            ek.m[j] = _mobility(β, ek.phi[j])
        end
        if w_new != w_old
            ek.pair_w[k, j] = w_new; ek.pair_w[j, k] = w_new
            ek.wsum[j] += w_new - w_old
        end
        phik += u_new; wk += w_new
    end
    ek.phi[k] = phik; ek.wsum[k] = wk
    ek.m[k] = _mobility(β, phik)
    # R is always resummed: mobilities span ~50 orders of magnitude (a relocated molecule can have
    # exp(βu) ≈ 1e48), so an incremental R = R + Δm suffers catastrophic cancellation when such a term
    # is removed (found by the LJ-diatomic MC↔eKMC test, 2026-10-11). The O(N) sum is negligible; only
    # the exponentials of the molecules whose φ changed are recomputed above.
    R = sum(ek.m)
    ek.R = R
    (R > 0.0 && isfinite(R)) || error("rigid eKMC: total rate R = $R after moving molecule $k")
    return nothing
end

"Rosenbluth (tower) selection with probability m_i / R."
function select_by_mobility(ek::RigidEKMCState, rng::AbstractRNG)::Int
    r = rand(rng) * ek.R
    c = 0.0
    @inbounds for i in eachindex(ek.m)
        c += ek.m[i]
        c > r && return i
    end
    return length(ek.m)
end

"""
    rigid_ekmc_step!(ek, sys, ff, acc::ChemicalPotentialAccumulator) -> Δt

One event: residence time Δt = ln(1/ξ)/R of the *current* configuration (accumulated into `acc`
with R/N), selection of a molecule by mobility, and its relocation to a uniform random position with
a uniform random orientation. Returns Δt (the weight of the configuration that was just left).
"""
function rigid_ekmc_step!(ek::RigidEKMCState, sys::RigidSystem, ff::SiteForceField,
                          acc::ChemicalPotentialAccumulator)::Float64
    R = ek.R
    u = rand(sys.rng)
    u = max(eps(Float64), min(u, 1.0 - eps(Float64)))
    dt = log(1 / u) / R
    N = sys.n_mol
    acc.t_total += dt
    RN = R / N
    acc.S += RN * dt; acc.S_sq += RN * RN * dt; acc.t_total_sq += dt * dt; acc.count += 1
    k = select_by_mobility(ek, sys.rng)
    L = sys.L
    com = MVector{3,Float64}(rand(sys.rng) * L, rand(sys.rng) * L, rand(sys.rng) * L)
    set_molecule!(sys, k, com, uniform_random_quaternion(sys.rng))
    update_rigid_ekmc_after_move!(ek, sys, ff, k)
    ek.events += 1
    if ek.debug_check_every > 0 && ek.events % ek.debug_check_every == 0
        phi_saved = copy(ek.phi); R_saved = ek.R; W_saved = ekmc_virial(ek)
        rebuild_rigid_ekmc!(ek, sys, ff)
        err = maximum(abs.(phi_saved .- ek.phi))
        relR = abs(R_saved - ek.R) / ek.R
        errW = abs(W_saved - ekmc_virial(ek))
        (err > 1e-8 || relR > 1e-8 || errW > 1e-8 * max(1.0, abs(W_saved))) &&
            @warn "rigid eKMC bookkeeping drift" err relR errW
    end
    return dt
end

"""
    rigid_ekmc_newton_volume_move!(ek, sys, ff, Pext, max_dV, P̄, slope)

Volume update of Kelly (PhD thesis, Guelph 2020, §3.2 Eq. 3.3; Kelly & Smith, Mol. Phys. 117 (2019)
2778): solve P̄(V) = P_set by a Newton step, ΔV = (P_set − P̄)/(dP̄/dV), with the block-averaged
(kMC-time-weighted) pressure P̄ of the block just finished. Kelly estimates the slope from the two
previous blocks; here `run_rigid_ekmc!` passes the running least-squares slope of P̄ against V over
all blocks so far, which makes the gain nearly independent of the noise in the current P̄ (a
two-point secant gain is correlated with that noise and shifted the fixed point by ≈ ±6 % in P for 32
LJ diatomics, dev experiment 2026-10-11). Safeguards: if the slope is not negative (noise) the step
falls back to the Tan auxiliary-pressure move, and during the full-gain phase |ΔV| is capped at
`max_dV`. `gain` < 1 implements the Robbins–Monro stochastic-approximation schedule
(H. Robbins, S. Monro, Ann. Math. Statist. 22 (1951) 400) that `run_rigid_ekmc!` applies after
`vol_equil_moves`: a_k = 1/(k − k0 + 1), so the volume converges to the root of ⟨P̄⟩(V) = P_set with
vanishing controller noise instead of random-walking around it. A capped, constant-gain step pins a
median-like statistic of the skewed P̄ distribution (⟨P⟩ 8 % above P* for 64 SPC/E at T* = 12), and
Tan's threshold rule has a Jensen bias (E[P̄/(P̄+P*)] = 1/2 ⇒ ⟨P̄⟩ > P* whenever P̄ fluctuates; ρ 35–60 %
too high for 32 LJ diatomics).
"""
function rigid_ekmc_newton_volume_move!(ek::RigidEKMCState, sys::RigidSystem, ff::SiteForceField,
                                        Pext::Float64, max_dV::Float64, P̄::Float64, slope::Float64;
                                        gain::Float64=1.0)
    V = sys.L^3
    if !(slope < 0.0) || !isfinite(slope)
        return rigid_ekmc_volume_move!(ek, sys, ff, Pext, max_dV; p=P̄)
    end
    dV = gain * (Pext - P̄) / slope
    # the cap is an equilibration safeguard; once the gain decays the step is small anyway
    gain >= 1.0 && (dV = clamp(dV, -max_dV, max_dV))
    V_new = V + dV
    V_new > 0.0 || (V_new = 0.5 * V)
    L_new = cbrt(V_new)
    sys.com .*= L_new / sys.L
    sys.L = L_new
    rebuild_rigid_ekmc!(ek, sys, ff)
    return P̄
end

"""
    estimate_relocation_rate(sys, ff; events=20_000) -> ⟨R⟩_time

Pilot run of pure eKMC (modifies `sys`) returning the time-weighted mean of the relocation rate,
⟨R⟩ = M / t_total = N exp(βμ_ex) (Tan 2018 Eq. 7), the natural scale for `mc_kernel_rate`.
"""
function estimate_relocation_rate(sys::RigidSystem, ff::SiteForceField; events::Int=20_000)
    ek = init_rigid_ekmc(sys, ff)
    acc = ChemicalPotentialAccumulator()
    t = 0.0
    for _ in 1:events
        t += rigid_ekmc_step!(ek, sys, ff, acc)
    end
    return events / t
end

"Time-weighted block averager: blocks of `block_events` events, block mean = Σ x Δt / Σ Δt."
mutable struct TimeWeightedBlockAverager
    block_events::Int
    sum_x::Float64
    sum_t::Float64
    n::Int
    block_means::Vector{Float64}
end
TimeWeightedBlockAverager(block_events::Int) = TimeWeightedBlockAverager(block_events, 0.0, 0.0, 0, Float64[])
function Base.push!(b::TimeWeightedBlockAverager, x::Float64, dt::Float64)
    b.sum_x += x * dt; b.sum_t += dt; b.n += 1
    if b.n >= b.block_events
        push!(b.block_means, b.sum_x / b.sum_t)
        b.sum_x = 0.0; b.sum_t = 0.0; b.n = 0
    end
    return b
end
function mean(b::TimeWeightedBlockAverager)
    isempty(b.block_means) && return b.sum_t > 0 ? b.sum_x / b.sum_t : NaN
    return sum(b.block_means) / length(b.block_means)
end
function stderr(b::TimeWeightedBlockAverager)
    n = length(b.block_means)
    n < 2 && return NaN
    m = mean(b)
    return sqrt(sum((x - m)^2 for x in b.block_means) / (n - 1) / n)
end

"""
    rigid_ekmc_volume_move!(ek, sys, ff, Pext, max_dV; p=nothing) -> p used

Tan et al. (2017) Eq. 12 auxiliary-pressure volume move: p_aux = n (p + p*), expand by n ΔV_max if
p > p_aux else contract; COMs are rescaled, orientations kept, everything rebuilt (Appendix 1).
`p` is the pressure compared with p*: the instantaneous virial pressure if not given, or the
kMC-time-weighted partial average since the previous volume move ("partial average pressure",
Tan, Do, Nicholson, Mol. Phys. 114 (2016) 186) supplied by `run_rigid_ekmc!`. The instantaneous
choice is biased: configurations at volume-move events follow the jump-chain (event) distribution,
which over-weights short-lived overlapping states with large virial pressure, so the volume settles
where the event-averaged pressure, not the time (Boltzmann) average, equals p* (dev experiment
2026-10-11: ρ 18 % too low for an LJ diatomic at T* = 2, P* = 0.1).
"""
function rigid_ekmc_volume_move!(ek::RigidEKMCState, sys::RigidSystem, ff::SiteForceField,
                                 Pext::Float64, max_dV::Float64; p::Union{Nothing,Float64}=nothing)
    if p === nothing
        _, W = total_pair_energy_virial(sys, ff)
        p = pressure(sys, ff, W)
    end
    n = rand(sys.rng)
    p_aux = n * (p + Pext)
    V = sys.L^3
    V_new = p > p_aux ? V + n * max_dV : V - n * max_dV
    V_new > 0.0 || (V_new = 0.5 * V)
    L_new = cbrt(V_new)
    sys.com .*= L_new / sys.L
    sys.L = L_new
    rebuild_rigid_ekmc!(ek, sys, ff)
    return p
end

"""
    run_rigid_ekmc!(sys, ff; events, Pext=nothing, max_dV=0.0, vol_every_sweeps=10,
                    sample_every=10, block_events=1000, debug_check_every=0,
                    adapt_dV_moves=0, adapt_dV_window=100,
                    vol_pressure=:partial_average, vol_scheme=:newton,
                    vol_equil_moves=200, burnin_events=0)

NVT (Pext === nothing) or NPT eKMC. In NPT a volume move is made every `vol_every_sweeps`·N events
with maximum change `max_dV`, using the kMC-time-weighted block average of the virial pressure over
every event since the previous volume move (`vol_pressure = :partial_average`, Tan 2016;
`:instantaneous` reproduces the biased variant, see `rigid_ekmc_volume_move!`). The virial is
maintained incrementally with the pair energies, so the pressure costs nothing per event. With
`vol_scheme = :newton` the first `vol_equil_moves` volume moves use the full Newton gain (capped at
`max_dV`), after which the Robbins–Monro gain 1/(k − k0 + 1) lets the volume converge; observables
are accumulated only after `burnin_events` events (choose ≥ `vol_equil_moves`·`vol_every_sweeps`·N).
`tail_route` (`:virial` or `:thermodynamic`, see `pressure`) selects the tail pressure used both in
the volume controller and in the reported P.

MC-kernel hybrid (`mc_kernel_rate` = λ > 0): an extra event of constant rate λ whose action is one
Metropolis sweep (N translations of half-width `mc_δ` and N rotations of half-angle `mc_δθ`, done
through the eKMC bookkeeping by `ekmc_mc_sweep!`). The generator is Q_eKMC + λ(K − I) with K the
Metropolis kernel; both terms satisfy detailed balance with the canonical distribution, so the
chain stays balanced and the time averages canonical. Residence times are ln(1/ξ)/(R + λ): with λ
of the order of R in relaxed states the weights of relaxed configurations become nearly uniform,
which removes the heavy tail that uniform relocation alone produces in hydrogen-bonded liquids
(JOURNAL entry 5), while a mis-placed molecule (R ≫ λ) is still relocated at once. The μ observable
is unchanged: ⟨R/N⟩_time with R = Σ exp(βu_i) (the relocation rate, not R + λ); the Tan shortcut
⟨R⟩ = M/t_total no longer applies. λ must be a constant for the run (a state-dependent λ would break
balance); `estimate_relocation_rate` gives the scale from a short pilot. With λ > 0 most events are
sweeps, so set `vol_every_events` (volume move every that many events) instead of `vol_every_sweeps`.
Validated 2026-10-11 (JOURNAL entries 7–9): agrees with MC and pure eKMC at a supercritical state for
λ = 1 and 10; at liquid density it reaches a lower-energy state that plain MC from a random start never
finds in 160k sweeps, and MC continued from that state stays there. `vol_scheme = :newton` (default) applies Kelly's secant
update `rigid_ekmc_newton_volume_move!` (first move: Tan rule); `:auxiliary` applies Tan's Eq. 12
threshold rule every time. If `adapt_dV_moves > 0`, ΔV_max is reset to
half the standard deviation of the kMC-time-weighted volume distribution every `adapt_dV_window`
volume moves during the first `adapt_dV_moves` moves (Tan 2017 Appendix 4). Use it only as a short
burn-in and read the final value from `max_dV` in the result: because the volume walk's spread is
itself set by ΔV_max, the feedback shrinks ΔV_max towards zero if left on (dev experiment
2026-10-11: 10.4 → 0.01 in 15 windows). Observables are time-weighted (Tan 2017 Eq. 5)
with the residence time of the sampled configuration. U = ½ Σ φ_i and W = ½ Σ_i Σ_j w_ij come from the
maintained pair matrices, so U, P and ρ are all sampled every `sample_every` events for free. Returns block averages of U/N (pair + tail), P, ρ,
the NVT chemical-potential accumulator (μ_ex = `mu_ex(acc, T)`), blocked R/N (`RN`, for
μ_ex = T ln mean(RN) with error T·stderr/mean) and, in NPT, the Eq. A.132 accumulator (`mu_ex_npt`)
plus blocked (R/N)/V (`RNV`) and V (`V`) so that μ_res = T ln(mean(RNV)·mean(V)) has a block error.
"""
function run_rigid_ekmc!(sys::RigidSystem, ff::SiteForceField; events::Int,
                         Pext::Union{Nothing,Float64}=nothing, max_dV::Float64=0.0,
                         vol_every_sweeps::Int=10, sample_every::Int=10, block_events::Int=1000,
                         debug_check_every::Int=0, adapt_dV_moves::Int=0, adapt_dV_window::Int=100,
                         vol_pressure::Symbol=:partial_average, vol_scheme::Symbol=:newton,
                         vol_equil_moves::Int=200, burnin_events::Int=0, tail_route::Symbol=:virial,
                         mc_kernel_rate::Float64=0.0, mc_δ::Float64=0.1, mc_δθ::Float64=0.3,
                         vol_every_events::Int=0)
    N = sys.n_mol
    λ = mc_kernel_rate
    vol_period = vol_every_events > 0 ? vol_every_events : vol_every_sweeps * N
    λ >= 0.0 || error("run_rigid_ekmc!: mc_kernel_rate must be ≥ 0")
    vol_pressure in (:partial_average, :instantaneous) ||
        error("run_rigid_ekmc!: vol_pressure must be :partial_average or :instantaneous")
    vol_scheme in (:newton, :auxiliary) ||
        error("run_rigid_ekmc!: vol_scheme must be :newton (Kelly 2020 Eq. 3.3) or :auxiliary (Tan 2017 Eq. 12)")
    # running least-squares regression of the block pressure P̄ on V for the Newton slope
    reg_n = 0; reg_V = 0.0; reg_P = 0.0; reg_VV = 0.0; reg_VP = 0.0
    p_part_sum = 0.0; p_part_t = 0.0
    ek = init_rigid_ekmc(sys, ff; debug_check_every=debug_check_every)
    acc = ChemicalPotentialAccumulator()
    acc_npt = NPTChemicalPotentialAccumulator()
    ba_U = TimeWeightedBlockAverager(block_events ÷ sample_every)
    ba_P = TimeWeightedBlockAverager(block_events ÷ sample_every)
    ba_rho = TimeWeightedBlockAverager(block_events ÷ sample_every)
    ba_V = TimeWeightedBlockAverager(block_events ÷ sample_every)
    ba_RN = TimeWeightedBlockAverager(block_events ÷ sample_every)   # R/N for blocked μ errors
    ba_RNV = TimeWeightedBlockAverager(block_events ÷ sample_every)  # (R/N)/V for blocked NPT μ (A.132)
    t_total = 0.0
    # Tan 2017 Appendix 4: during the first `adapt_dV_moves` volume moves, every `adapt_dV_window`
    # moves set ΔV_max to half the standard deviation of the kMC-time-weighted volume histogram
    vol_moves = 0
    t_since_vol = 0.0
    win_V = Float64[]; win_t = Float64[]
    max_dV_current = max_dV
    dV_history = Float64[]
    for ev in 1:events
        # weight of the current configuration: draw its Δt here and reuse it in the step
        R = ek.R                       # relocation rate Σ exp(βu_i) (the μ observable)
        Rtot = R + λ                   # total event rate incl. the MC-kernel event
        u = rand(sys.rng); u = max(eps(Float64), min(u, 1.0 - eps(Float64)))
        dt = log(1 / u) / Rtot
        V = sys.L^3
        if Pext !== nothing
            # pressure is free from the maintained virial; time-weight it over every event of the block
            p_part_sum += pressure(sys, ff, ekmc_virial(ek); tail=tail_route) * dt; p_part_t += dt
        end
        if ev % sample_every == 0 && ev > burnin_events
            Up = 0.5 * sum(ek.phi)
            push!(ba_U, (Up + lrc_energy(sys, ff)) / N, dt)
            push!(ba_P, pressure(sys, ff, ekmc_virial(ek); tail=tail_route), dt)
            push!(ba_rho, N / V, dt)
            push!(ba_V, V, dt)
            push!(ba_RN, R / N, dt)
            push!(ba_RNV, R / N / V, dt)
            push!(acc_npt, R / N, V, dt)
        end
        if ev > burnin_events
            acc.t_total += dt; RN = R / N
            acc.S += RN * dt; acc.S_sq += RN * RN * dt; acc.t_total_sq += dt * dt; acc.count += 1
        end
        # the event: MC-kernel sweep with probability λ/(R+λ), otherwise a relocation
        if λ > 0.0 && rand(sys.rng) * Rtot < λ
            ekmc_mc_sweep!(ek, sys, ff, mc_δ, mc_δθ)
        else
            k = select_by_mobility(ek, sys.rng)
            L = sys.L
            set_molecule!(sys, k, MVector{3,Float64}(rand(sys.rng) * L, rand(sys.rng) * L, rand(sys.rng) * L),
                          uniform_random_quaternion(sys.rng))
            update_rigid_ekmc_after_move!(ek, sys, ff, k)
        end
        ek.events += 1
        t_total += dt
        t_since_vol += dt
        if ek.debug_check_every > 0 && ek.events % ek.debug_check_every == 0
            phi_saved = copy(ek.phi); R_saved = ek.R; W_saved = ekmc_virial(ek)
            rebuild_rigid_ekmc!(ek, sys, ff)
            (maximum(abs.(phi_saved .- ek.phi)) > 1e-8 || abs(R_saved - ek.R) / ek.R > 1e-8 ||
             abs(W_saved - ekmc_virial(ek)) > 1e-8 * max(1.0, abs(W_saved))) &&
                @warn "rigid eKMC bookkeeping drift" maximum(abs.(phi_saved .- ek.phi)) abs(R_saved - ek.R) / ek.R
        end
        if Pext !== nothing && ev % vol_period == 0
            if vol_moves < adapt_dV_moves
                push!(win_V, sys.L^3); push!(win_t, t_since_vol)
                if length(win_V) >= adapt_dV_window
                    tw = sum(win_t)
                    mV = sum(win_V .* win_t) / tw
                    sdV = sqrt(max(sum(((win_V .- mV) .^ 2) .* win_t) / tw, 0.0))
                    sdV > 0.0 && (max_dV_current = 0.5 * sdV)
                    push!(dV_history, max_dV_current)
                    empty!(win_V); empty!(win_t)
                end
            end
            t_since_vol = 0.0
            p_cmp = (vol_pressure === :partial_average && p_part_t > 0.0) ? p_part_sum / p_part_t : nothing
            if vol_scheme === :newton && p_cmp !== nothing
                V_now = sys.L^3
                reg_n += 1; reg_V += V_now; reg_P += p_cmp; reg_VV += V_now * V_now; reg_VP += V_now * p_cmp
                var = reg_VV - reg_V * reg_V / reg_n
                slope = (reg_n >= 2 && var > 0.0) ? (reg_VP - reg_V * reg_P / reg_n) / var : NaN
                # Robbins–Monro gain after the full-gain equilibration phase: a_k = 1/(k − k0 + 1)
                gain = vol_moves < vol_equil_moves ? 1.0 : 1.0 / (vol_moves - vol_equil_moves + 1)
                rigid_ekmc_newton_volume_move!(ek, sys, ff, Pext, max_dV_current, p_cmp, slope; gain=gain)
            else
                rigid_ekmc_volume_move!(ek, sys, ff, Pext, max_dV_current; p=p_cmp)
            end
            p_part_sum = 0.0; p_part_t = 0.0
            vol_moves += 1
        end
    end
    return (U=ba_U, P=ba_P, rho=ba_rho, V=ba_V, RN=ba_RN, RNV=ba_RNV, mu=acc, mu_npt=acc_npt,
            t_total=t_total, ek=ek, vol_moves=vol_moves, max_dV=max_dV_current, dV_history=dV_history,
            mc_events=ek.mc_events, mc_acceptance=ek.mc_trials > 0 ? ek.mc_accepted / ek.mc_trials : NaN)
end

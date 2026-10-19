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
    bonded::Union{Nothing,BondedTerms}   # intramolecular terms (reduced units) ⇒ a FLEXIBLE molecule; nothing = rigid
    function RigidTemplate(name::String, sites_body::Matrix{Float64}, site_type::Vector{Int},
                           site_q::Vector{Float64}, site_mass::Vector{Float64};
                           bonded::Union{Nothing,BondedTerms}=nothing)
        n = size(sites_body, 2)
        length(site_type) == n && length(site_q) == n && length(site_mass) == n ||
            error("RigidTemplate: inconsistent site counts")
        bonded === nothing || bonded.n_sites == n || error("RigidTemplate: bonded terms for a different site count")
        mtot = sum(site_mass)
        mtot > 0.0 || error("RigidTemplate: total mass must be positive")
        com = sites_body * site_mass ./ mtot
        sb = sites_body .- com
        ext = 0.0
        for k in 1:n
            (site_type[k] > 0 || site_q[k] != 0.0) && (ext = max(ext, sqrt(sum(abs2, view(sb, :, k)))))
        end
        new(name, n, sb, copy(site_type), copy(site_q), copy(site_mass), ext, bonded)
    end
end

"A template with bonded terms is a flexible molecule (DESIGN §7); `sites_body` is then only its reference geometry."
isflexible(t::RigidTemplate) = t.bonded !== nothing

"""
    SiteForceField(σ_types, ϵ_types; rc, coulomb=:none, α=0.0, ke=1.0, use_lrc=true,
                   cap_overlaps=true)

LJ parameters per site type (Lorentz–Berthelot mixing, same tables as `LJParams`), site-based cutoff
`rc` shared by LJ and Coulomb, and the Coulomb model: `:none`, `:wolf` (damping `α`), or `:rf` (GROMACS
reaction field with `epsilon_rf`, default ∞; `RFParams`), the model GROMACS can run too.
`shift_lj=true` uses the cut-and-shifted LJ, u(r) − u(rc), which is continuous at rc so that the
virial pressure needs no impulsive correction (the `:shifted` model of `LJParams`); the tail
corrections are those of the truncated potential in both cases. `cap_factor` sets the molecule-pair overlap cap
U_cap = cap_factor·ϵ_ab (default `OVERLAP_U_REDUCED` = 100, the LJ value of Kelly's thesis §3.5). **For
charged molecules the cap must exceed any binding energy a molecule can gain by overlapping**: with
100 ϵ the SPC/E liquid at 298 K keeps ~1 % of its molecules in O–O overlaps (a capped pair costs
+100 ϵ = 65 kJ/mol while a water molecule's bonds are worth −140 ϵ), which lowers the energy, raises
the density and hides +8000 ϵ of real Coulomb repulsion per overlapped pair (2026-10-12 diagnostic);
`spce_model` therefore uses cap_factor = 1000 (mobilities are clamped at exp(MOBILITY_LOG_MAX) so
multiply-capped molecules cannot overflow R). For polyatomic molecules use `kT_overlap_cap` (DESIGN §9): U_cap = 100 kT for every molecule pair, orientation-scanned trigger radii and min(u_bare, U_cap). `cutoff=:molecule` includes all site
pairs of a molecule pair iff the COM separation is below rc (group-based cutoff, Allen & Tildesley
§5.2.1): the Wolf shift terms −q_i q_j erfc(α rc)/rc then cancel exactly for neutral molecules, which
removes the artificial ±|q_i q_j| erfc(α rc)/rc energies that a site-based cutoff gives to molecule
pairs straddling rc (−68 kJ/mol for an O–O pair inside rc = 4.5 Å with the H's outside; ±1.2 kJ/mol
per site pair at 10 Å). `mixing=:geometric` uses σ_ab = √(σ_a σ_b) (OPLS) instead of the arithmetic
Lorentz rule; ϵ_ab = √(ϵ_a ϵ_b) in both cases.
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
    rf::RFParams                # reaction-field parameters (used when coulomb === :rf)
    cap_min_u::Float64          # kT overlap cap (DESIGN §9): a molecule pair is min(u_bare, cap_min_u); Inf = legacy rule only
end

function SiteForceField(σ_types::Vector{Float64}, ϵ_types::Vector{Float64}; rc::Float64,
                        coulomb::Symbol=:none, α::Float64=0.0, ke::Float64=1.0,
                        use_lrc::Bool=true, cap_overlaps::Bool=true, shift_lj::Bool=false,
                        cutoff::Symbol=:site, cap_factor::Float64=Float64(OVERLAP_U_REDUCED),
                        mixing::Symbol=:lorentz_berthelot, epsilon_rf::Float64=Inf, epsilon_r::Float64=1.0)
    coulomb in (:none, :wolf, :rf, :ewald) || error("SiteForceField: coulomb must be :none, :wolf, :rf or :ewald")
    cutoff in (:site, :molecule) || error("SiteForceField: cutoff must be :site or :molecule")
    coulomb === :ewald && cutoff !== :site && error("SiteForceField: :ewald needs the site cut-off (real-space erfc sum)")
    mixing in (:lorentz_berthelot, :geometric) ||
        error("SiteForceField: mixing must be :lorentz_berthelot or :geometric")
    σ_mix, ϵ_mix = _compute_mixing_tables(σ_types, ϵ_types)
    if mixing === :geometric
        # OPLS convention (Jorgensen, Maxwell, Tirado-Rives, JACS 118 (1996) 11225): σ_ab = √(σ_a σ_b) as well
        for a in eachindex(σ_types), b in eachindex(σ_types)
            σ_mix[a, b] = sqrt(σ_types[a] * σ_types[b])
        end
    end
    rcap2 = (OVERLAP_R_REDUCED .* σ_mix) .^ 2
    ucap = cap_factor .* ϵ_mix
    wp = WolfParams(α, rc, ke; model=:wolf)
    u_rc = [4.0 * ϵ_mix[a, b] * ((σ_mix[a, b] / rc)^12 - (σ_mix[a, b] / rc)^6)
            for a in 1:length(σ_types), b in 1:length(σ_types)]
    rfp = RFParams(rc, ke; ε_rf=epsilon_rf, ε_r=epsilon_r)
    return SiteForceField(length(σ_types), σ_mix, ϵ_mix, rcap2, ucap, rc, rc * rc, coulomb, wp,
                          use_lrc, cap_overlaps, shift_lj, u_rc, cutoff, rfp, Inf)
end

# ============================================================================
# SPC/E in reduced units
# ============================================================================

"Coulomb constant e²/(4π ε₀ k_B) in K·Å (CODATA 2010, as on the NIST SPC/E pages)."
const KE_K_ANGSTROM_CODATA2010 = (1.602176565e-19)^2 / (4π * 8.854187817e-12 * 1.3806488e-23) * 1e10

"""
    spce_model(; α_per_angstrom=0.16, rc_angstrom=10.0, coulomb=:wolf, epsilon_rf=Inf, cutoff=:site, ...)

SPC/E water (Berendsen, Grigera, Straatsma, J. Phys. Chem. 91 (1987) 6269; NIST parameter values) in
reduced units with σ_OO = ϵ_OO = k_B = 1. Returns a NamedTuple with the template, the force field,
and the conversion constants (`σ_angstrom`, `ϵ_kelvin`, `q_scale`): T* = T/ϵ_kelvin,
ρ* = ρ[Å⁻³] σ_angstrom³, q* = q[e]·q_scale. Default Wolf parameters from dev/wolf_parameter_scan.jl.
"""
function spce_model(; α_per_angstrom::Float64=0.16, rc_angstrom::Float64=10.0,
                    use_lrc::Bool=true, cap_overlaps::Bool=true, shift_lj::Bool=false,
                    cutoff::Symbol=:site, cap_factor::Float64=1000.0, coulomb::Symbol=:wolf,
                    epsilon_rf::Float64=Inf)
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
    ff = SiteForceField([1.0], [1.0]; rc=rc_angstrom / σ_A, coulomb=coulomb,
                        α=α_per_angstrom * σ_A, ke=1.0, use_lrc=use_lrc, cap_overlaps=cap_overlaps,
                        shift_lj=shift_lj, cutoff=cutoff, cap_factor=cap_factor, epsilon_rf=epsilon_rf)
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
    body::Vector{Matrix{Float64}}   # flexible molecules: current COM-centred conformation in the body frame (3 × n); rigid: 3 × 0
    ext::Vector{Float64}            # per-molecule extent (template extent for rigid molecules, current conformation for flexible)
    recip::Union{Nothing,EwaldRecip}   # reciprocal-space Ewald state (coulomb = :ewald, `attach_ewald!`); nothing otherwise
    cells::Union{Nothing,MolCells,SiteCells}   # cell list for the real-space rows (`attach_cells!`); nothing = all pairs
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
    body = [isflexible(templates[tmpl[i]]) ? copy(templates[tmpl[i]].sites_body) : zeros(3, 0) for i in 1:n_mol]
    ext = [templates[tmpl[i]].extent for i in 1:n_mol]
    sys = RigidSystem(n_mol, L, β, templates, copy(tmpl), zeros(3, n_mol), zeros(4, n_mol),
                      zeros(3, n), site_start, n, rng, body, ext, nothing, nothing)
    for i in 1:n_mol
        sys.quat[1, i] = 1.0
        update_site_offsets!(sys, i)
    end
    return sys
end

"Body-frame site coordinates of molecule i: the template's for a rigid molecule, its own conformation if flexible."
@inline body_sites(sys::RigidSystem, i::Int) = isflexible(sys.templates[sys.tmpl[i]]) ? sys.body[i] : sys.templates[sys.tmpl[i]].sites_body

"Rotate the body-frame sites of molecule i into the lab frame (offsets from the COM)."
function update_site_offsets!(sys::RigidSystem, i::Int)
    t = sys.templates[sys.tmpl[i]]
    R = quaternion_to_rotation_matrix(view(sys.quat, :, i))
    s0 = sys.site_start[i]
    B = body_sites(sys, i)
    @inbounds for n in 1:t.n_sites
        bx, by, bz = B[1, n], B[2, n], B[3, n]
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

"""
    set_conformation!(sys, i, body::AbstractMatrix)

Replace the conformation of flexible molecule i by `body` (3 × n lab-oriented site coordinates relative
to any origin): the COM is removed, the orientation quaternion reset to identity (the conformation
carries the orientation), `site_off` and `ext` updated. The COM position `com[:, i]` is unchanged.
"""
function set_conformation!(sys::RigidSystem, i::Int, body::AbstractMatrix{Float64})
    t = sys.templates[sys.tmpl[i]]
    isflexible(t) || error("set_conformation!: molecule $i is rigid")
    m = t.site_mass
    c = body * m ./ sum(m)
    B = sys.body[i]
    size(B) == size(body) || (B = zeros(3, t.n_sites); sys.body[i] = B)
    ext = 0.0
    @inbounds for n in 1:t.n_sites, d in 1:3
        B[d, n] = body[d, n] - c[d]
    end
    @inbounds for n in 1:t.n_sites
        (t.site_type[n] > 0 || t.site_q[n] != 0.0) && (ext = max(ext, sqrt(B[1, n]^2 + B[2, n]^2 + B[3, n]^2)))
    end
    sys.ext[i] = ext
    sys.quat[1, i] = 1.0; sys.quat[2, i] = 0.0; sys.quat[3, i] = 0.0; sys.quat[4, i] = 0.0
    update_site_offsets!(sys, i)
    return nothing
end

"Lab-frame site coordinates of molecule i (COM + offsets, not wrapped), 3 × n_sites."
function molecule_sites(sys::RigidSystem, i::Int)
    t = sys.templates[sys.tmpl[i]]
    s0 = sys.site_start[i]
    X = zeros(3, t.n_sites)
    @inbounds for n in 1:t.n_sites, d in 1:3
        X[d, n] = sys.com[d, i] + sys.site_off[d, s0 + n]
    end
    return X
end

"""
    flex_intra_energy(t::RigidTemplate, ff, X) -> U_intra (reduced units)

Intramolecular energy of a flexible molecule with site coordinates X (3 × n, any frame): bonded terms +
1-4 and further intramolecular LJ/Coulomb (plain Coulomb with ke = 1 for the 1-4 pairs, as GROMACS's
[pairs]), plus, for `coulomb = :rf`, GROMACS's reaction-field correction of every excluded pair
(`rf_excluded_u`, conformation-dependent; −ke k_rf |μ|² for a neutral molecule) and the reaction-field pair
term for the non-excluded intramolecular pairs — so the Hamiltonian stays identical to a GROMACS
reaction-field run of the flexible topology. `:wolf` is not defined for flexible molecules.
"""
function flex_intra_energy(t::RigidTemplate, ff::SiteForceField, X::AbstractMatrix{Float64})::Float64
    bt = t.bonded
    bt === nothing && return 0.0
    ff.coulomb === :wolf && error("flex_intra_energy: Wolf electrostatics are not defined for flexible molecules (use :rf or :none)")
    eb, ea, ed = bonded_energy(bt, X)
    E = eb + ea + ed
    @inbounds for (a, b) in bt.pairs14
        r = _dist(X, a, b)
        if t.site_type[a] > 0 && t.site_type[b] > 0
            s6 = (ff.σ_mix[t.site_type[a], t.site_type[b]] / r)^6
            E += bt.fudge_lj * 4.0 * ff.ϵ_mix[t.site_type[a], t.site_type[b]] * (s6 * s6 - s6)
        end
        E += bt.fudge_qq * t.site_q[a] * t.site_q[b] / r
    end
    @inbounds for (a, b) in bt.nonbonded
        r = _dist(X, a, b)
        if t.site_type[a] > 0 && t.site_type[b] > 0
            s6 = (ff.σ_mix[t.site_type[a], t.site_type[b]] / r)^6
            E += 4.0 * ff.ϵ_mix[t.site_type[a], t.site_type[b]] * (s6 * s6 - s6)
        end
        if ff.coulomb === :rf
            r < ff.rc && (E += rf_pair_u(r, t.site_q[a], t.site_q[b], ff.rf))
        else
            E += t.site_q[a] * t.site_q[b] / r
        end
    end
    if ff.coulomb === :rf
        nb = Set(bt.nonbonded)
        for a in 1:t.n_sites-1, b in a+1:t.n_sites
            (a, b) in nb && continue
            (t.site_q[a] == 0.0 || t.site_q[b] == 0.0) && continue
            E += rf_excluded_u(_dist(X, a, b), t.site_q[a], t.site_q[b], ff.rf)
        end
        E += rf_self_energy(t.site_q, ff.rf)
    end
    return E
end

"Intramolecular energy of molecule i in its current conformation (0 for rigid molecules)."
intra_energy_molecule(sys::RigidSystem, ff::SiteForceField, i::Int)::Float64 =
    isflexible(sys.templates[sys.tmpl[i]]) ? flex_intra_energy(sys.templates[sys.tmpl[i]], ff, sys.body[i]) : 0.0

"Total intramolecular energy of all flexible molecules."
function total_intra_energy(sys::RigidSystem, ff::SiteForceField)::Float64
    E = 0.0
    for i in 1:sys.n_mol
        E += intra_energy_molecule(sys, ff, i)
    end
    return E
end

"""
    rotatable_bonds(t::RigidTemplate) -> Vector{Tuple{Int,Int,Vector{Int}}}

Bonds (a, b) of a flexible template that are not in a ring and have at least one more atom on each side; for each,
the atoms that a torsion move rotates (the smaller side, without its axis atom). Rotating that fragment about the
a–b axis by φ is volume-preserving in Cartesian coordinates (an orthogonal map of the moved atoms; the axis atoms do
not move) and its inverse is the rotation by −φ, so a symmetric φ density gives a symmetric proposal (JOURNAL 79–80).
"""
function rotatable_bonds(t::RigidTemplate)
    bt = t.bonded
    bt === nothing && return Tuple{Int,Int,Vector{Int}}[]
    n = t.n_sites
    adj = [Int[] for _ in 1:n]
    for b in bt.bonds
        push!(adj[b.i], b.j); push!(adj[b.j], b.i)
    end
    out = Tuple{Int,Int,Vector{Int}}[]
    for b in bt.bonds
        a, c = b.i, b.j
        # side of c without crossing a–c
        seen = falses(n); seen[c] = true; stack = [c]; ring = false
        while !isempty(stack)
            x = pop!(stack)
            for y in adj[x]
                (x == c && y == a) && continue
                y == a && (ring = true; continue)
                seen[y] || (seen[y] = true; push!(stack, y))
            end
        end
        ring && continue
        side_c = findall(seen); side_a = setdiff(1:n, side_c)
        (length(side_c) >= 2 && length(side_a) >= 2) || continue
        mov = length(side_c) <= length(side_a) ? setdiff(side_c, [c]) : setdiff(side_a, [a])
        push!(out, (a, c, mov))
    end
    return out
end

"Rotate the sites `mov` of X (3 × n, any frame) by φ about the axis through sites a and b (Rodrigues)."
function rotate_fragment!(X::AbstractMatrix{Float64}, a::Int, b::Int, mov::Vector{Int}, φ::Float64)
    ux = X[1, b] - X[1, a]; uy = X[2, b] - X[2, a]; uz = X[3, b] - X[3, a]
    nu = sqrt(ux^2 + uy^2 + uz^2); ux /= nu; uy /= nu; uz /= nu
    cφ = cos(φ); sφ = sin(φ)
    @inbounds for m in mov
        vx = X[1, m] - X[1, a]; vy = X[2, m] - X[2, a]; vz = X[3, m] - X[3, a]
        d = ux * vx + uy * vy + uz * vz
        cx = uy * vz - uz * vy; cy = uz * vx - ux * vz; cz = ux * vy - uy * vx
        X[1, m] = X[1, a] + vx * cφ + cx * sφ + ux * d * (1 - cφ)
        X[2, m] = X[2, a] + vy * cφ + cy * sφ + uy * d * (1 - cφ)
        X[3, m] = X[3, a] + vz * cφ + cz * sφ + uz * d * (1 - cφ)
    end
    return X
end

"""
    ConformerReservoir

Sample of isolated-molecule (ideal-gas) conformations of one flexible template, p_id(c) ∝ exp(−βU_intra(c)),
stored COM-centred (3 × n each). Built by `build_reservoir` with vacuum Metropolis MC (single-site Cartesian
displacements: exact for the Cartesian measure of harmonic bonds, Frenkel & Smit §13.2); drawn uniformly by
`draw_conformer`. The relocation of a flexible molecule takes its new conformation from here (DESIGN §7.2):
π(x) k(x→x′) = e^{−βU_rest} e^{−βU_intra(c)} e^{−βU_intra(c′)}/(Z_id V 8π²) is symmetric, so detailed balance
holds with the rate exp(βu_inter) unchanged. Reusing a finite sample introduces correlation but no bias
(the sample is unbiased); rebuild with `build_reservoir` when in doubt.
"""
struct ConformerReservoir
    template::Int                      # template index the conformers belong to
    confs::Vector{Matrix{Float64}}
    U_intra::Vector{Float64}           # intramolecular energy of each conformer (reduced units)
    acceptance::Float64
end

"""
    build_reservoir(templates, template_idx, ff, β, rng; M=10_000, burnin_sweeps=2_000, stride_sweeps=20, δ=0.03,
                    torsion=false, φmax=π)

Vacuum Metropolis MC on one molecule of `template_idx`: a sweep = one displacement attempt per site
(uniform in a cube of half-side δ, reduced units). `M` conformations are stored every `stride_sweeps`
sweeps after `burnin_sweeps`. `torsion=true` adds, per sweep, one rigid-fragment torsion attempt per rotatable bond
(random bond, φ uniform in [−φmax, φmax], `rotatable_bonds`): single-site steps of δ do not cross heavy-atom torsion
barriers (1-propanol C–C–C–O: 0 flips in 4e5 sweeps, JOURNAL 79).
"""
function build_reservoir(templates::Vector{RigidTemplate}, template_idx::Int, ff::SiteForceField, β::Float64,
                         rng::AbstractRNG; M::Int=10_000, burnin_sweeps::Int=2_000, stride_sweeps::Int=20,
                         δ::Float64=0.03, torsion::Bool=false, φmax::Float64=Float64(π))
    t = templates[template_idx]
    isflexible(t) || error("build_reservoir: template $(t.name) is rigid")
    rb = torsion ? rotatable_bonds(t) : Tuple{Int,Int,Vector{Int}}[]
    X = copy(t.sites_body)
    U = flex_intra_energy(t, ff, X)
    confs = Matrix{Float64}[]; Us = Float64[]
    att = 0; acc = 0
    m = t.site_mass; mtot = sum(m)
    total = burnin_sweeps + M * stride_sweeps
    for sweep in 1:total
        for _ in 1:t.n_sites
            k = rand(rng, 1:t.n_sites)
            o1, o2, o3 = X[1, k], X[2, k], X[3, k]
            X[1, k] += (2rand(rng) - 1) * δ; X[2, k] += (2rand(rng) - 1) * δ; X[3, k] += (2rand(rng) - 1) * δ
            Un = flex_intra_energy(t, ff, X)
            att += 1
            if Un <= U || rand(rng) < exp(-β * (Un - U))
                U = Un; acc += 1
            else
                X[1, k] = o1; X[2, k] = o2; X[3, k] = o3
            end
        end
        for _ in 1:length(rb)                                  # torsion moves (torsion = true)
            (a, b, mov) = rb[rand(rng, 1:length(rb))]
            Xo = X[:, mov]
            rotate_fragment!(X, a, b, mov, (2rand(rng) - 1) * φmax)
            Un = flex_intra_energy(t, ff, X)
            att += 1
            if Un <= U || rand(rng) < exp(-β * (Un - U))
                U = Un; acc += 1
            else
                X[:, mov] .= Xo
            end
        end
        if sweep > burnin_sweeps && (sweep - burnin_sweeps) % stride_sweeps == 0
            c = X * m ./ mtot
            push!(confs, X .- c); push!(Us, U)
        end
    end
    return ConformerReservoir(template_idx, confs, Us, acc / att)
end

"""
    build_reservoir_ekmc(templates, template_idx, ff, β, rng; M=10_000, burnin_sweeps=2_000, stride_sweeps=20, δ=0.03,
                         ν=1.0, νD=1.0)

Vacuum conformer reservoir by rejection-free continuous-time sampling (DESIGN §11.3 E3; no Metropolis step). The
molecule carries one candidate displacement per site (d_n uniform in a cube of half-side δ, the L-move construction of
`LocalMoves` applied to single sites): event n moves site n by d_n and stores −d_n in the slot, at rate
ν exp(−½β[U_intra(x + d_n) − U_intra(x)]) (detailed balance on the space augmented by the slots); a refresh event of
rate νD redraws all slots. Conformers are recorded at equally spaced kMC times τ (time sampling of a continuous-time
chain gives draws from its stationary density e^{−βU_intra}); τ = stride_sweeps · n_sites · ⟨Δt⟩ of the burn-in
(`burnin_sweeps` · n_sites events), i.e. the spacing of the Metropolis reservoir measured in events. `acceptance` of the
result holds the fraction of events that were moves (not refreshes).
`torsion=true` adds one slot per rotatable bond (`rotatable_bonds`): rotate the fragment by Φ_j (uniform in [−φmax, φmax]),
store −Φ_j, rate ν exp(−½β ΔU_intra); the refresh redraws them too (JOURNAL 79–80: single-site slots alone do not cross
heavy-atom torsion barriers).
"""
function build_reservoir_ekmc(templates::Vector{RigidTemplate}, template_idx::Int, ff::SiteForceField, β::Float64,
                              rng::AbstractRNG; M::Int=10_000, burnin_sweeps::Int=2_000, stride_sweeps::Int=20,
                              δ::Float64=0.03, ν::Float64=1.0, νD::Float64=1.0, torsion::Bool=false, φmax::Float64=Float64(π))
    t = templates[template_idx]
    isflexible(t) || error("build_reservoir_ekmc: template $(t.name) is rigid")
    n = t.n_sites
    rb = torsion ? rotatable_bonds(t) : Tuple{Int,Int,Vector{Int}}[]   # one torsion slot per rotatable bond (torsion = true)
    nt = length(rb)
    Φ = [(2rand(rng) - 1) * φmax for _ in 1:nt]
    Ut = zeros(nt); wt = zeros(nt)
    X = copy(t.sites_body)
    U = flex_intra_energy(t, ff, X)
    D = [(2rand(rng) - 1) * δ for _ in 1:3, _ in 1:n]          # slot n: displacement of site n
    Uc = zeros(n); w = zeros(n)
    function cand!()                                           # candidate energies of all slots
        for k in 1:n
            for d in 1:3; X[d, k] += D[d, k]; end
            Uc[k] = flex_intra_energy(t, ff, X)
            for d in 1:3; X[d, k] -= D[d, k]; end
        end
        for j in 1:nt
            (a, b, mov) = rb[j]
            Xo = X[:, mov]
            rotate_fragment!(X, a, b, mov, Φ[j]); Ut[j] = flex_intra_energy(t, ff, X)
            X[:, mov] .= Xo
        end
    end
    cand!()
    m = t.site_mass; mtot = sum(m)
    confs = Matrix{Float64}[]; Us = Float64[]
    nmoves = 0; nev = 0
    t_burn = 0.0; nburn = burnin_sweeps * n
    τ = NaN; t_next = 0.0; tnow = 0.0
    while length(confs) < M
        W = 0.0
        for k in 1:n
            w[k] = ν * exp(min(-0.5 * β * (Uc[k] - U), MOBILITY_LOG_MAX)); W += w[k]
        end
        Wt = 0.0
        for j in 1:nt
            wt[j] = ν * exp(min(-0.5 * β * (Ut[j] - U), MOBILITY_LOG_MAX)); Wt += wt[j]
        end
        Rtot = W + Wt + νD
        ξ0 = rand(rng); ξ0 = max(eps(Float64), min(ξ0, 1.0 - eps(Float64)))
        dt = log(1 / ξ0) / Rtot
        nev += 1
        if nev <= nburn
            t_burn += dt
            if nev == nburn
                τ = stride_sweeps * n * t_burn / nburn; t_next = τ; tnow = 0.0
            end
        else
            # the current state occupies [tnow, tnow + dt): record it at every sampling time inside
            while t_next < tnow + dt && length(confs) < M
                c = X * m ./ mtot
                push!(confs, X .- c); push!(Us, U)
                t_next += τ
            end
            tnow += dt
        end
        ξ = rand(rng) * Rtot
        if ξ < W
            k = n
            for i in 1:n
                ξ < w[i] && (k = i; break)
                ξ -= w[i]
            end
            for d in 1:3; X[d, k] += D[d, k]; D[d, k] = -D[d, k]; end
            U = Uc[k]
            nmoves += 1
        elseif ξ < W + Wt
            ξ -= W; j = nt
            for i in 1:nt
                ξ < wt[i] && (j = i; break)
                ξ -= wt[i]
            end
            (a, b, mov) = rb[j]
            rotate_fragment!(X, a, b, mov, Φ[j]); Φ[j] = -Φ[j]          # the slot now holds the move back
            U = Ut[j]
            nmoves += 1
        else
            for k in 1:n, d in 1:3
                D[d, k] = (2rand(rng) - 1) * δ
            end
            for j in 1:nt
                Φ[j] = (2rand(rng) - 1) * φmax
            end
        end
        cand!()
    end
    return ConformerReservoir(template_idx, confs, Us, nmoves / nev)
end

"Uniformly drawn conformation (a reference into the reservoir — do not modify)."
@inline draw_conformer(res::ConformerReservoir, rng::AbstractRNG) = res.confs[rand(rng, 1:length(res.confs))]

"Copy a COM-centred conformation into molecule i's body frame and update its extent (no site_off update)."
function _set_body!(sys::RigidSystem, i::Int, B::AbstractMatrix{Float64})
    t = sys.templates[sys.tmpl[i]]
    size(sys.body[i]) == size(B) || (sys.body[i] = zeros(3, t.n_sites))
    copyto!(sys.body[i], B)
    ext = 0.0
    @inbounds for n in 1:t.n_sites
        (t.site_type[n] > 0 || t.site_q[n] != 0.0) && (ext = max(ext, sqrt(B[1, n]^2 + B[2, n]^2 + B[3, n]^2)))
    end
    sys.ext[i] = ext
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

# Site cut-off with large molecules in a small box (L/2 < rc + ext_i + ext_j): a site pair's nearest image need not be the
# image of the nearest COM, so each site separation is then wrapped on its own (GROMACS's convention; one image per site
# pair because rc < L/2). Molecule pairs with |R_k| + ext_i + ext_j ≤ L/2 on every axis keep the COM image — the same site
# separations, bitwise. The molecule cut-off defines the image by the COM and is never wrapped. Per-site images need
# rc < L/2 (each site pair then has one image within rc; GROMACS refuses smaller boxes); with rc ≥ L/2 (tests that sample
# isolated pair geometries in a tiny box) the COM image is kept: the energy of the two molecules as one isolated pair.
@inline _pair_wraps(Rx::Float64, Ry::Float64, Rz::Float64, e::Float64, L::Float64) =
    abs(Rx) + e > 0.5 * L || abs(Ry) + e > 0.5 * L || abs(Rz) + e > 0.5 * L
@inline _site_images(ff, L::Float64) = ff.cutoff !== :molecule && ff.rc < 0.5 * L

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
        rmax = ff.rc + sys.ext[i] + sys.ext[j]
        R2 >= rmax * rmax && return 0.0
    end
    si = sys.site_start[i]; sj = sys.site_start[j]
    rc2 = ff.rc2
    wrap = _site_images(ff, L) && _pair_wraps(Rx, Ry, Rz, sys.ext[i] + sys.ext[j], L)
    u = 0.0
    @inbounds for n in 1:ti.n_sites
        ox, oy, oz = sys.site_off[1, si + n], sys.site_off[2, si + n], sys.site_off[3, si + n]
        tyn = ti.site_type[n]; qn = ti.site_q[n]
        for m in 1:tj.n_sites
            dx = Rx + sys.site_off[1, sj + m] - ox
            dy = Ry + sys.site_off[2, sj + m] - oy
            dz = Rz + sys.site_off[3, sj + m] - oz
            if wrap
                dx = _min_image(dx, L); dy = _min_image(dy, L); dz = _min_image(dz, L)
            end
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
            elseif ff.coulomb === :rf && qn != 0.0 && qm != 0.0
                u += rf_pair_u(sqrt(r2), qn, qm, ff.rf)
            elseif ff.coulomb === :ewald && qn != 0.0 && qm != 0.0
                r = sqrt(r2)
                u += ff.wolf.ke * qn * qm * erfc(ff.wolf.α * r) / r      # Ewald real space (reciprocal part: sys.recip)
            end
        end
    end
    ff.cap_overlaps && u > ff.cap_min_u && return ff.cap_min_u     # kT cap: min(u_bare, U_cap) (DESIGN §9)
    return u
end

"""
    _site_pair_uf(ff, tyn, qn, tym, qm, r2) -> (u, −r du/dr)

One site pair at r² < rc² without the overlap cap — the terms of `pair_energy_virial` (kept identical; used by the site
cell list, MolCells.jl).
"""
@inline function _site_pair_uf(ff::SiteForceField, tyn::Int, qn::Float64, tym::Int, qm::Float64, r2::Float64)
    u = 0.0; fr = 0.0
    @inbounds if tyn > 0 && tym > 0
        σ2 = ff.σ_mix[tyn, tym]^2
        s6 = (σ2 / r2)^3
        ϵ = ff.ϵ_mix[tyn, tym]
        u += 4.0 * ϵ * (s6 * s6 - s6)
        ff.shift_lj && (u -= ff.u_rc_mix[tyn, tym])
        fr += 24.0 * ϵ * (2.0 * s6 * s6 - s6)
    end
    if qn != 0.0 && qm != 0.0
        α = ff.wolf.α; ke = ff.wolf.ke
        if ff.coulomb === :wolf
            r = sqrt(r2)
            u += wolf_pair_u(r, qn, qm, ff.wolf)
            fr += ke * qn * qm * (erfc(α * r) / r + (2.0 * α / SQRT_PI) * exp(-α * α * r2))
        elseif ff.coulomb === :rf
            r = sqrt(r2)
            u += rf_pair_u(r, qn, qm, ff.rf)
            fr += rf_pair_fr(r, qn, qm, ff.rf)
        elseif ff.coulomb === :ewald
            r = sqrt(r2)
            u += ke * qn * qm * erfc(α * r) / r
            fr += ke * qn * qm * (erfc(α * r) / r + (2.0 * α / SQRT_PI) * exp(-α * α * r2))
        end
    end
    return u, fr
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
        rmax = ff.rc + sys.ext[i] + sys.ext[j]
        R2 >= rmax * rmax && return 0.0, 0.0
    end
    si = sys.site_start[i]; sj = sys.site_start[j]
    rc2 = ff.rc2
    wrap = _site_images(ff, L) && _pair_wraps(Rx, Ry, Rz, sys.ext[i] + sys.ext[j], L)
    u = 0.0; w = 0.0
    α = ff.wolf.α; ke = ff.wolf.ke
    @inbounds for n in 1:ti.n_sites
        ox, oy, oz = sys.site_off[1, si + n], sys.site_off[2, si + n], sys.site_off[3, si + n]
        tyn = ti.site_type[n]; qn = ti.site_q[n]
        for m in 1:tj.n_sites
            dx = Rx + sys.site_off[1, sj + m] - ox
            dy = Ry + sys.site_off[2, sj + m] - oy
            dz = Rz + sys.site_off[3, sj + m] - oz
            Rwx = Rx; Rwy = Ry; Rwz = Rz                     # COM separation of the image this site pair uses (virial)
            if wrap                                          # dx − L·round(dx/L) = _min_image(dx, L), bitwise
                kx = L * round(dx / L); ky = L * round(dy / L); kz = L * round(dz / L)
                dx -= kx; dy -= ky; dz -= kz; Rwx -= kx; Rwy -= ky; Rwz -= kz
            end
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
            elseif ff.coulomb === :rf && qn != 0.0 && qm != 0.0
                r = sqrt(r2)
                u += rf_pair_u(r, qn, qm, ff.rf)
                fr += rf_pair_fr(r, qn, qm, ff.rf)
            elseif ff.coulomb === :ewald && qn != 0.0 && qm != 0.0
                r = sqrt(r2)
                u += ke * qn * qm * erfc(α * r) / r                      # Ewald real space (reciprocal part: sys.recip)
                fr += ke * qn * qm * (erfc(α * r) / r + (2.0 * α / SQRT_PI) * exp(-α * α * r2))
            end
            # R_ij · r̂^{nm} (−du/dr) = (R·r)/r² · (−r du/dr)
            w += (Rwx * dx + Rwy * dy + Rwz * dz) / r2 * fr
        end
    end
    ff.cap_overlaps && u > ff.cap_min_u && return ff.cap_min_u, 0.0   # kT cap: capped pairs carry no virial
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
# bins of the per-rung dihedral histograms of a ladder run (`track_dihedrals` of init_rigid_ekmc_run; 10° each)
const DIH_BINS = 36

"Smallest number of molecules for which the pair-row loops are threaded."
const PARALLEL_MIN_MOL = 96

@inline _use_threads(N::Int) = Threads.nthreads() > 1 && N >= PARALLEL_MIN_MOL

"""
Fill row_u[j], row_w[j] = pair energy/virial of molecule k with every j ≠ k. With reciprocal-space Ewald (`sys.recip`)
and `recip = true` the Fourier pair term ψ_kj (k at its current pose, the others committed) is added to row_u (row_w
carries the real-space virial only); `recip = false` gives the real-space + LJ row (candidate rows keep the reciprocal
part separately, EwaldRecip.jl).
"""
function eval_row!(row_u::Vector{Float64}, row_w::Vector{Float64}, sys::RigidSystem,
                   ff::SiteForceField, k::Int; recip::Bool=true)
    N = sys.n_mol
    c = sys.cells
    if c isa SiteCells && _cells_for(c, ff) && site_cells_row!(row_u, row_w, c, sys, ff, k)
        n = -2                                         # row done by the site cell list
    else
        n = (c isa MolCells && _cells_for(c, ff)) ? cells_gather!(c, sys, k) : -1
    end
    if n == -2
    elseif n >= 0                                          # cell list: every molecule not gathered is beyond the pair range
        fill!(row_u, 0.0); fill!(row_w, 0.0)
        nbr = c.nbr
        if _use_threads(n)
            Threads.@threads :static for t in 1:n
                @inbounds j = nbr[t]
                u, w = pair_energy_virial(sys, ff, k, j)
                @inbounds row_u[j] = u; @inbounds row_w[j] = w
            end
        else
            @inbounds for t in 1:n
                j = nbr[t]
                u, w = pair_energy_virial(sys, ff, k, j)
                row_u[j] = u; row_w[j] = w
            end
        end
    elseif _use_threads(N)
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
    recip && _has_recip(sys, ff) && recip_row_add!(row_u, sys.recip, sys, k)
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
removal_energy(sys::RigidSystem, ff::SiteForceField, k::Int)::Float64 =
    _row_energy_sum(sys, ff, k) + (!_has_recip(sys, ff) ? 0.0 : recip_removal(sys.recip, sys, k))

# Per-task partial sums are indexed by the task number t (rows i ≡ t mod nt, which also balances the
# triangular loop), never by Threads.threadid(): with Julia ≥ 1.12 the default pool's thread ids start
# at 2 when an interactive thread exists, so threadid()-indexed arrays overflow.

"""
Total intermolecular pair energy (no tail, no Wolf constants). With reciprocal-space Ewald the structure factors are
rebuilt from the current positions and the Fourier and image terms added (Σ = `ewald_energy(...).total` + LJ).
"""
function total_pair_energy(sys::RigidSystem, ff::SiteForceField)::Float64
    !_has_recip(sys, ff) && return _total_pair_energy_real(sys, ff)
    recip_rebuild!(sys.recip, sys)
    return _total_pair_energy_real(sys, ff) + recip_energy_total(sys.recip)
end

function _total_pair_energy_real(sys::RigidSystem, ff::SiteForceField)::Float64
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

"""
    total_pair_energy_virial(sys, ff) -> (U, W)

Total pair energy (as `total_pair_energy`, Fourier and image terms included with reciprocal-space Ewald) and the
real-space molecular virial (the reciprocal virial is not implemented: `pressure` returns NaN with Ewald).
"""
function total_pair_energy_virial(sys::RigidSystem, ff::SiteForceField)
    U, W = _total_pair_energy_virial_real(sys, ff)
    !_has_recip(sys, ff) && return U, W
    recip_rebuild!(sys.recip, sys)
    return U + recip_energy_total(sys.recip), W
end

function _total_pair_energy_virial_real(sys::RigidSystem, ff::SiteForceField)
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
    ff.coulomb === :none && return 0.0
    E = 0.0
    for i in 1:sys.n_mol
        E += coulomb_molecule_constant(sys.templates[sys.tmpl[i]], ff)
    end
    return E
end

"""
    coulomb_molecule_constant(template, ff) -> E

Configuration-independent Coulomb terms of one rigid molecule in the force field's model: for `:wolf`
the self term and the intramolecular exclusion correction (see `wolf_intra_energy`), for `:rf` the
GROMACS excluded-pair corrections plus self terms (= −ke k_rf |μ|²/ε_r for a neutral molecule), 0 for
`:none`. Per molecule these are constants of the sampled Hamiltonian; they enter absolute energies and —
for `:rf`, where GROMACS carries them in a decoupling ΔG — the chemical potential of the species:
μ_ex,GROMACS-RF = kT ln⟨exp(βu_s)⟩ + E_const(s) + ΔU_tail(s).
"""
function coulomb_molecule_constant(t::RigidTemplate, ff::SiteForceField)::Float64
    E = 0.0
    if ff.coulomb === :wolf
        wp = ff.wolf
        for n in 1:t.n_sites
            E -= wp.ke * (erfc(wp.α * wp.rc) / (2 * wp.rc) + wp.α / SQRT_PI) * t.site_q[n]^2
            for m in n+1:t.n_sites
                (t.site_q[n] == 0.0 || t.site_q[m] == 0.0) && continue
                r = norm(t.sites_body[:, n] - t.sites_body[:, m])
                E -= wp.ke * t.site_q[n] * t.site_q[m] / r - wolf_pair_u(r, t.site_q[n], t.site_q[m], wp)
            end
        end
    elseif ff.coulomb === :rf
        p = ff.rf
        E += rf_self_energy(t.site_q, p)
        for n in 1:t.n_sites, m in n+1:t.n_sites
            (t.site_q[n] == 0.0 || t.site_q[m] == 0.0) && continue
            E += rf_excluded_u(norm(t.sites_body[:, n] - t.sites_body[:, m]), t.site_q[n], t.site_q[m], p)
        end
    end
    return E
end
const coulomb_constant_energy = wolf_constant_energy

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
    !_has_recip(sys, ff) || return NaN      # reciprocal-space virial not implemented yet (DESIGN §13): no pressure
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
Molecules are placed largest extent first (stable order otherwise): a large solute placed last into a
liquid-density box of solvent finds no free cavity (1-propanol in 500 SPC/E, seed 1, JOURNAL 54).
"""
function random_rigid_system!(sys::RigidSystem, ff::SiteForceField; max_tries::Int=10_000)
    L = sys.L
    order = sortperm(sys.ext; rev=true, alg=Base.Sort.DEFAULT_STABLE)
    for (k, i) in enumerate(order)
        placed = false
        for _ in 1:max_tries
            com = MVector{3,Float64}(rand(sys.rng) * L, rand(sys.rng) * L, rand(sys.rng) * L)
            set_molecule!(sys, i, com, uniform_random_quaternion(sys.rng))
            ok = true
            for j in view(order, 1:k-1)
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
# Energy minimisation of a starting configuration (steepest descent)
# ============================================================================

"Copy of `ff` with the overlap cap switched off (the bare site–site potential)."
uncapped(ff::SiteForceField) = SiteForceField(ff.n_types, ff.σ_mix, ff.ϵ_mix, ff.rcap2_mix, ff.ucap_mix, ff.rc,
                                              ff.rc2, ff.coulomb, ff.wolf, ff.use_lrc, false, ff.shift_lj,
                                              ff.u_rc_mix, ff.cutoff, ff.rf, Inf)

"""
    min_orientation_energy(sys, ffu, a, b, r, n_orient, rng) -> u_min

Lowest bare molecule-pair energy found over `n_orient` uniformly random orientations of molecules 1 and 2 of
`sys` (a two-molecule system in a large box) with site `a` of molecule 1 and site `b` of molecule 2 held at
distance `r` along x. A sampled minimum (an upper bound on the true minimum).
"""
function min_orientation_energy(sys::RigidSystem, ffu::SiteForceField, a::Int, b::Int, r::Float64,
                                n_orient::Int, rng::AbstractRNG)
    c = 0.5 * sys.L
    best = Inf
    for _ in 1:n_orient
        set_molecule!(sys, 1, MVector(c, c, c), uniform_random_quaternion(rng))
        set_molecule!(sys, 2, MVector(c, c, c), uniform_random_quaternion(rng))
        sa = sys.site_start[1] + a; sb = sys.site_start[2] + b
        set_molecule!(sys, 1, MVector(c - sys.site_off[1, sa], c - sys.site_off[2, sa], c - sys.site_off[3, sa]),
                      view(sys.quat, :, 1))
        set_molecule!(sys, 2, MVector(c + r - sys.site_off[1, sb], c - sys.site_off[2, sb], c - sys.site_off[3, sb]),
                      view(sys.quat, :, 2))
        best = min(best, pair_energy(sys, ffu, 1, 2))
    end
    return best
end

"""
    kT_overlap_cap(ff, templates, T; U_kT=100.0, n_orient=20_000, seed=20261013) -> (ff_capped, rstar)

Overlap cap for polyatomic molecules (DESIGN §9, user decision 2026-10-13: U_cap = 100 kT). Generalises the
rule of Kelly's thesis §3.5 — trigger radius r* where the bare potential equals the cap — to orientation-
dependent molecule pairs:
1. U_cap = `U_kT`·T (reduced energy) for every molecule pair.
2. For every pair of templates (A, B) and LJ sites a ∈ A, b ∈ B, r*_ab is the largest site–site distance at
   which every sampled orientation of the pair has bare (uncapped) energy ≥ U_cap: scan down from σ_ab in
   steps of 0.02 σ_ab to the first distance where the sampled minimum is ≥ U_cap, then bisect (12 steps).
   The trigger radius of LJ site types (t_a, t_b) is the minimum of r*_ab over all such pairs, so the
   property holds for every molecule pair. Templates are scanned in their reference geometry.
3. A molecule pair is U_cap if any LJ site pair is inside its trigger radius, otherwise min(u_bare, U_cap)
   (`cap_min_u`). Hence no core is lower than its rim (no Metropolis trap) and the capped Hamiltonian differs
   from the bare one only where u_bare ≥ U_cap (Boltzmann weight ≤ e^{−U_kT}); sampling error in r* is
   harmless for the same reason.
Charge-collapse check: an LJ-less charged site h (bonded at d_h to the nearest LJ site X of its molecule) can
only reach an opposite charge at site Y of another molecule if |X − Y| ≤ d_h (+ d_h' if Y is LJ-less too, with
its own host); every such distance must be inside the trigger radius, else an error is thrown.
Returns the capped force field and the matrix of trigger radii (reduced units).
"""
function kT_overlap_cap(ff::SiteForceField, templates::Vector{RigidTemplate}, T::Float64; U_kT::Float64=100.0,
                        n_orient::Int=20_000, seed::Int=20261013)
    U = U_kT * T
    ffu = uncapped(ff)
    nt = ff.n_types
    rstar = fill(Inf, nt, nt)
    rng = Xoshiro(seed)
    for A in eachindex(templates), B in A:length(templates)
        tA = templates[A]; tB = templates[B]
        ext = tA.extent + tB.extent
        sys = RigidSystem([tA, tB], [1, 2], 4.0 * (ext + 3.0 * maximum(ff.σ_mix)) + 2.0 * ff.rc, 1.0 / T, Xoshiro(seed))
        for a in 1:tA.n_sites, b in 1:tB.n_sites
            ta = tA.site_type[a]; tb = tB.site_type[b]
            (ta > 0 && tb > 0) || continue
            σ = ff.σ_mix[ta, tb]
            r_hi = σ
            f(r) = min_orientation_energy(sys, ffu, a, b, r, n_orient, rng)
            f(r_hi) < U || error("kT_overlap_cap: pair ($(tA.name) site $a, $(tB.name) site $b) is above U_cap at r = σ_ab")
            r_lo = r_hi
            while true
                r_lo -= 0.02 * σ
                r_lo > 0.1 * σ || error("kT_overlap_cap: no distance with energy ≥ U_cap for ($(tA.name) $a, $(tB.name) $b)")
                f(r_lo) >= U && break
                r_hi = r_lo
            end
            for _ in 1:12
                rm = 0.5 * (r_lo + r_hi)
                f(rm) >= U ? (r_lo = rm) : (r_hi = rm)
            end
            rstar[ta, tb] = min(rstar[ta, tb], r_lo); rstar[tb, ta] = rstar[ta, tb]
        end
    end
    for t1 in 1:nt, t2 in 1:nt                       # type pairs that never meet: keep the legacy radius
        isfinite(rstar[t1, t2]) || (rstar[t1, t2] = sqrt(ff.rcap2_mix[t1, t2]))
    end
    # charge-collapse coverage: (host X of h, d_h) for every charged site (LJ sites are their own host, d = 0)
    function hosts(t::RigidTemplate)
        out = Tuple{Int,Float64,Float64}[]          # (host LJ type, d_h, q)
        for h in 1:t.n_sites
            t.site_q[h] == 0.0 && continue
            if t.site_type[h] > 0
                push!(out, (t.site_type[h], 0.0, t.site_q[h])); continue
            end
            dbest = Inf; tbest = 0
            for x in 1:t.n_sites
                t.site_type[x] > 0 || continue
                d = sqrt(sum(abs2, t.sites_body[:, h] .- t.sites_body[:, x]))
                d < dbest && (dbest = d; tbest = t.site_type[x])
            end
            tbest > 0 || error("kT_overlap_cap: charged site $h of $(t.name) has no LJ site in its molecule")
            push!(out, (tbest, dbest, t.site_q[h]))
        end
        return out
    end
    for A in eachindex(templates), B in A:length(templates)
        for (x, dh, qh) in hosts(templates[A]), (y, dy, qy) in hosts(templates[B])
            (qh * qy < 0.0 && (dh > 0.0 || dy > 0.0)) || continue
            dh + dy < rstar[x, y] ||
                error("kT_overlap_cap: opposite charges of $(templates[A].name)/$(templates[B].name) can collapse outside " *
                      "the trigger radius (|X−Y| ≤ $(dh + dy) ≥ r* = $(rstar[x, y])); use a smaller U_kT")
        end
    end
    ffc = SiteForceField(ff.n_types, ff.σ_mix, ff.ϵ_mix, rstar .^ 2, fill(U, nt, nt), ff.rc, ff.rc2, ff.coulomb,
                         ff.wolf, ff.use_lrc, true, ff.shift_lj, ff.u_rc_mix, ff.cutoff, ff.rf, U)
    return ffc, rstar
end

"""
    rigid_forces_torques!(F, τ, sys, ff) -> U_pair

Pair energy and, per molecule, the total force on the COM (`F`, 3 × n_mol) and the torque about the COM
(`τ`, lab frame, 3 × n_mol) from the site–site LJ and Coulomb (Wolf/RF) terms, using the same cutoff rules
and pair derivative (−r du/dr) as `pair_energy_virial`. Honours `ff.cap_overlaps` (a capped pair contributes
its cap energy and no force), so call it with `uncapped(ff)` for the bare potential. Intramolecular terms
of flexible molecules are not included (conformations are held fixed).
"""
function rigid_forces_torques!(F::Matrix{Float64}, τ::Matrix{Float64}, sys::RigidSystem, ff::SiteForceField)
    fill!(F, 0.0); fill!(τ, 0.0)
    N = sys.n_mol; L = sys.L
    molcut = ff.cutoff === :molecule
    α = ff.wolf.α; ke = ff.wolf.ke
    U = 0.0
    @inbounds for i in 1:N-1, j in i+1:N
        Rx = _min_image(sys.com[1, j] - sys.com[1, i], L)
        Ry = _min_image(sys.com[2, j] - sys.com[2, i], L)
        Rz = _min_image(sys.com[3, j] - sys.com[3, i], L)
        R2 = Rx * Rx + Ry * Ry + Rz * Rz
        if molcut
            R2 >= ff.rc2 && continue
        else
            rmax = ff.rc + sys.ext[i] + sys.ext[j]
            R2 >= rmax * rmax && continue
        end
        ti = sys.templates[sys.tmpl[i]]; tj = sys.templates[sys.tmpl[j]]
        si = sys.site_start[i]; sj = sys.site_start[j]
        wrap = _site_images(ff, L) && _pair_wraps(Rx, Ry, Rz, sys.ext[i] + sys.ext[j], L)
        # capped pair (geometric trigger, or bare energy above the kT cap)? then its cap energy and no
        # force — exactly as pair_energy
        capped = false
        if ff.cap_overlaps
            for n in 1:ti.n_sites, m in 1:tj.n_sites
                tyn = ti.site_type[n]; tym = tj.site_type[m]
                (tyn > 0 && tym > 0) || continue
                dx = Rx + sys.site_off[1, sj + m] - sys.site_off[1, si + n]
                dy = Ry + sys.site_off[2, sj + m] - sys.site_off[2, si + n]
                dz = Rz + sys.site_off[3, sj + m] - sys.site_off[3, si + n]
                if wrap
                    dx = _min_image(dx, L); dy = _min_image(dy, L); dz = _min_image(dz, L)
                end
                r2 = dx * dx + dy * dy + dz * dz
                (molcut || r2 < ff.rc2) || continue
                if r2 < ff.rcap2_mix[tyn, tym]
                    capped = true; U += ff.ucap_mix[tyn, tym]; break
                end
            end
            if !capped && isfinite(ff.cap_min_u) && pair_energy(sys, ff, i, j) >= ff.cap_min_u
                capped = true; U += ff.cap_min_u
            end
        end
        capped && continue
        for n in 1:ti.n_sites
            ox, oy, oz = sys.site_off[1, si + n], sys.site_off[2, si + n], sys.site_off[3, si + n]
            tyn = ti.site_type[n]; qn = ti.site_q[n]
            for m in 1:tj.n_sites
                px, py, pz = sys.site_off[1, sj + m], sys.site_off[2, sj + m], sys.site_off[3, sj + m]
                dx = Rx + px - ox; dy = Ry + py - oy; dz = Rz + pz - oz
                if wrap
                    dx = _min_image(dx, L); dy = _min_image(dy, L); dz = _min_image(dz, L)
                end
                r2 = dx * dx + dy * dy + dz * dz
                (molcut || r2 < ff.rc2) || continue
                tym = tj.site_type[m]; qm = tj.site_q[m]
                fr = 0.0
                if tyn > 0 && tym > 0
                    σ2 = ff.σ_mix[tyn, tym]^2; s6 = (σ2 / r2)^3; ϵ = ff.ϵ_mix[tyn, tym]
                    U += 4.0 * ϵ * (s6 * s6 - s6)
                    ff.shift_lj && (U -= ff.u_rc_mix[tyn, tym])
                    fr += 24.0 * ϵ * (2.0 * s6 * s6 - s6)
                end
                if ff.coulomb === :wolf && qn != 0.0 && qm != 0.0
                    r = sqrt(r2)
                    U += molcut ? wolf_pair_u_nocut(r, qn, qm, ff.wolf) : wolf_pair_u(r, qn, qm, ff.wolf)
                    fr += ke * qn * qm * (erfc(α * r) / r + (2.0 * α / SQRT_PI) * exp(-α * α * r2))
                elseif ff.coulomb === :rf && qn != 0.0 && qm != 0.0
                    r = sqrt(r2)
                    U += rf_pair_u(r, qn, qm, ff.rf)
                    fr += rf_pair_fr(r, qn, qm, ff.rf)
                end
                # force on site m of j: (fr/r²) d (d points from i's site to j's site); opposite on i
                fx = fr / r2 * dx; fy = fr / r2 * dy; fz = fr / r2 * dz
                F[1, j] += fx; F[2, j] += fy; F[3, j] += fz
                F[1, i] -= fx; F[2, i] -= fy; F[3, i] -= fz
                τ[1, j] += py * fz - pz * fy; τ[2, j] += pz * fx - px * fz; τ[3, j] += px * fy - py * fx
                τ[1, i] -= oy * fz - oz * fy; τ[2, i] -= oz * fx - ox * fz; τ[3, i] -= ox * fy - oy * fx
            end
        end
    end
    return U
end

"Rotation quaternion by angle θ about the unit lab-frame axis (ax, ay, az)."
@inline function _axis_angle_quaternion(ax::Float64, ay::Float64, az::Float64, θ::Float64)
    h = 0.5 * θ; s = sin(h)
    return MVector{4,Float64}(cos(h), s * ax, s * ay, s * az)
end

"""
    minimize_rigid!(sys, ff; nsteps=1000, h0=0.01, ftol=0.0) -> (U_start, U_end, steps, gmax, n_capped)

Steepest-descent energy minimisation of the rigid-body coordinates (COM translations and rotations; flexible
conformations held fixed), with the adaptive step rule of the GROMACS steepest-descent minimiser (GROMACS
reference manual, "Energy minimization — steepest descent"): every molecule moves along its generalised force
scaled so that the largest site displacement is h; a step that lowers the energy is accepted and h ← 1.2 h,
otherwise it is undone and h ← 0.2 h. Rotations: the torque τ is converted to a force-equivalent |τ|/ext at the
molecular extent ext and the rotation angle is h·|τ|/(ext²·g_max) about τ̂, so no site moves more than h.

The minimisation runs on the UNCAPPED potential (`uncapped(ff)`): inside the overlap cap the capped energy is
flat (zero force) and, for LJ with cap 100 ε at r = 0.75 σ, lower than the bare LJ just outside it
(U_LJ(0.75 σ) = 104 ε), so the capped surface would let a pair fall into the core and stay there — the defect
that trapped Metropolis runs from `random_rigid_system!` (JOURNAL entry 24). Use after `random_rigid_system!`
and before Metropolis equilibration. Stops after `nsteps`, when g_max ≤ `ftol`, or when h < 1e-12.
Returns the uncapped pair energies before/after, the steps taken, the final largest generalised force and
the number of capped molecule pairs left (by `pair_energy` with the real `ff`).
"""
function minimize_rigid!(sys::RigidSystem, ff::SiteForceField; nsteps::Int=1000, h0::Float64=0.01, ftol::Float64=0.0)
    N = sys.n_mol
    ffu = uncapped(ff)
    F = zeros(3, N); τ = zeros(3, N); F2 = zeros(3, N); τ2 = zeros(3, N)
    com0 = copy(sys.com); q0 = copy(sys.quat)
    U = rigid_forces_torques!(F, τ, sys, ffu)
    U_start = U
    h = h0; steps = 0; gmax = Inf
    while steps < nsteps
        gmax = 0.0
        @inbounds for i in 1:N
            gmax = max(gmax, sqrt(F[1, i]^2 + F[2, i]^2 + F[3, i]^2))
            e = sys.ext[i]
            e > 0.0 && (gmax = max(gmax, sqrt(τ[1, i]^2 + τ[2, i]^2 + τ[3, i]^2) / e))
        end
        (gmax <= ftol || h < 1e-12 || !isfinite(gmax)) && break
        steps += 1
        copyto!(com0, sys.com); copyto!(q0, sys.quat)
        @inbounds for i in 1:N
            com = MVector{3,Float64}(sys.com[1, i] + h * F[1, i] / gmax, sys.com[2, i] + h * F[2, i] / gmax,
                                     sys.com[3, i] + h * F[3, i] / gmax)
            q = MVector{4,Float64}(sys.quat[1, i], sys.quat[2, i], sys.quat[3, i], sys.quat[4, i])
            e = sys.ext[i]; tn = sqrt(τ[1, i]^2 + τ[2, i]^2 + τ[3, i]^2)
            if e > 0.0 && tn > 0.0
                θ = h * tn / (e * e * gmax)
                q = quat_mul(_axis_angle_quaternion(τ[1, i] / tn, τ[2, i] / tn, τ[3, i] / tn, θ), q)
                q = q ./ sqrt(sum(abs2, q))
            end
            set_molecule!(sys, i, com, q)
        end
        Un = rigid_forces_torques!(F2, τ2, sys, ffu)
        if Un < U
            U = Un; F, F2 = F2, F; τ, τ2 = τ2, τ
            h *= 1.2
        else
            @inbounds for i in 1:N
                set_molecule!(sys, i, view(com0, :, i), view(q0, :, i))
            end
            h *= 0.2
        end
    end
    ncap = 0
    caps = minimum(ff.ucap_mix) * 0.999
    for i in 1:N-1, j in i+1:N
        pair_energy(sys, ff, i, j) >= caps && (ncap += 1)
    end
    return (U_start=U_start, U_end=U, steps=steps, gmax=gmax, n_capped=ncap)
end

# ============================================================================
# Metropolis MC
# ============================================================================

"Hamilton product q1 ⊗ q2 (both (w, x, y, z))."
@inline function quat_mul(q1::AbstractVector{Float64}, q2::AbstractVector{Float64})
    w1, x1, y1, z1 = q1[1], q1[2], q1[3], q1[4]
    w2, x2, y2, z2 = q2[1], q2[2], q2[3], q2[4]
    return SVector{4,Float64}(w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
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
    site_att::Int; site_acc::Int       # flexible molecules: single-site displacements
    swap_att::Int; swap_acc::Int       # flexible molecules: conformer exchanges with the reservoir
    tors_att::Int; tors_acc::Int       # flexible molecules: rigid-fragment torsion rotations (JOURNAL 80)
end
RigidMCStats() = RigidMCStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

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
        molecule_committed!(sys, i)
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
        molecule_committed!(sys, i)
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
        molecule_committed!(sys, i)
        return true
    end
    set_molecule!(sys, i, com, q_old)
    return false
end

"""
    mc_site_move!(sys, ff, i, δ, st) -> accepted

Single-site Cartesian displacement of a random site of flexible molecule i (uniform in a cube of half-side
δ), accepted on ΔU_intra + ΔU_inter (Frenkel & Smit §13.2; exact for the Cartesian measure of harmonic
bonds). The COM moves with the site and the body is re-centred, so the molecule-based cutoff and the pair
pre-filter stay consistent.
"""
function mc_site_move!(sys::RigidSystem, ff::SiteForceField, i::Int, δ::Float64, st::RigidMCStats)::Bool
    t = sys.templates[sys.tmpl[i]]
    isflexible(t) || return false
    st.site_att += 1
    u_old = removal_energy(sys, ff, i) + flex_intra_energy(t, ff, sys.body[i])
    com_old = MVector{3,Float64}(sys.com[1, i], sys.com[2, i], sys.com[3, i])
    q_old = MVector{4,Float64}(sys.quat[1, i], sys.quat[2, i], sys.quat[3, i], sys.quat[4, i])
    body_old = copy(sys.body[i]); ext_old = sys.ext[i]
    X = molecule_sites(sys, i)
    k = rand(sys.rng, 1:t.n_sites)
    X[1, k] += (2rand(sys.rng) - 1) * δ; X[2, k] += (2rand(sys.rng) - 1) * δ; X[3, k] += (2rand(sys.rng) - 1) * δ
    c = X * t.site_mass ./ sum(t.site_mass)
    set_conformation!(sys, i, X)                       # body = X − c (lab-oriented), quat = identity
    set_molecule!(sys, i, c, MVector{4,Float64}(1.0, 0.0, 0.0, 0.0))
    ΔU = removal_energy(sys, ff, i) + flex_intra_energy(t, ff, sys.body[i]) - u_old
    if ΔU <= 0.0 || rand(sys.rng) < exp(-sys.β * ΔU)
        st.site_acc += 1
        molecule_committed!(sys, i)
        return true
    end
    _set_body!(sys, i, body_old); sys.ext[i] = ext_old
    set_molecule!(sys, i, com_old, q_old)
    return false
end

"""
    mc_torsion_move!(sys, ff, i, φmax, st) -> accepted

Metropolis torsion move of flexible molecule i (yardstick for the rejection-free LM_D slot, JOURNAL 80): a random
rotatable bond (`rotatable_bonds`), the fragment rotated by φ uniform in [−φmax, φmax] in the body frame, body re-centred
and the COM shifted at fixed orientation (volume-preserving, symmetric), accepted on ΔU_intra + ΔU_inter. Single-site
moves alone do not cross heavy-atom torsion barriers (JOURNAL 79).
"""
function mc_torsion_move!(sys::RigidSystem, ff::SiteForceField, i::Int, φmax::Float64, st::RigidMCStats)::Bool
    t = sys.templates[sys.tmpl[i]]
    isflexible(t) || return false
    rb = rotatable_bonds(t)
    isempty(rb) && return false
    st.tors_att += 1
    u_old = removal_energy(sys, ff, i) + flex_intra_energy(t, ff, sys.body[i])
    com_old = MVector{3,Float64}(sys.com[1, i], sys.com[2, i], sys.com[3, i])
    q_old = MVector{4,Float64}(sys.quat[1, i], sys.quat[2, i], sys.quat[3, i], sys.quat[4, i])
    body_old = copy(sys.body[i]); ext_old = sys.ext[i]
    Y = copy(sys.body[i])
    (a, b, mov) = rb[rand(sys.rng, 1:length(rb))]
    rotate_fragment!(Y, a, b, mov, (2rand(sys.rng) - 1) * φmax)
    c = (Y * t.site_mass) ./ sum(t.site_mass); Y .-= c
    sh = quaternion_to_rotation_matrix(q_old) * c
    _set_body!(sys, i, Y)
    set_molecule!(sys, i, MVector{3,Float64}(com_old[1] + sh[1], com_old[2] + sh[2], com_old[3] + sh[3]), q_old)
    ΔU = removal_energy(sys, ff, i) + flex_intra_energy(t, ff, sys.body[i]) - u_old
    if ΔU <= 0.0 || rand(sys.rng) < exp(-sys.β * ΔU)
        st.tors_acc += 1
        molecule_committed!(sys, i)
        return true
    end
    _set_body!(sys, i, body_old); sys.ext[i] = ext_old
    set_molecule!(sys, i, com_old, q_old)
    return false
end

"""
    mc_conformer_swap!(sys, ff, i, res, st) -> accepted

Replace the conformation of flexible molecule i by a reservoir conformer in a uniform random orientation
(COM fixed). The proposal density p_id(c′) ∝ exp(−βU_intra(c′)) cancels the intramolecular Boltzmann factors,
so the acceptance is min(1, exp(−βΔU_inter)) (the ideal-gas conformer exchange, Frenkel & Smit §13.3).
"""
function mc_conformer_swap!(sys::RigidSystem, ff::SiteForceField, i::Int, res::ConformerReservoir, st::RigidMCStats)::Bool
    t = sys.templates[sys.tmpl[i]]
    isflexible(t) || return false
    st.swap_att += 1
    u_old = removal_energy(sys, ff, i)
    com = MVector{3,Float64}(sys.com[1, i], sys.com[2, i], sys.com[3, i])
    q_old = MVector{4,Float64}(sys.quat[1, i], sys.quat[2, i], sys.quat[3, i], sys.quat[4, i])
    body_old = copy(sys.body[i]); ext_old = sys.ext[i]
    _set_body!(sys, i, draw_conformer(res, sys.rng))
    set_molecule!(sys, i, com, uniform_random_quaternion(sys.rng))
    ΔU = removal_energy(sys, ff, i) - u_old
    if ΔU <= 0.0 || rand(sys.rng) < exp(-sys.β * ΔU)
        st.swap_acc += 1
        molecule_committed!(sys, i)
        return true
    end
    _set_body!(sys, i, body_old); sys.ext[i] = ext_old
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
    !_has_recip(sys, ff) || recip_rebuild!(sys.recip, sys)      # total_energy rebuilt it at the trial volume
    sys.cells === nothing || cells_rebuild!(sys.cells, sys)
    return false, U_current
end

"""
    widom_rigid(sys, ff, template_idx; ninsert) -> mean of exp(−β ΔU)

Widom insertion of a rigid molecule at a uniform random position and orientation (Frenkel & Smit
§7.2.1); the excess chemical potential is −kT ln of the returned average (intermolecular part; the
tail contribution of the inserted molecule, if any, is added by `widom_lrc_shift`).
"""
function widom_rigid(sys::RigidSystem, ff::SiteForceField, template_idx::Int; ninsert::Int=1000,
                     reservoir::Union{Nothing,ConformerReservoir}=nothing)
    !_has_recip(sys, ff) || error("widom_rigid: not implemented for reciprocal-space Ewald (use the eKMC/ladder estimators)")
    # temporarily append a ghost molecule
    ghost = RigidSystem(sys.templates, vcat(sys.tmpl, template_idx), sys.L, sys.β, sys.rng)
    ghost.com[:, 1:sys.n_mol] .= sys.com
    ghost.quat[:, 1:sys.n_mol] .= sys.quat
    ghost.site_off[:, 1:sys.n_sites_total] .= sys.site_off
    for i in 1:sys.n_mol
        ghost.body[i] = sys.body[i]; ghost.ext[i] = sys.ext[i]
    end
    g = ghost.n_mol
    flex = isflexible(sys.templates[template_idx])
    flex && reservoir === nothing && error("widom_rigid: a flexible template needs a ConformerReservoir (ideal-gas conformers)")
    acc = 0.0
    L = sys.L
    for _ in 1:ninsert
        flex && _set_body!(ghost, g, draw_conformer(reservoir, sys.rng))
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
`inverse_widom_molecule=i` accumulates exp(βu_i) of molecule i at every sample (`inv_widom`): the
particle-removal estimator exp(βμ_ex) = ⟨exp(βu_i)⟩ (Shing & Gubbins, Mol. Phys. 46 (1982) 1109),
i.e. the Metropolis counterpart of the eKMC per-species ⟨R_s/N_s⟩, with the same `_mobility` clamp.
`R_every=k` evaluates the eKMC relocation rate R = Σ_i exp(βu_i) of every k-th sample (N removal energies,
one sweep's cost) into `R`: the Metropolis estimate of the *relaxed-state* R, the right scale for the
hybrid's MC-kernel rate λ (λ = fac·⟨R⟩ gives a relocation fraction ≈ 1/(1 + fac) in relaxed states). A pure
eKMC pilot of 2e4 events estimates ⟨R⟩ as M/t_total and can miss the relaxed states entirely in a dense
liquid (2026-10-12: λ from the pilot ranged over 1e-2 … 4e3 between seeds of the same system, and at λ ≫ R the
solute is never relocated, so its μ degenerates to the Metropolis removal estimator).
`torsion_φ > 0` adds, per sweep and flexible molecule, one `mc_torsion_move!` per rotatable bond (half-angle torsion_φ).
"""
function run_rigid_mc!(sys::RigidSystem, ff::SiteForceField; sweeps::Int, δ::Float64, δθ::Float64,
                       Pext::Union{Nothing,Float64}=nothing, max_dlnV::Float64=0.1,
                       vol_every::Int=1, sample_every::Int=1, block_size::Int=10,
                       widom_template::Int=0, widom_per_sample::Int=0, tail_route::Symbol=:virial,
                       inverse_widom_molecule::Int=0, R_every::Int=0,
                       reservoirs::Union{Nothing,Vector}=nothing, δ_site::Float64=0.03, torsion_φ::Float64=0.0)
    st = RigidMCStats()
    N = sys.n_mol
    U = total_energy(sys, ff)
    flex_mols = [i for i in 1:N if isflexible(sys.templates[sys.tmpl[i]])]
    ba_Ui = BlockAverager(block_size)                                      # intramolecular energy per molecule (flexible)
    ba_U = BlockAverager(block_size); ba_P = BlockAverager(block_size)
    ba_rho = BlockAverager(block_size); ba_w = BlockAverager(block_size)
    ba_V = BlockAverager(block_size); ba_wV = BlockAverager(block_size)   # NPT Widom: <V e^{-βΔU}>/<V>
    ba_iw = BlockAverager(block_size)                                      # inverse Widom of one molecule
    ba_R = BlockAverager(max(1, block_size ÷ max(R_every, 1)))             # relocation rate R = Σ exp(βu_i) of relaxed states
    ba_Rs = [BlockAverager(max(1, block_size ÷ max(R_every, 1))) for _ in sys.templates]   # per template
    Rs = zeros(length(sys.templates))
    n_samples = 0
    for sweep in 1:sweeps
        for _ in 1:N
            i = rand(sys.rng, 1:N)
            mc_translate!(sys, ff, i, δ, st)
            i = rand(sys.rng, 1:N)
            mc_rotate!(sys, ff, i, δθ, st)
        end
        # flexible molecules: one displacement attempt per site and one conformer exchange per molecule
        for i in flex_mols
            for _ in 1:sys.templates[sys.tmpl[i]].n_sites
                mc_site_move!(sys, ff, i, δ_site, st)
            end
            if reservoirs !== nothing && reservoirs[sys.tmpl[i]] !== nothing
                mc_conformer_swap!(sys, ff, i, reservoirs[sys.tmpl[i]], st)
            end
            if torsion_φ > 0.0                                             # torsion moves (JOURNAL 80)
                for _ in 1:length(rotatable_bonds(sys.templates[sys.tmpl[i]]))
                    mc_torsion_move!(sys, ff, i, torsion_φ, st)
                end
            end
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
            isempty(flex_mols) || push!(ba_Ui, total_intra_energy(sys, ff) / length(flex_mols))
            if widom_template > 0 && widom_per_sample > 0
                w = widom_rigid(sys, ff, widom_template; ninsert=widom_per_sample,
                                reservoir=reservoirs === nothing ? nothing : reservoirs[widom_template])
                push!(ba_w, w)
                push!(ba_wV, V * w)
            end
            if inverse_widom_molecule > 0
                push!(ba_iw, _mobility(sys.β, removal_energy(sys, ff, inverse_widom_molecule)))
            end
            n_samples += 1
            if R_every > 0 && n_samples % R_every == 0
                fill!(Rs, 0.0)
                for i in 1:N
                    Rs[sys.tmpl[i]] += _mobility(sys.β, removal_energy(sys, ff, i))
                end
                push!(ba_R, sum(Rs))
                for s in eachindex(Rs)
                    push!(ba_Rs[s], Rs[s])
                end
            end
        end
    end
    return (U=ba_U, P=ba_P, rho=ba_rho, V=ba_V, widom=ba_w, widomV=ba_wV, stats=st, inv_widom=ba_iw, R=ba_R,
            R_species=ba_Rs, U_intra=ba_Ui)
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
    c::Vector{Float64}          # per-molecule rate scale: m_i = c_i exp(βu_i) (species importance sampling)
    # fractional molecule (coupling ladder, DESIGN §10.4): pair energies/virials of `frac` with every other molecule
    # are stored scaled by lam[frac] in pair_u/pair_w (the λ-Hamiltonian); bare_u/bare_w keep the unscaled row,
    # frac_u/frac_w their sums. frac = 0: no fractional molecule (stock bookkeeping, no overhead).
    lam::Vector{Float64}
    frac::Int
    bare_u::Vector{Float64}
    bare_w::Vector{Float64}
    frac_u::Float64
    frac_w::Float64
end

"""
    init_rigid_ekmc(sys, ff; debug_check_every=0, rate_scale=nothing)

`rate_scale[i]` multiplies molecule i's relocation rate: m_i = c_i exp(βu_i). A constant factor per
molecule leaves detailed balance intact (π(x) c_i e^{βu_i(x)}/V = c_i e^{−βU_rest}/V is still symmetric in
x ↔ x′) and the time-weighted averages canonical; only the *frequency* with which species are relocated
changes. The μ observables use exp(βu_i) = m_i/c_i, so they are unchanged. Used to give a dilute solute
the same relocation frequency as the whole solvent (JOURNAL 2026-10-12 entry 19).
"""
function init_rigid_ekmc(sys::RigidSystem, ff::SiteForceField; debug_check_every::Int=0,
                         rate_scale::Union{Nothing,AbstractVector{Float64}}=nothing)
    N = sys.n_mol
    c = rate_scale === nothing ? ones(N) : Vector{Float64}(rate_scale)
    length(c) == N && all(c .> 0) || error("init_rigid_ekmc: rate_scale must have one positive entry per molecule")
    ek = RigidEKMCState(zeros(N, N), zeros(N, N), zeros(N), zeros(N), zeros(N), 0.0, debug_check_every, 0,
                        zeros(N), zeros(N), 0, 0, 0, c, ones(N), 0, zeros(N), zeros(N), 0.0, 0.0)
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
    # fractional molecule (coupling ladder): the trial is a Metropolis step on H_k, so its pairs are scaled by λ; the
    # bare values are kept aside and committed only on acceptance
    bare_row = nothing; bare_s = 0.0; bare_sw = 0.0
    if ek.frac > 0
        s = ek.frac; λ = ek.lam[s]
        if i == s
            bare_row = (copy(ek.row_u), copy(ek.row_w))
            ek.row_u .*= λ; ek.row_w .*= λ
        else
            bare_s = ek.row_u[s]; bare_sw = ek.row_w[s]
            ek.row_u[s] *= λ; ek.row_w[s] *= λ
        end
    end
    ΔU = 0.0
    @inbounds for j in 1:N
        j == i && continue
        ΔU += ek.row_u[j] - ek.pair_u[i, j]
    end
    e_new = 0.0
    if _has_recip(sys, ff)                            # image energy of i at the trial pose (λ-scaled for the fractional molecule)
        e_new = recip_image_now(sys.recip, sys, i)
        ΔU += (i == ek.frac ? ek.lam[i] : 1.0) * (e_new - sys.recip.eimg[i])
    end
    if ΔU <= 0.0 || rand(sys.rng) < exp(-sys.β * ΔU)
        β = sys.β
        molecule_committed!(sys, i)
        if ek.frac > 0
            s = ek.frac
            if i == s
                ek.bare_u .= bare_row[1]; ek.bare_w .= bare_row[2]
                ek.frac_u = sum(ek.bare_u) + e_new; ek.frac_w = sum(ek.bare_w)
            else
                ek.frac_u += bare_s - ek.bare_u[i]; ek.frac_w += bare_sw - ek.bare_w[i]
                ek.bare_u[i] = bare_s; ek.bare_w[i] = bare_sw
            end
        end
        phii = 0.0; wi = 0.0
        @inbounds for j in 1:N
            j == i && continue
            du = ek.row_u[j] - ek.pair_u[i, j]; dw = ek.row_w[j] - ek.pair_w[i, j]
            ek.pair_u[i, j] = ek.row_u[j]; ek.pair_u[j, i] = ek.row_u[j]
            ek.pair_w[i, j] = ek.row_w[j]; ek.pair_w[j, i] = ek.row_w[j]
            if du != 0.0
                ek.phi[j] += du
                ek.m[j] = _mob(ek, β, j)
            end
            ek.wsum[j] += dw
            phii += ek.row_u[j]; wi += ek.row_w[j]
        end
        !_has_recip(sys, ff) || (phii += (i == ek.frac ? ek.lam[i] : 1.0) * e_new)
        ek.phi[i] = phii; ek.wsum[i] = wi
        ek.m[i] = _mob(ek, β, i)
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

"""
    ekmc_mc_sweep_local!(ek, sys, ff, δ, δθ, r_loc) -> n_shell

Local Metropolis sweep around the fractional molecule s (DESIGN §10.8, R2c): one translation and one rotation trial
per molecule for the n_shell molecules whose COM lies within r_loc of the COM of s (s itself fixed during the sweep),
each selected uniformly from that fixed set, and a trial that would carry the molecule OUT of the sphere is rejected
(so the selection set is the same before and after every accepted move: symmetric proposal, detailed balance with H_k).
The shell changes only through relocations and the moves of s. Then one translation and one rotation trial of s itself
(ordinary symmetric trials). Cost ≈ 2(n_shell + 1) row evaluations instead of 2N.
"""
function ekmc_mc_sweep_local!(ek::RigidEKMCState, sys::RigidSystem, ff::SiteForceField, δ::Float64, δθ::Float64, r_loc::Float64)
    s = ek.frac
    s > 0 || error("ekmc_mc_sweep_local!: no fractional molecule")
    N = sys.n_mol; L = sys.L; r2 = r_loc * r_loc
    cs = (sys.com[1, s], sys.com[2, s], sys.com[3, s])
    inside(i) = begin
        dx = _min_image(sys.com[1, i] - cs[1], L); dy = _min_image(sys.com[2, i] - cs[2], L); dz = _min_image(sys.com[3, i] - cs[3], L)
        dx * dx + dy * dy + dz * dz < r2
    end
    shell = [i for i in 1:N if i != s && inside(i)]
    n = length(shell)
    for _ in 1:n
        for rot in (false, true)
            i = shell[rand(sys.rng, 1:n)]
            com_old = MVector{3,Float64}(sys.com[1, i], sys.com[2, i], sys.com[3, i])
            q_old = MVector{4,Float64}(sys.quat[1, i], sys.quat[2, i], sys.quat[3, i], sys.quat[4, i])
            if ekmc_mc_trial!(ek, sys, ff, i, δ, δθ; rotate=rot) && !inside(i)
                # the accepted move left the sphere: undo it (rejection keeps the selection set fixed)
                set_molecule!(sys, i, com_old, q_old)
                update_rigid_ekmc_after_move!(ek, sys, ff, i)
                ek.mc_accepted -= 1
            end
        end
    end
    if ek.lam[s] > 0.0
        ekmc_mc_trial!(ek, sys, ff, s, δ, δθ; rotate=false)
        ekmc_mc_trial!(ek, sys, ff, s, δ, δθ; rotate=true)
    end
    ek.R = sum(ek.m)
    ek.mc_events += 1
    return n
end

"Total molecular virial from the maintained pair virials."
ekmc_virial(ek::RigidEKMCState) = 0.5 * sum(ek.wsum)
# mobility of molecule i: an absent fractional molecule (λ = 0) is never a mover
@inline _mob(ek::RigidEKMCState, β::Float64, i::Int) = (i == ek.frac && ek.lam[i] == 0.0) ? 0.0 : ek.c[i] * _mobility(β, ek.phi[i])

"Recompute all pair energies and virials, φ, mobilities and R from scratch."
function rebuild_rigid_ekmc!(ek::RigidEKMCState, sys::RigidSystem, ff::SiteForceField)
    N = sys.n_mol
    !_has_recip(sys, ff) || recip_rebuild!(sys.recip, sys)
    sys.cells === nothing || cells_rebuild!(sys.cells, sys)
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
    if _has_recip(sys, ff)                           # Fourier pair terms ψ_ij = Σ_k c_k Re[S_i^* S_j]: one product Sᴴ diag(c) S
        Rr = sys.recip
        # on the backend with :double; with :mixed the rebuild is the exact Float64 resynchronisation (host)
        # Re[Sᴴ diag(c) S] = Srᵀ diag(c ⊗ (1,1)) Sr with Sr the real (interleaved Re, Im) view: one real product
        Ψ = if Rr.precision === :double
            Array(transpose(Rr.Sdr) * (repeat(Rr.cd; inner=2) .* Rr.Sdr))
        else
            Sr = _realview(Rr.S); transpose(Sr) * (repeat(Rr.c; inner=2) .* Sr)
        end
        @inbounds for j in 1:N, i in 1:N
            i == j || (ek.pair_u[i, j] += Ψ[i, j])
        end
        @inbounds for i in 1:N
            ek.phi[i] = sum(view(ek.pair_u, :, i))
        end
    end
    if ek.frac > 0                                     # fractional molecule: store the bare row, scale the matrix
        s = ek.frac; λ = ek.lam[s]
        ek.frac_u = 0.0; ek.frac_w = 0.0
        @inbounds for j in 1:N
            ek.bare_u[j] = ek.pair_u[s, j]; ek.bare_w[j] = ek.pair_w[s, j]
            ek.frac_u += ek.bare_u[j]; ek.frac_w += ek.bare_w[j]
            ek.pair_u[s, j] = λ * ek.bare_u[j]; ek.pair_u[j, s] = ek.pair_u[s, j]
            ek.pair_w[s, j] = λ * ek.bare_w[j]; ek.pair_w[j, s] = ek.pair_w[s, j]
        end
        @inbounds for i in 1:N
            ek.phi[i] = sum(view(ek.pair_u, :, i)); ek.wsum[i] = sum(view(ek.pair_w, :, i))
        end
    end
    if _has_recip(sys, ff)                           # image energies e_i (EwaldRecip.jl): per-molecule part of φ_i
        @inbounds for i in 1:N
            e = sys.recip.eimg[i]
            if i == ek.frac
                ek.frac_u += e; ek.phi[i] += ek.lam[i] * e
            else
                ek.phi[i] += e
            end
        end
    end
    R = 0.0
    for i in 1:N
        ek.m[i] = _mob(ek, sys.β, i); R += ek.m[i]
    end
    ek.R = R
    (R > 0.0 && isfinite(R)) || error("rigid eKMC: total rate R = $R")
    return nothing
end

"Σ_i λ_i e_i: the image energies inside φ (λ-scaled for the fractional molecule)."
function _image_energy_sum(ek::RigidEKMCState, sys::RigidSystem)::Float64
    E = 0.0
    @inbounds for i in 1:sys.n_mol
        E += (i == ek.frac ? ek.lam[i] : 1.0) * sys.recip.eimg[i]
    end
    return E
end

"Appendix-1 update after molecule k has been moved: row/column k of energies and virials, all φ_j, Σ_j w_ij, mobilities, R."
function update_rigid_ekmc_after_move!(ek::RigidEKMCState, sys::RigidSystem, ff::SiteForceField, k::Int)
    N = sys.n_mol
    β = sys.β
    molecule_committed!(sys, k)                         # reciprocal space S_k, ΔS_k, Stot, e_k; cell list (no-ops otherwise)
    eval_row!(ek.row_u, ek.row_w, sys, ff, k; recip=false)   # threaded when N ≥ PARALLEL_MIN_MOL
    if _has_recip(sys, ff)                              # Fourier row from the committed column S_k (no recomputation)
        recip_row_add_committed!(ek.row_u, sys.recip, k)
        @inbounds ek.row_u[k] = 0.0
    end
    if ek.frac > 0                                      # fractional molecule: bare row kept, matrix entries scaled by λ
        s = ek.frac; λ = ek.lam[s]
        if k == s
            ek.frac_u = 0.0; ek.frac_w = 0.0
            @inbounds for j in 1:N
                ek.bare_u[j] = ek.row_u[j]; ek.bare_w[j] = ek.row_w[j]
                ek.frac_u += ek.row_u[j]; ek.frac_w += ek.row_w[j]
                ek.row_u[j] *= λ; ek.row_w[j] *= λ
            end
        else
            ek.frac_u += ek.row_u[s] - ek.bare_u[k]; ek.frac_w += ek.row_w[s] - ek.bare_w[k]
            ek.bare_u[k] = ek.row_u[s]; ek.bare_w[k] = ek.row_w[s]
            ek.row_u[s] *= λ; ek.row_w[s] *= λ
        end
    end
    phik = 0.0; wk = 0.0
    @inbounds for j in 1:N
        j == k && continue
        u_old = ek.pair_u[k, j]; w_old = ek.pair_w[k, j]
        u_new = ek.row_u[j]; w_new = ek.row_w[j]
        if u_new != u_old
            ek.pair_u[k, j] = u_new; ek.pair_u[j, k] = u_new
            ek.phi[j] += u_new - u_old
            ek.m[j] = _mob(ek, β, j)
        end
        if w_new != w_old
            ek.pair_w[k, j] = w_new; ek.pair_w[j, k] = w_new
            ek.wsum[j] += w_new - w_old
        end
        phik += u_new; wk += w_new
    end
    if _has_recip(sys, ff)
        e = sys.recip.eimg[k]
        k == ek.frac ? (ek.frac_u += e; phik += ek.lam[k] * e) : (phik += e)
    end
    ek.phi[k] = phik; ek.wsum[k] = wk
    ek.m[k] = _mob(ek, β, k)
    # R is always resummed: mobilities span ~50 orders of magnitude (a relocated molecule can have
    # exp(βu) ≈ 1e48), so an incremental R = R + Δm suffers catastrophic cancellation when such a term
    # is removed (found by the LJ-diatomic MC↔eKMC test, 2026-10-11). The O(N) sum is negligible; only
    # the exponentials of the molecules whose φ changed are recomputed above.
    R = sum(ek.m)
    ek.R = R
    (R > 0.0 && isfinite(R)) || error("rigid eKMC: total rate R = $R after moving molecule $k")
    return nothing
end

"""
    set_fractional_coupling!(ek, β, λ_new)

Change the coupling of the fractional molecule from `ek.lam[frac]` to `λ_new` without moving anything: the scaled
row, every φ_j, φ_s, the virial sums, the mobilities and R follow (O(N)). With λ_new = 0 the molecule is absent
(m_s = 0). Used by the rung hops of the coupling ladder (DESIGN §10.4).
"""
function set_fractional_coupling!(ek::RigidEKMCState, β::Float64, λ_new::Float64)
    s = ek.frac
    s > 0 || error("set_fractional_coupling!: no fractional molecule")
    λ_old = ek.lam[s]
    N = length(ek.phi)
    @inbounds for j in 1:N
        j == s && continue
        du = (λ_new - λ_old) * ek.bare_u[j]; dw = (λ_new - λ_old) * ek.bare_w[j]
        ek.pair_u[s, j] = λ_new * ek.bare_u[j]; ek.pair_u[j, s] = ek.pair_u[s, j]
        ek.pair_w[s, j] = λ_new * ek.bare_w[j]; ek.pair_w[j, s] = ek.pair_w[s, j]
        ek.phi[j] += du; ek.wsum[j] += dw
        du != 0.0 && (ek.m[j] = _mob(ek, β, j))
    end
    ek.lam[s] = λ_new
    ek.phi[s] = λ_new * ek.frac_u; ek.wsum[s] = λ_new * ek.frac_w
    ek.m[s] = _mob(ek, β, s)
    R = sum(ek.m)
    ek.R = R
    (R > 0.0 && isfinite(R)) || error("rigid eKMC: total rate R = $R after recoupling to λ = $λ_new")
    return nothing
end

"""
    capped_overlaps(sys, ff, k) -> (n, ucap_sum)

Number of molecule pairs (k, j) that `pair_energy(sys, ff, k, j)` would return as capped (a LJ site pair
closer than the overlap radius) and the sum of their cap energies, found without evaluating any energy:
COM pre-filter against extent_k + extent_j + r_cap, then site distances with early exit (same site order
as `pair_energy`, so the same cap value). O(N) distance checks, ≈ 20× cheaper than a row. Used by the
overlap fast path of `run_rigid_ekmc!` (DESIGN §5c item 1).
"""
function capped_overlaps(sys::RigidSystem, ff::SiteForceField, k::Int)
    ff.cap_overlaps || return 0, 0.0
    L = sys.L; N = sys.n_mol
    tk = sys.templates[sys.tmpl[k]]; sk = sys.site_start[k]
    rcap = sqrt(maximum(ff.rcap2_mix))
    n = 0; usum = 0.0
    @inbounds for j in 1:N
        j == k && continue
        tj = sys.templates[sys.tmpl[j]]
        Rx = _min_image(sys.com[1, j] - sys.com[1, k], L)
        Ry = _min_image(sys.com[2, j] - sys.com[2, k], L)
        Rz = _min_image(sys.com[3, j] - sys.com[3, k], L)
        dmax = sys.ext[k] + sys.ext[j] + rcap
        Rx * Rx + Ry * Ry + Rz * Rz > dmax * dmax && continue
        sj = sys.site_start[j]
        wrapk = _site_images(ff, L) && _pair_wraps(Rx, Ry, Rz, sys.ext[k] + sys.ext[j], L)
        found = false; ucap = 0.0
        for a in 1:tk.n_sites
            ta = tk.site_type[a]
            ta > 0 || continue
            ox, oy, oz = sys.site_off[1, sk + a], sys.site_off[2, sk + a], sys.site_off[3, sk + a]
            for b in 1:tj.n_sites
                tb = tj.site_type[b]
                tb > 0 || continue
                dx = Rx + sys.site_off[1, sj + b] - ox
                dy = Ry + sys.site_off[2, sj + b] - oy
                dz = Rz + sys.site_off[3, sj + b] - oz
                if wrapk
                    dx = _min_image(dx, L); dy = _min_image(dy, L); dz = _min_image(dz, L)
                end
                if dx * dx + dy * dy + dz * dz < ff.rcap2_mix[ta, tb]
                    found = true; ucap = ff.ucap_mix[ta, tb]
                    break
                end
            end
            found && break
        end
        if found
            n += 1; usum += ucap
        end
    end
    return n, usum
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
        sys.cells === nothing || (nb = cells_check(sys.cells, sys)) == 0 || @warn "cell list not current" nb
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

"""
Time-weighted block averager: blocks of `block_events` events, block mean = Σ x Δt / Σ Δt. The block
sums X_k = Σ x Δt and T_k = Σ Δt are kept as well (`block_sum_x`, `block_sum_t`) for the global ratio
estimator `ratio_mean` and its jackknife error `ratio_jackknife_stderr`.
"""
mutable struct TimeWeightedBlockAverager
    block_events::Int
    sum_x::Float64
    sum_t::Float64
    n::Int
    block_means::Vector{Float64}
    block_sum_x::Vector{Float64}
    block_sum_t::Vector{Float64}
end
TimeWeightedBlockAverager(block_events::Int) =
    TimeWeightedBlockAverager(block_events, 0.0, 0.0, 0, Float64[], Float64[], Float64[])
function Base.push!(b::TimeWeightedBlockAverager, x::Float64, dt::Float64)
    b.sum_x += x * dt; b.sum_t += dt; b.n += 1
    if b.n >= b.block_events
        push!(b.block_means, b.sum_x / b.sum_t)
        push!(b.block_sum_x, b.sum_x); push!(b.block_sum_t, b.sum_t)
        b.sum_x = 0.0; b.sum_t = 0.0; b.n = 0
    end
    return b
end

"""
    ratio_mean(b) -> Σ_k X_k / Σ_k T_k

Global kMC time average over the completed blocks (Tan et al. 2017: ⟨A⟩ = Σ A Δt / Σ Δt). This, not
the mean of the block ratios (`mean`), is the estimator when the block times T_k differ (JOURNAL
2026-10-12 entry 10: the blocked μ was biased by several kJ/mol for liquid water).
"""
ratio_mean(b::TimeWeightedBlockAverager) =
    isempty(b.block_sum_x) ? (b.sum_t > 0 ? b.sum_x / b.sum_t : NaN) : sum(b.block_sum_x) / sum(b.block_sum_t)

"""
    ratio_jackknife_stderr(b; f=identity)

Delete-one-block jackknife standard error (Efron, *The Jackknife, the Bootstrap and Other Resampling
Plans*, SIAM 1982, Ch. 2) of f(Σ X_k / Σ T_k): with θ_(k) = f((ΣX − X_k)/(ΣT − T_k)) and n blocks,
se² = (n − 1)/n · Σ_k (θ_(k) − θ̄_(·))². For equal block times and f = identity this is exactly the
standard error of the block means. Blocks must be long enough to be independent (Flyvbjerg &
Petersen, J. Chem. Phys. 91 (1989) 461). `f = log` gives the error of βμ_ex = ln⟨R/N⟩ directly.
"""
function ratio_jackknife_stderr(b::TimeWeightedBlockAverager; f=identity)
    n = length(b.block_sum_x)
    n < 2 && return NaN
    SX = sum(b.block_sum_x); ST = sum(b.block_sum_t)
    θ = [f((SX - b.block_sum_x[k]) / (ST - b.block_sum_t[k])) for k in 1:n]
    θ̄ = sum(θ) / n
    return sqrt((n - 1) / n * sum((θk - θ̄)^2 for θk in θ))
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
    RigidEKMCRun

Complete state of a rigid-molecule eKMC run between two events: the options, the eKMC bookkeeping, every
accumulator and counter. `init_rigid_ekmc_run` builds it, `advance_rigid_ekmc!` continues it and
`rigid_ekmc_results` returns the named tuple of `run_rigid_ekmc!`. With the `RigidSystem` (which owns the RNG)
it is the whole run, so serializing (sys, run) and continuing after deserialization reproduces the
uninterrupted run event for event (checkpoint/restart of long runs; test/test_ekmc_checkpoint.jl).
"""
mutable struct RigidEKMCRun
    events::Int
    Pext::Union{Nothing,Float64}
    sample_every::Int
    burnin_events::Int
    tail_route::Symbol
    λ::Float64
    mc_δ::Float64
    mc_δθ::Float64
    species_mu::Bool
    trace_molecule::Int
    overlap_fast_path::Bool
    reservoirs::Union{Nothing,Vector}
    flex_mols::Vector{Int}
    vol_period::Int
    adapt_dV_moves::Int
    adapt_dV_window::Int
    vol_pressure::Symbol
    vol_scheme::Symbol
    vol_equil_moves::Int
    ek::RigidEKMCState
    Uin::Vector{Float64}
    acc::ChemicalPotentialAccumulator
    acc_npt::NPTChemicalPotentialAccumulator
    ba_U::TimeWeightedBlockAverager
    ba_P::TimeWeightedBlockAverager
    ba_rho::TimeWeightedBlockAverager
    ba_V::TimeWeightedBlockAverager
    ba_RN::TimeWeightedBlockAverager
    ba_Ui::TimeWeightedBlockAverager
    ba_RNV::TimeWeightedBlockAverager
    ba_RNs::Vector{TimeWeightedBlockAverager}
    nspec::Int
    n_spec::Vector{Int}
    sp_sum::Vector{Float64}
    trace_phi::Vector{Float64}
    trace_dt::Vector{Float64}
    win_V::Vector{Float64}
    win_t::Vector{Float64}
    dV_history::Vector{Float64}
    reg_n::Int
    vol_moves::Int
    next_vol::Int
    n_fast::Int
    n_reloc::Int
    n_bursts::Int
    ev::Int
    n_full::Int
    reg_V::Float64
    reg_P::Float64
    reg_VV::Float64
    reg_VP::Float64
    p_part_sum::Float64
    p_part_t::Float64
    t_total::Float64
    t_since_vol::Float64
    max_dV_current::Float64
    bar_species::Int
    bar::Union{Nothing,BennettAccumulator}
    bar_members::Vector{Int}
    ladder::Union{Nothing,CouplingLadder}
    mc_local_radius::Float64
    lmoves::Union{Nothing,LocalMoves}
    slmoves::Union{Nothing,SolventLocalMoves}
    # optional diagnostic (production, ladder runs): time-weighted histograms of dihedrals of the fractional molecule per rung,
    # dih_hist[k][d][b] = kMC time at rung k with dihedral d (atoms dih_atoms[d], body frame) in bin b of DIH_BINS over (−π, π]
    dih_atoms::Vector{NTuple{4,Int}}
    dih_hist::Vector{Vector{Vector{Float64}}}
end

"""
    init_rigid_ekmc_run(sys, ff; events, kwargs...) -> RigidEKMCRun

Validate the options of `run_rigid_ekmc!` (same keywords) and build the run state at event 0.
"""
function init_rigid_ekmc_run(sys::RigidSystem, ff::SiteForceField; events::Int,
                         track_dihedrals::Vector{NTuple{4,Int}}=NTuple{4,Int}[],
                         Pext::Union{Nothing,Float64}=nothing, max_dV::Float64=0.0,
                         vol_every_sweeps::Int=10, sample_every::Int=10, block_events::Int=1000,
                         debug_check_every::Int=0, adapt_dV_moves::Int=0, adapt_dV_window::Int=100,
                         vol_pressure::Symbol=:partial_average, vol_scheme::Symbol=:newton,
                         vol_equil_moves::Int=200, burnin_events::Int=0, tail_route::Symbol=:virial,
                         mc_kernel_rate::Float64=0.0, mc_δ::Float64=0.1, mc_δθ::Float64=0.3,
                         vol_every_events::Int=0, species_mu::Bool=false, trace_molecule::Int=0,
                         species_rate_scale::Union{Nothing,AbstractVector{Float64}}=nothing,
                         overlap_fast_path::Bool=false, reservoirs::Union{Nothing,Vector}=nothing,
                         bar_species::Int=0, ladder_solute::Int=0, ladder_lambda::Vector{Float64}=LADDER_LAMBDA_DEFAULT,
                         ladder_nu::Float64=NaN, ladder_eta::Union{Nothing,Vector{Float64}}=nothing, ladder_wl_events::Int=0,
                         ladder_wl_fmin::Float64=1e-3, ladder_wl_flat::Float64=0.8, ladder_eta_ramp::Float64=0.0,
                         ladder_solute_rate_frac::Float64=0.1, ladder_adapt_blocks::Bool=false, ladder_adapt_damp::Float64=1.0,
                         mc_local_radius::Float64=0.0, ladder_jump::Bool=false, ladder_jump_nu::Float64=NaN,
                         ladder_jump_max::Int=0, ladder_local::Bool=false, local_kinds::Vector{Int}=Int[],
                         local_a::Float64=NaN, local_θ::Float64=NaN, local_nu::Float64=NaN, local_nuD::Float64=NaN,
                         local_internal::Bool=false, local_site_a::Float64=0.03, local_φ::Float64=NaN,
                         local_solvent::Bool=false, solvent_kinds::Vector{Int}=Int[], solvent_a::Float64=NaN,
                         solvent_θ::Float64=NaN, solvent_nu::Float64=NaN, solvent_nuD::Float64=NaN,
                         neighbor_cells::Bool=false, solvent_flexible::Bool=false, solvent_site_a::Float64=NaN,
                         solvent_φ::Float64=NaN)
    N = sys.n_mol
    λ = mc_kernel_rate
    flex_mols = [i for i in 1:N if isflexible(sys.templates[sys.tmpl[i]])]
    if !isempty(flex_mols)
        reservoirs === nothing && error("run_rigid_ekmc!: flexible molecules need `reservoirs` (one ConformerReservoir per flexible template)")
        # the MC kernel moves flexible molecules as rigid bodies (intramolecular energy unchanged, so the Metropolis
        # acceptance on the intermolecular ΔU is exact); conformations are resampled at relocations/recouplings as in stock eKMC
        for i in flex_mols
            reservoirs[sys.tmpl[i]] === nothing && error("run_rigid_ekmc!: no reservoir for template $(sys.tmpl[i])")
        end
    end
    Uin = [intra_energy_molecule(sys, ff, i) for i in 1:N]      # intramolecular energies, updated at relocations
    vol_period = vol_every_events > 0 ? vol_every_events : vol_every_sweeps * N
    λ >= 0.0 || error("run_rigid_ekmc!: mc_kernel_rate must be ≥ 0")
    vol_pressure in (:partial_average, :instantaneous) ||
        error("run_rigid_ekmc!: vol_pressure must be :partial_average or :instantaneous")
    vol_scheme in (:newton, :auxiliary) ||
        error("run_rigid_ekmc!: vol_scheme must be :newton (Kelly 2020 Eq. 3.3) or :auxiliary (Tan 2017 Eq. 12)")
    # running least-squares regression of the block pressure P̄ on V for the Newton slope
    reg_n = 0; reg_V = 0.0; reg_P = 0.0; reg_VV = 0.0; reg_VP = 0.0
    p_part_sum = 0.0; p_part_t = 0.0
    rate_scale = species_rate_scale === nothing ? nothing : [species_rate_scale[sys.tmpl[i]] for i in 1:N]
    neighbor_cells && attach_cells!(sys, ff)          # molecular cell list for the real-space rows (MolCells.jl)
    ek = init_rigid_ekmc(sys, ff; debug_check_every=debug_check_every, rate_scale=rate_scale)
    # with a coupling ladder the standard observables are read at the coupled rung only: ≈ 1/K of the kMC TIME when η is
    # flat, but far fewer EVENTS (events crowd into the burst states of the soft rungs; dimer test: 0.6 % of the events at
    # 11 % of the time), so their blocks are shortened by 10K; check `length(U.block_means)` before trusting the errors
    std_block = ladder_solute > 0 ? max(sample_every, block_events ÷ (10 * length(ladder_lambda))) : block_events
    acc = ChemicalPotentialAccumulator()
    acc_npt = NPTChemicalPotentialAccumulator()
    ba_U = TimeWeightedBlockAverager(std_block ÷ sample_every)
    ba_P = TimeWeightedBlockAverager(std_block ÷ sample_every)
    ba_rho = TimeWeightedBlockAverager(std_block ÷ sample_every)
    ba_V = TimeWeightedBlockAverager(std_block ÷ sample_every)
    ba_RN = TimeWeightedBlockAverager(std_block)                  # R/N, every event (the μ observable)
    ba_Ui = TimeWeightedBlockAverager(std_block ÷ sample_every)   # intramolecular energy per flexible molecule
    ba_RNV = TimeWeightedBlockAverager(std_block ÷ sample_every)  # (R/N)/V for blocked NPT μ (A.132)
    # per-species μ: exp(βμ_ex,s) = ⟨(1/N_s) Σ_{i∈s} exp(βu_i)⟩_time (Tan 2017 Eq. 8 applied to the
    # molecules of species s; the particle-removal estimator of Shing & Gubbins, Mol. Phys. 46 (1982)
    # 1109, with the kMC time weights). For a single solute (N_s = 1) it is ⟨exp(βu_solute)⟩.
    nspec = length(sys.templates)
    n_spec = zeros(Int, nspec)
    for i in 1:N
        n_spec[sys.tmpl[i]] += 1
    end
    ba_RNs = species_mu ? [TimeWeightedBlockAverager(std_block) for _ in 1:nspec] :      # every event
                          TimeWeightedBlockAverager[]
    sp_sum = zeros(Float64, nspec)
    trace_phi = Float64[]; trace_dt = Float64[]     # (u_i, Δt) of `trace_molecule` at every sample
    # Bennett estimator (DESIGN §10.3): landing and time-weighted energy histograms of one species
    0 <= bar_species <= nspec || error("run_rigid_ekmc!: bar_species must be 0 (off) or a template index")
    bar = bar_species > 0 ? BennettAccumulator(bar_species, std_block) : nothing
    bar_members = [i for i in 1:N if sys.tmpl[i] == bar_species]
    # coupling ladder (DESIGN §10.4): one fractional molecule, started fully coupled
    ladder = nothing
    if ladder_solute > 0
        1 <= ladder_solute <= N || error("run_rigid_ekmc!: ladder_solute out of range")
        Pext === nothing || error("run_rigid_ekmc!: the coupling ladder is NVT only")
        overlap_fast_path && error("run_rigid_ekmc!: the coupling ladder is not implemented with the overlap fast path")
        isfinite(ladder_nu) || error("run_rigid_ekmc!: ladder_nu (hop attempt frequency) is required")
        # initial weights: given η, or the linear ramp η_k = λ_k·βμ_guess (flat if F(λ) were linear; Wang–Landau refines)
        η0 = ladder_eta !== nothing ? ladder_eta : ladder_eta_ramp .* ladder_lambda
        ladder = CouplingLadder(ladder_solute, ladder_lambda, block_events; η=η0, ν=ladder_nu, wl_events=ladder_wl_events,
                                wl_fmin=ladder_wl_fmin, wl_flat=ladder_wl_flat,
                                csol=solute_rate_scale(ladder_lambda, ladder_nu, ladder_eta_ramp; frac=ladder_solute_rate_frac),
                                adapt_blocks=ladder_adapt_blocks, adapt_damp=ladder_adapt_damp,
                                jump=ladder_jump, jump_ν=isnan(ladder_jump_nu) ? ladder_nu : ladder_jump_nu,
                                jump_max=ladder_jump_max > 0 ? ladder_jump_max : length(ladder_lambda))
        ek.frac = ladder_solute; ek.lam[ladder_solute] = 1.0
        ladder_jump && (ek.c[ladder_solute] = ladder.csol[end])     # hub: no relocation at the coupled rung either
        rebuild_rigid_ekmc!(ek, sys, ff)
    end
    # rejection-free local moves of the fractional molecule (DESIGN §11.2): default slots 3 T + 3 R (+ 2 C if flexible),
    # step sizes mc_δ / mc_δθ, ν_L = ν_D = ladder_nu unless given
    lmoves = nothing
    if ladder_local
        ladder === nothing && error("run_rigid_ekmc!: ladder_local needs the coupling ladder (ladder_solute)")
        sflex = isflexible(sys.templates[sys.tmpl[ladder_solute]])
        kinds = isempty(local_kinds) ? vcat(fill(LM_T, 3), fill(LM_R, 3), sflex ? fill(LM_C, 2) : Int[]) : copy(local_kinds)
        # local_internal (JOURNAL 80): + 4 single-site slots and, with a rotatable bond, 2 torsion slots
        if local_internal && isempty(local_kinds)
            sflex || error("run_rigid_ekmc!: local_internal needs a flexible fractional molecule")
            append!(kinds, fill(LM_S, 4))
            isempty(rotatable_bonds(sys.templates[sys.tmpl[ladder_solute]])) || append!(kinds, fill(LM_D, 2))
        end
        (!sflex && any(k -> k in (LM_C, LM_S, LM_D), kinds)) && error("run_rigid_ekmc!: conformer/site/torsion slots for a rigid molecule")
        lmoves = LocalMoves(sys, ladder_solute, length(ladder_lambda); kinds=kinds,
                            a=isnan(local_a) ? mc_δ : local_a, θmax=isnan(local_θ) ? mc_δθ : local_θ,
                            ν=isnan(local_nu) ? ladder_nu : local_nu, νD=isnan(local_nuD) ? ladder_nu : local_nuD,
                            reservoir=sflex ? reservoirs[sys.tmpl[ladder_solute]] : nothing,
                            site_a=local_site_a, φmax=isnan(local_φ) ? mc_δθ : local_φ)
    end
    # rejection-free local moves of every rigid molecule except the fractional one (DESIGN §11.2 stage 2): default slots
    # 3 T + 3 R, half-widths mc_δ / mc_δθ, ν = solvent_nu (default ladder_nu; required without a ladder), ν_D = ν
    slmoves = nothing
    if local_solvent
        Pext === nothing || error("run_rigid_ekmc!: solvent L-moves are NVT only")
        overlap_fast_path && error("run_rigid_ekmc!: solvent L-moves are not implemented with the overlap fast path")
        νs = isnan(solvent_nu) ? ladder_nu : solvent_nu
        isfinite(νs) || error("run_rigid_ekmc!: local_solvent needs solvent_nu (or a ladder's ladder_nu)")
        if solvent_flexible
            # DESIGN §14: every molecule but the fractional one, flexible ones with single-site and torsion slots. Mixtures of
            # rigid and flexible movers (e.g. ethanol + SPC/E): the slot list is shared, a rigid mover's LM_S/LM_D slots and the
            # LM_D slots of a molecule without a rotatable bond are inactive (rate 0, SolventLocalMoves.act)
            smols = [i for i in 1:N if i != ladder_solute]
            skinds = if isempty(solvent_kinds)
                flex = any(i -> isflexible(sys.templates[sys.tmpl[i]]), smols)
                tors = any(i -> isflexible(sys.templates[sys.tmpl[i]]) && !isempty(rotatable_bonds(sys.templates[sys.tmpl[i]])), smols)
                vcat(fill(LM_T, 3), fill(LM_R, 3), flex ? fill(LM_S, 4) : Int[], tors ? fill(LM_D, 2) : Int[])
            else
                solvent_kinds
            end
        else
            smols = [i for i in 1:N if i != ladder_solute && !isflexible(sys.templates[sys.tmpl[i]])]
            skinds = isempty(solvent_kinds) ? vcat(fill(LM_T, 3), fill(LM_R, 3)) : solvent_kinds
        end
        slmoves = SolventLocalMoves(sys, ff, smols; kinds=skinds,
                                    a=isnan(solvent_a) ? mc_δ : solvent_a, θmax=isnan(solvent_θ) ? mc_δθ : solvent_θ,
                                    ν=νs, νD=isnan(solvent_nuD) ? νs : solvent_nuD, frac=ladder_solute,
                                    site_a=isnan(solvent_site_a) ? local_site_a : solvent_site_a,
                                    φmax=isnan(solvent_φ) ? mc_δθ : solvent_φ)
    end
    t_total = 0.0
    # Tan 2017 Appendix 4: during the first `adapt_dV_moves` volume moves, every `adapt_dV_window`
    # moves set ΔV_max to half the standard deviation of the kMC-time-weighted volume histogram
    vol_moves = 0
    t_since_vol = 0.0
    win_V = Float64[]; win_t = Float64[]
    max_dV_current = max_dV
    dV_history = Float64[]
    # Overlap fast path (DESIGN §5c item 1; a switch, OFF by default). A relocated molecule k that lands with
    # n ≥ 1 capped pairs is in a state X′ with m_k′ = c_k exp(β(Σcaps + u_rest)). Its residence time
    # ln(1/ξ)/R′ is below resolution and X′ contributes ln(1/ξ)·m_k′/R′ to Σ R Δt and ln(1/ξ)/c_k · m_k′/R′ to
    # the species sum of k. The fast path takes m_k′/R′ = 1 and redraws k as the next mover without a row
    # evaluation (dt_f = ln(1/ξ)/m_k′ with m_k′ from the caps alone keeps the products exact), evaluating the
    # row once at the first landing without caps. APPROXIMATION: a capped partner j has m_j′ = c_j exp(β(φ_j +
    # U_cap)), so m_j′/m_k′ = (c_j/c_k) exp(β(φ_j − u_rest)); the fast path neglects (a) j moving next and
    # (b) j's species receiving ln(1/ξ)·m_j′/R′. Both are O(exp(β(φ_j − u_rest))): ≈ e^{−19} for liquid water
    # (every partner strongly bound), but ≈ e^{−1} when the partner is a weakly bound solute — then the
    # solute's μ is biased at the percent level. Use only when every species is strongly bound; the exact route
    # to most of the saving is a cell list for the row. At liquid density 92 % of landings are capped.
    next_vol = vol_period
    n_fast = 0; n_reloc = 0; n_bursts = 0
    ev = 0; n_full = 0
    return RigidEKMCRun(events, Pext, sample_every, burnin_events, tail_route, λ, mc_δ, mc_δθ, species_mu, trace_molecule, overlap_fast_path, reservoirs, flex_mols, vol_period, adapt_dV_moves, adapt_dV_window, vol_pressure, vol_scheme, vol_equil_moves, ek, Uin, acc, acc_npt, ba_U, ba_P, ba_rho, ba_V, ba_RN, ba_Ui, ba_RNV, ba_RNs, nspec, n_spec, sp_sum, trace_phi, trace_dt, win_V, win_t, dV_history, reg_n, vol_moves, next_vol, n_fast, n_reloc, n_bursts, ev, n_full, reg_V, reg_P, reg_VV, reg_VP, p_part_sum, p_part_t, t_total, t_since_vol, max_dV_current, bar_species, bar, bar_members, ladder, mc_local_radius, lmoves, slmoves,
                        track_dihedrals, [[zeros(DIH_BINS) for _ in track_dihedrals] for _ in 1:(ladder === nothing ? 0 : nrungs(ladder))])
end

"""
    advance_rigid_ekmc!(r, sys, ff; until=r.events, wall_seconds=Inf) -> r

Continue the run until event `min(until, r.events)` or until `wall_seconds` of wall time have passed (checked
every 4096 events); a stopped run continues exactly where it left off at the next call. `r.ev` is the number
of events done. (An overlap fast-path burst, if that switch is on, may run past `until`, as it would past
any event of an uninterrupted run; it never runs past `r.events`.)
"""
function advance_rigid_ekmc!(r::RigidEKMCRun, sys::RigidSystem, ff::SiteForceField; until::Int=r.events,
                             wall_seconds::Float64=Inf)
    N = sys.n_mol
    deadline = time() + wall_seconds
    events = r.events
    Pext = r.Pext
    sample_every = r.sample_every
    burnin_events = r.burnin_events
    tail_route = r.tail_route
    λ = r.λ
    mc_δ = r.mc_δ
    mc_δθ = r.mc_δθ
    species_mu = r.species_mu
    trace_molecule = r.trace_molecule
    overlap_fast_path = r.overlap_fast_path
    reservoirs = r.reservoirs
    flex_mols = r.flex_mols
    vol_period = r.vol_period
    adapt_dV_moves = r.adapt_dV_moves
    adapt_dV_window = r.adapt_dV_window
    vol_pressure = r.vol_pressure
    vol_scheme = r.vol_scheme
    vol_equil_moves = r.vol_equil_moves
    ek = r.ek
    Uin = r.Uin
    acc = r.acc
    acc_npt = r.acc_npt
    ba_U = r.ba_U
    ba_P = r.ba_P
    ba_rho = r.ba_rho
    ba_V = r.ba_V
    ba_RN = r.ba_RN
    ba_Ui = r.ba_Ui
    ba_RNV = r.ba_RNV
    ba_RNs = r.ba_RNs
    nspec = r.nspec
    n_spec = r.n_spec
    sp_sum = r.sp_sum
    trace_phi = r.trace_phi
    trace_dt = r.trace_dt
    win_V = r.win_V
    win_t = r.win_t
    dV_history = r.dV_history
    reg_n = r.reg_n
    vol_moves = r.vol_moves
    next_vol = r.next_vol
    n_fast = r.n_fast
    n_reloc = r.n_reloc
    n_bursts = r.n_bursts
    ev = r.ev
    n_full = r.n_full
    reg_V = r.reg_V
    reg_P = r.reg_P
    reg_VV = r.reg_VV
    reg_VP = r.reg_VP
    p_part_sum = r.p_part_sum
    p_part_t = r.p_part_t
    t_total = r.t_total
    t_since_vol = r.t_since_vol
    max_dV_current = r.max_dV_current
    bar_species = r.bar_species; bar = r.bar; bar_members = r.bar_members
    ladder = r.ladder; mc_local_radius = r.mc_local_radius; lmoves = r.lmoves; slmoves = r.slmoves
    stop = min(until, events)
    while ev < stop
        (ev & 4095 == 0) && time() > deadline && break
        ev += 1; n_full += 1          # n_full counts the states that get a full update (sampling stride for U, P)
        # weight of the current configuration: draw its Δt here and reuse it in the step
        R = ek.R                       # relocation rate Σ exp(βu_i) (the μ observable)
        w_up = 0.0; w_dn = 0.0; w_hub = 0.0; w_loc = 0.0; w_ref = 0.0
        ladder === nothing || ((w_up, w_dn) = hop_rates(ladder, sys.β, ek.frac_u); w_hub = hub_rate(ladder, sys.β, ek.frac_u))
        if lmoves !== nothing && ek.lam[lmoves.mol] > 0.0     # L-moves and the refresh of their candidates (λ > 0 only)
            w_loc = lm_total_rate!(lmoves, sys, ff, sys.β, ek.lam[lmoves.mol], ek.frac_u); w_ref = lmoves.νD
        end
        w_sl = 0.0; w_slref = 0.0
        if slmoves !== nothing                                # solvent L-moves and their refreshes
            w_sl = slm_total_rate!(slmoves, ek, sys.β); w_slref = length(slmoves.mols) * slmoves.νD
        end
        Rtot = R + λ + w_up + w_dn + w_hub + w_loc + w_ref + w_sl + w_slref   # total event rate incl. the MC-kernel event, the rung hops, the λ = 0 hub jumps and the L-moves
        # with a ladder the standard observables are read at the fully coupled rung only, after η is frozen
        coupled_now = ladder === nothing || (coupled(ladder) && !adapting(ladder))
        ladder_prod = ladder !== nothing && !adapting(ladder) && ev > burnin_events
        u = rand(sys.rng); u = max(eps(Float64), min(u, 1.0 - eps(Float64)))
        dt = log(1 / u) / Rtot
        V = sys.L^3
        if Pext !== nothing
            # pressure is free from the maintained virial; time-weight it over every event of the block
            p_part_sum += pressure(sys, ff, ekmc_virial(ek); tail=tail_route) * dt; p_part_t += dt
        end
        ladder_prod && ladder_time!(ladder, sys.β * ek.frac_u, dt)
        if ladder_prod && !isempty(r.dih_atoms)
            Bf = sys.body[ladder.solute]
            @inbounds for (d, (a1, a2, a3, a4)) in enumerate(r.dih_atoms)
                φd = dihedral_angle(Bf, a1, a2, a3, a4)
                r.dih_hist[ladder.k][d][clamp(floor(Int, (φd + π) / (2π) * DIH_BINS) + 1, 1, DIH_BINS)] += dt
            end
        end
        k_before = ladder === nothing ? 0 : ladder.k           # rung occupied during dt (for Wang–Landau)
        if ev > burnin_events && coupled_now
            if bar !== nothing
                bennett_event!(bar)
                @inbounds for i in bar_members
                    push_time!(bar, sys.β * ek.phi[i], dt)
                end
            end
            if n_full % sample_every == 0
                Up = 0.5 * sum(ek.phi)
                !_has_recip(sys, ff) || (Up += 0.5 * _image_energy_sum(ek, sys))   # e_i counted once, not halved
                push!(ba_U, (Up + lrc_energy(sys, ff)) / N, dt)
                push!(ba_P, pressure(sys, ff, ekmc_virial(ek); tail=tail_route), dt)
                push!(ba_rho, N / V, dt)
                push!(ba_V, V, dt)
                push!(ba_RNV, R / N / V, dt)
                push!(acc_npt, R / N, V, dt)
                isempty(flex_mols) || push!(ba_Ui, sum(Uin) / length(flex_mols), dt)
            end
            RN = R / N
            push!(ba_RN, RN, dt)
            acc.t_total += dt; acc.S += RN * dt; acc.S_sq += RN * RN * dt; acc.t_total_sq += dt * dt; acc.count += 1
            if species_mu
                fill!(sp_sum, 0.0)
                @inbounds for i in 1:N
                    sp_sum[sys.tmpl[i]] += ek.c[i] > 0.0 ? ek.m[i] / ek.c[i] : _mobility(sys.β, ek.phi[i])   # exp(βu_i), scale removed (c = 0: hub ladder)
                end
                for s in 1:nspec
                    n_spec[s] > 0 && push!(ba_RNs[s], sp_sum[s] / n_spec[s], dt)
                end
            end
            if trace_molecule > 0
                push!(trace_phi, ek.phi[trace_molecule]); push!(trace_dt, dt)
            end
        end
        # the event: MC-kernel sweep with probability λ/Rtot, a rung hop with (w_up + w_dn)/Rtot, otherwise a relocation
        # (a random number is drawn only when there is a non-relocation event type, so the stock stream is unchanged)
        ξev = (λ > 0.0 || ladder !== nothing || slmoves !== nothing) ? rand(sys.rng) * Rtot : Inf
        if λ > 0.0 && ξev < λ
            if mc_local_radius > 0.0 && ek.frac > 0
                ekmc_mc_sweep_local!(ek, sys, ff, mc_δ, mc_δθ, mc_local_radius)
            else
                ekmc_mc_sweep!(ek, sys, ff, mc_δ, mc_δθ)
            end
            lmoves === nothing || (lmoves.stale = true)
            slmoves === nothing || slm_rebuild_all!(slmoves, sys, ff)
        elseif ladder !== nothing && ξev < λ + w_up + w_dn
            up = ξev < λ + w_up
            knew = up ? ladder.k + 1 : ladder.k - 1
            sfr = ladder.solute
            if ladder.k == 1                           # recoupling from λ = 0: uniform pose (and conformer), bare row scaled by λ_2
                L = sys.L
                sflex = isflexible(sys.templates[sys.tmpl[sfr]])
                sflex && _set_body!(sys, sfr, draw_conformer(reservoirs[sys.tmpl[sfr]], sys.rng))
                set_molecule!(sys, sfr, MVector{3,Float64}(rand(sys.rng) * L, rand(sys.rng) * L, rand(sys.rng) * L),
                              uniform_random_quaternion(sys.rng))
                ek.lam[sfr] = ladder.λ[knew]; ek.c[sfr] = ladder.csol[knew]
                update_rigid_ekmc_after_move!(ek, sys, ff, sfr)
                sflex && (Uin[sfr] = flex_intra_energy(sys.templates[sys.tmpl[sfr]], ff, sys.body[sfr]))
                lmoves === nothing || (lmoves.stale = true)
                slmoves === nothing || slm_moved!(slmoves, sys, ff, sfr)
                ladder.n_recouple += 1
                ladder_prod && ladder_landing!(ladder, sys.β * ek.frac_u)     # a λ = 0 sample of βu_s (jump-chain identity)
            else
                ek.c[sfr] = ladder.csol[knew]
                set_fractional_coupling!(ek, sys.β, ladder.λ[knew])
                knew == 1 && (ladder.n_decouple += 1)
            end
            ladder.k = knew
            ladder_prod && (ladder.n_hop[knew] += 1)
        elseif ladder !== nothing && ξev < λ + w_up + w_dn + w_hub
            # λ = 0 hub (JOURNAL entry 56): from rung 1 recouple directly at rung m ≥ 3 with a uniform pose; from rung m ≥ 3 decouple
            sfr = ladder.solute
            if ladder.k == 1
                knew = hub_target(ladder, sys.rng)
                L = sys.L
                sflex = isflexible(sys.templates[sys.tmpl[sfr]])
                sflex && _set_body!(sys, sfr, draw_conformer(reservoirs[sys.tmpl[sfr]], sys.rng))
                set_molecule!(sys, sfr, MVector{3,Float64}(rand(sys.rng) * L, rand(sys.rng) * L, rand(sys.rng) * L),
                              uniform_random_quaternion(sys.rng))
                ek.lam[sfr] = ladder.λ[knew]; ek.c[sfr] = ladder.csol[knew]
                update_rigid_ekmc_after_move!(ek, sys, ff, sfr)
                sflex && (Uin[sfr] = flex_intra_energy(sys.templates[sys.tmpl[sfr]], ff, sys.body[sfr]))
                lmoves === nothing || (lmoves.stale = true)
                slmoves === nothing || slm_moved!(slmoves, sys, ff, sfr)
                ladder.n_recouple += 1
                ladder_prod && ladder_landing!(ladder, sys.β * ek.frac_u)     # uniform pose: a λ = 0 sample of βu_s
            else
                knew = 1
                ek.c[sfr] = ladder.csol[1]
                set_fractional_coupling!(ek, sys.β, 0.0)
                ladder.n_decouple += 1
            end
            ladder.k = knew
            ladder_prod && (ladder.n_jump[knew] += 1)
        elseif lmoves !== nothing && ξev < λ + w_up + w_dn + w_hub + w_loc
            # L-move of the fractional molecule (DESIGN §11.2): slot chosen ∝ its rate, inverse stored in the slot
            sfr = lmoves.mol
            jfired = lm_fire!(lmoves, sys, ff, ek, ξev - (λ + w_up + w_dn + w_hub))
            lmoves.kinds[jfired] in (LM_C, LM_S, LM_D) && (Uin[sfr] = flex_intra_energy(sys.templates[sys.tmpl[sfr]], ff, sys.body[sfr]))
            slmoves === nothing || slm_moved!(slmoves, sys, ff, sfr)
            if ladder_prod
                lmoves.n_move_rung[ladder.k] += 1
                ladder_pose_moved!(ladder)
            end
        elseif lmoves !== nothing && ξev < λ + w_up + w_dn + w_hub + w_loc + w_ref
            sfr = lmoves.mol
            lm_refresh!(lmoves, sys, isflexible(sys.templates[sys.tmpl[sfr]]) ? reservoirs[sys.tmpl[sfr]] : nothing)
        elseif slmoves !== nothing && ξev < λ + w_up + w_dn + w_hub + w_loc + w_ref + w_sl
            # solvent L-move (DESIGN §11.2 stage 2)
            i_mv = slm_fire!(slmoves, sys, ff, ek, ξev - (λ + w_up + w_dn + w_hub + w_loc + w_ref))
            # a site or torsion slot changed the mover's conformation: keep the U_intra observable current
            isflexible(sys.templates[sys.tmpl[i_mv]]) && (Uin[i_mv] = flex_intra_energy(sys.templates[sys.tmpl[i_mv]], ff, sys.body[i_mv]))
            if lmoves !== nothing
                ek.lam[lmoves.mol] > 0.0 ? lm_molecule_moved!(lmoves, sys, ff, i_mv) : (lmoves.stale = true)
            end
        elseif slmoves !== nothing && ξev < λ + w_up + w_dn + w_hub + w_loc + w_ref + w_sl + w_slref
            slm_refresh!(slmoves, sys, ff)
        else
            k = select_by_mobility(ek, sys.rng)
            L = sys.L
            kflex = isflexible(sys.templates[sys.tmpl[k]])
            kflex && _set_body!(sys, k, draw_conformer(reservoirs[sys.tmpl[k]], sys.rng))   # ideal-gas conformation
            set_molecule!(sys, k, MVector{3,Float64}(rand(sys.rng) * L, rand(sys.rng) * L, rand(sys.rng) * L),
                          uniform_random_quaternion(sys.rng))
            if overlap_fast_path && ff.cap_overlaps
                ncap, ucap = capped_overlaps(sys, ff, k)
                ncap > 0 && (n_bursts += 1)
                ck = ek.c[k]; sk = sys.tmpl[k]
                while ncap > 0 && ev < events
                    ev += 1
                    ξ = rand(sys.rng); ξ = max(eps(Float64), min(ξ, 1.0 - eps(Float64)))
                    lnξ = log(1 / ξ)
                    mk = ck * _mobility(sys.β, ucap)          # m_k′ from the caps alone; only m_k′·dt_f = ln(1/ξ) matters
                    dtf = lnξ / mk
                    if ev > burnin_events
                        push!(ba_RN, mk / N, dtf)
                        acc.t_total += dtf; acc.S += (mk / N) * dtf; acc.S_sq += (mk / N)^2 * dtf
                        acc.t_total_sq += dtf * dtf; acc.count += 1
                        if species_mu
                            for s in 1:nspec
                                n_spec[s] > 0 && push!(ba_RNs[s], s == sk ? (mk / ck) / n_spec[sk] : 0.0, dtf)
                            end
                        end
                        if trace_molecule == k          # other molecules: their stale φ is not their energy here
                            push!(trace_phi, ucap); push!(trace_dt, dtf)
                        end
                        bar !== nothing && sk == bar_species && push_landing!(bar, sys.β * ucap)   # capped landing: e^{−βu′} ≈ 0
                    end
                    ek.events += 1; n_fast += 1
                    t_total += dtf; t_since_vol += dtf
                    kflex && _set_body!(sys, k, draw_conformer(reservoirs[sys.tmpl[k]], sys.rng))
                    set_molecule!(sys, k, MVector{3,Float64}(rand(sys.rng) * L, rand(sys.rng) * L, rand(sys.rng) * L),
                                  uniform_random_quaternion(sys.rng))
                    ncap, ucap = capped_overlaps(sys, ff, k)
                end
            end
            update_rigid_ekmc_after_move!(ek, sys, ff, k)
            # landing energy u′_k: a Widom insertion sample into the (N−1)-system (jump-chain identity, DESIGN §10.3)
            bar !== nothing && ev > burnin_events && coupled_now && sys.tmpl[k] == bar_species && push_landing!(bar, sys.β * ek.phi[k])
            kflex && (Uin[k] = flex_intra_energy(sys.templates[sys.tmpl[k]], ff, sys.body[k]))
            slmoves === nothing || slm_moved!(slmoves, sys, ff, k)
            if lmoves !== nothing
                if k == lmoves.mol
                    lmoves.stale = true
                    ladder_prod && ladder_pose_moved!(ladder)
                elseif ek.lam[lmoves.mol] > 0.0
                    lm_molecule_moved!(lmoves, sys, ff, k)
                else
                    lmoves.stale = true                     # λ = 0: rebuild at the next recoupling instead
                end
            end
            n_reloc += 1
        end
        ek.events += 1
        t_total += dt
        t_since_vol += dt
        ladder !== nothing && adapting(ladder) && wl_update!(ladder, ev, dt, k_before)
        if _has_recip(sys, ff) && sys.recip.precision === :mixed && ek.events % RECIP_RESYNC_EVENTS == 0
            # mixed precision (GROMACS convention): exact Float64 rebuild removes the single-precision increment drift
            rebuild_rigid_ekmc!(ek, sys, ff)
            lmoves === nothing || (lmoves.stale = true)
            slmoves === nothing || slm_rebuild_all!(slmoves, sys, ff)
        end
        if ek.debug_check_every > 0 && ek.events % ek.debug_check_every == 0
            sys.cells === nothing || (nb = cells_check(sys.cells, sys)) == 0 || @warn "cell list not current" nb
            phi_saved = copy(ek.phi); R_saved = ek.R; W_saved = ekmc_virial(ek)
            rebuild_rigid_ekmc!(ek, sys, ff)
            # with :mixed reciprocal-space precision the single-precision increments drift by design (removed by the
            # rebuild just made): the drift is reported as information, not as a bookkeeping error
            mixed = _has_recip(sys, ff) && sys.recip.precision === :mixed
            dφ = maximum(abs.(phi_saved .- ek.phi)); dR = abs(R_saved - ek.R) / ek.R
            if dφ > 1e-8 || dR > 1e-8 || abs(W_saved - ekmc_virial(ek)) > 1e-8 * max(1.0, abs(W_saved))
                mixed ? (@info "rigid eKMC mixed-precision drift (resynchronised)" dφ dR) : (@warn "rigid eKMC bookkeeping drift" dφ dR)
            end
            if lmoves !== nothing && !lmoves.stale
                cu = copy(lmoves.cand_u); lm_rebuild!(lmoves, sys, ff)
                dc = maximum(abs.(cu .- lmoves.cand_u))
                dc > 1e-8 && (mixed ? (@info "L-move mixed-precision drift (resynchronised)" dc) : (@warn "L-move candidate energy drift" dc))
            end
            if slmoves !== nothing
                d = slm_check!(slmoves, sys, ff)
                d > 1e-8 && (mixed ? (@info "solvent L-move mixed-precision drift (resynchronised)" d) : (@warn "solvent L-move candidate energy drift" d))
            end
        end
        if Pext !== nothing && ev >= next_vol
            next_vol += vol_period
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
    r.reg_n = reg_n
    r.vol_moves = vol_moves
    r.next_vol = next_vol
    r.n_fast = n_fast
    r.n_reloc = n_reloc
    r.n_bursts = n_bursts
    r.ev = ev
    r.n_full = n_full
    r.reg_V = reg_V
    r.reg_P = reg_P
    r.reg_VV = reg_VV
    r.reg_VP = reg_VP
    r.p_part_sum = p_part_sum
    r.p_part_t = p_part_t
    r.t_total = t_total
    r.t_since_vol = t_since_vol
    r.max_dV_current = max_dV_current
    return r
end

"""
    rigid_ekmc_results(r) -> NamedTuple

The result of `run_rigid_ekmc!` from a run state (complete or partial).
"""
function rigid_ekmc_results(r::RigidEKMCRun)
    ba_U = r.ba_U
    ba_P = r.ba_P
    ba_rho = r.ba_rho
    ba_V = r.ba_V
    ba_RN = r.ba_RN
    ba_RNV = r.ba_RNV
    acc = r.acc
    acc_npt = r.acc_npt
    t_total = r.t_total
    ek = r.ek
    vol_moves = r.vol_moves
    max_dV_current = r.max_dV_current
    dV_history = r.dV_history
    ba_RNs = r.ba_RNs
    trace_phi = r.trace_phi
    trace_dt = r.trace_dt
    n_fast = r.n_fast
    n_bursts = r.n_bursts
    n_reloc = r.n_reloc
    ba_Ui = r.ba_Ui
    return (U=ba_U, P=ba_P, rho=ba_rho, V=ba_V, RN=ba_RN, RNV=ba_RNV, mu=acc, mu_npt=acc_npt,
            t_total=t_total, ek=ek, vol_moves=vol_moves, max_dV=max_dV_current, dV_history=dV_history,
            mc_events=ek.mc_events, mc_acceptance=ek.mc_trials > 0 ? ek.mc_accepted / ek.mc_trials : NaN,
            RN_species=ba_RNs, trace_phi=trace_phi, trace_dt=trace_dt,
            fast_events=n_fast, bursts=n_bursts, relocations=n_reloc, U_intra=ba_Ui, bar=r.bar, ladder=r.ladder, lmoves=r.lmoves, slmoves=r.slmoves)
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
Every blocked observable also carries its block sums, so the global time average is `ratio_mean` and
its error `ratio_jackknife_stderr` (use these for μ: `T·log(ratio_mean(RN))` ± `T·ratio_jackknife_stderr(RN; f=log)`).
`species_mu=true` adds `RN_species[s]`, the blocked (1/N_s) Σ_{i∈s} exp(βu_i) of every template s
(per-species μ_ex,s = T ln⟨·⟩; the mobilities are the clamped `_mobility` values, as in R), and
`trace_molecule=i` records (u_i, Δt) at every sample in `trace_phi`, `trace_dt` (tail diagnostics for
a single solute). `species_rate_scale[s]` multiplies the relocation rate of every molecule of template s
(`init_rigid_ekmc` `rate_scale`): balance and the per-species μ are unaffected, the share of relocation
bursts per species changes; with scaling the global R/N accumulators (`RN`, `mu`, `RNV`, `mu_npt`) no longer
give a physical μ — use `RN_species`. `overlap_fast_path` (default false) counts capped landings analytically
and redraws the mover without a row evaluation; it neglects the capped partner's mobility, a relative error
of order exp(β(φ_partner − u_rest)) — negligible for liquid water, not for a weakly bound solute (see the
comment in the loop); `fast_events`, `bursts` and `relocations` in the result count the events. The
diagnostic trace records fast events only when the traced molecule is the mover. R/N and the species sums are accumulated at every event
(block size `block_events`); U, P, ρ, V every `sample_every` events. `bar_species=s` (default 0 = off) collects the
landing-energy and time-weighted-energy histograms of template s in `bar` (a `BennettAccumulator`: Widom from
landings, Bennett acceptance ratio, removal estimator; DESIGN §10.3). Cost O(N_s) per event.
`ladder_solute=i` makes molecule i the fractional molecule of a coupling ladder (DESIGN §10.4, `CouplingLadder`):
rungs `ladder_lambda` (0 … 1), hop frequency `ladder_nu` (required; of the order of ⟨R⟩), weights `ladder_eta` or
Wang–Landau over the first `ladder_wl_events` events (`ladder_wl_fmin`, `ladder_wl_flat`); NVT, λ = 0, no fast path.
U, P, R/N, species μ and `bar` are then accumulated at the fully coupled rung only after η is frozen; the ladder's own
accumulators give `ladder_mu_occupancy` and `ladder_mu_mbar`; `ladder_summary` reports times per rung and hops.
`ladder_jump=true` adds the λ = 0 hub (direct jumps rung 1 ↔ m ≥ 3, attempt frequency `ladder_jump_nu`, default
`ladder_nu`) and stops relocating the fractional molecule at rungs ≥ 3 (JOURNAL entry 56, `hub_rate`).
`ladder_adapt_blocks=true` re-flattens η at every production block boundary from the previous block's rung times (the
occupancy estimator uses the η in force in each block; MBAR is η-independent) — for solvents where WL cannot converge
within its budget (water, JOURNAL entry 45).
`ladder_jump_max=m` limits the hub to rungs 3…m (both directions). `ladder_local=true` adds rejection-free local moves of the
fractional molecule at every rung with λ > 0 (`LocalMoves`, DESIGN §11.2): `local_kinds` slots (default 3 translations,
3 rotations, 2 conformer swaps if flexible), half-widths `local_a` (default `mc_δ`) and `local_θ` (default `mc_δθ`),
per-candidate prefactor `local_nu` and refresh rate `local_nuD` (both default `ladder_nu`); `local_internal=true` (flexible
fractional molecule, JOURNAL 80) adds 4 single-site slots (LM_S, half-width `local_site_a`, default 0.03 as the MC-stage site
move) and, if the molecule has a rotatable bond, 2 torsion slots (LM_D, half-angle `local_φ`, default `mc_δθ`), whose rates
carry ΔU_intra unscaled by λ; counters in `lmoves`,
residence diagnostics (`res_frac`, `pose_frac`) in `ladder_summary`. `local_solvent=true` gives every rigid molecule other than
the fractional one rejection-free local moves (`SolventLocalMoves`, DESIGN §11.2 stage 2): slots `solvent_kinds` (default 3
translations + 3 rotations), half-widths `solvent_a`/`solvent_θ` (default `mc_δ`/`mc_δθ`), per-candidate prefactor `solvent_nu`
(default `ladder_nu`; required without a ladder) and per-molecule refresh rate `solvent_nuD` (default `solvent_nu`); NVT only.
`solvent_flexible=true` (with `local_solvent`) makes every molecule but the fractional one a mover, flexible molecules with
single-site and torsion slots (DESIGN §14; default 3 T + 3 R + 4 S + 2 D, half-widths `solvent_site_a` (default
`local_site_a`) and `solvent_φ` (default `mc_δθ`)); rigid movers in a mixture keep only their T/R slots, molecules without a
rotatable bond no torsion slots (inactive slots have rate 0).
`neighbor_cells=true` evaluates the real-space rows (events, candidates) with a molecular cell list (`MolCells`); the rows are
identical to the all-pairs ones, the rebuilds stay all-pairs.
"""
function run_rigid_ekmc!(sys::RigidSystem, ff::SiteForceField; events::Int,
                         Pext::Union{Nothing,Float64}=nothing, max_dV::Float64=0.0,
                         vol_every_sweeps::Int=10, sample_every::Int=10, block_events::Int=1000,
                         debug_check_every::Int=0, adapt_dV_moves::Int=0, adapt_dV_window::Int=100,
                         vol_pressure::Symbol=:partial_average, vol_scheme::Symbol=:newton,
                         vol_equil_moves::Int=200, burnin_events::Int=0, tail_route::Symbol=:virial,
                         mc_kernel_rate::Float64=0.0, mc_δ::Float64=0.1, mc_δθ::Float64=0.3,
                         vol_every_events::Int=0, species_mu::Bool=false, trace_molecule::Int=0,
                         species_rate_scale::Union{Nothing,AbstractVector{Float64}}=nothing,
                         overlap_fast_path::Bool=false, reservoirs::Union{Nothing,Vector}=nothing,
                         bar_species::Int=0, ladder_solute::Int=0, ladder_lambda::Vector{Float64}=LADDER_LAMBDA_DEFAULT,
                         ladder_nu::Float64=NaN, ladder_eta::Union{Nothing,Vector{Float64}}=nothing, ladder_wl_events::Int=0,
                         ladder_wl_fmin::Float64=1e-3, ladder_wl_flat::Float64=0.8, ladder_eta_ramp::Float64=0.0,
                         ladder_solute_rate_frac::Float64=0.1, ladder_adapt_blocks::Bool=false, ladder_adapt_damp::Float64=1.0,
                         mc_local_radius::Float64=0.0, ladder_jump::Bool=false, ladder_jump_nu::Float64=NaN,
                         ladder_jump_max::Int=0, ladder_local::Bool=false, local_kinds::Vector{Int}=Int[],
                         local_a::Float64=NaN, local_θ::Float64=NaN, local_nu::Float64=NaN, local_nuD::Float64=NaN,
                         local_internal::Bool=false, local_site_a::Float64=0.03, local_φ::Float64=NaN,
                         local_solvent::Bool=false, solvent_kinds::Vector{Int}=Int[], solvent_a::Float64=NaN,
                         solvent_θ::Float64=NaN, solvent_nu::Float64=NaN, solvent_nuD::Float64=NaN,
                         neighbor_cells::Bool=false, solvent_flexible::Bool=false, solvent_site_a::Float64=NaN,
                         solvent_φ::Float64=NaN)
    r = init_rigid_ekmc_run(sys, ff; events=events, Pext=Pext, max_dV=max_dV, vol_every_sweeps=vol_every_sweeps,
                            sample_every=sample_every, block_events=block_events, debug_check_every=debug_check_every,
                            adapt_dV_moves=adapt_dV_moves, adapt_dV_window=adapt_dV_window, vol_pressure=vol_pressure,
                            vol_scheme=vol_scheme, vol_equil_moves=vol_equil_moves, burnin_events=burnin_events,
                            tail_route=tail_route, mc_kernel_rate=mc_kernel_rate, mc_δ=mc_δ, mc_δθ=mc_δθ,
                            vol_every_events=vol_every_events, species_mu=species_mu, trace_molecule=trace_molecule,
                            species_rate_scale=species_rate_scale, overlap_fast_path=overlap_fast_path, reservoirs=reservoirs,
                            bar_species=bar_species, ladder_solute=ladder_solute, ladder_lambda=ladder_lambda, ladder_nu=ladder_nu,
                            ladder_eta=ladder_eta, ladder_wl_events=ladder_wl_events, ladder_wl_fmin=ladder_wl_fmin, ladder_wl_flat=ladder_wl_flat,
                            ladder_eta_ramp=ladder_eta_ramp, ladder_solute_rate_frac=ladder_solute_rate_frac,
                            ladder_adapt_blocks=ladder_adapt_blocks, ladder_adapt_damp=ladder_adapt_damp, mc_local_radius=mc_local_radius,
                            ladder_jump=ladder_jump, ladder_jump_nu=ladder_jump_nu, ladder_jump_max=ladder_jump_max,
                            ladder_local=ladder_local, local_kinds=local_kinds, local_a=local_a, local_θ=local_θ,
                            local_nu=local_nu, local_nuD=local_nuD, local_internal=local_internal, local_site_a=local_site_a,
                            local_φ=local_φ, local_solvent=local_solvent, solvent_kinds=solvent_kinds,
                            solvent_a=solvent_a, solvent_θ=solvent_θ, solvent_nu=solvent_nu, solvent_nuD=solvent_nuD,
                            neighbor_cells=neighbor_cells, solvent_flexible=solvent_flexible,
                            solvent_site_a=solvent_site_a, solvent_φ=solvent_φ)
    advance_rigid_ekmc!(r, sys, ff)
    return rigid_ekmc_results(r)
end

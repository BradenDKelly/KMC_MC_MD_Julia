# ============================================================================
# Rigid united-atom solutes (TraPPE-UA) in the reduced units of `spce_model`
# ============================================================================
#
# Infinite-dilution solutes for the SPC/E eKMC validation (JOURNAL 2026-10-12, entry 14). Everything
# is in the units of `spce_model`: σ_OO = ϵ_OO = k_B = 1, q* = q[e]·q_scale (ke* = 1). Unlike-site LJ
# parameters follow Lorentz–Berthelot, as TraPPE-UA prescribes (Martin & Siepmann 1998, §2); the SPC/E
# oxygen is LJ type 1, the solute types follow.

"""
    trappe_ua_solute(solute::Symbol) -> (lj, sites, types, q, mass, name, mixing)

Rigid solute parameters in Å / K / e at the force field's equilibrium geometry (bending and torsion are
not sampled). All-atom OPLS-AA (Jorgensen, Maxwell, Tirado-Rives, JACS 118 (1996) 11225; parameters as in
the GROMACS `oplsaa.ff` tables; geometric mixing):
- `:methane` — C + 4 H, tetrahedral, C–H 1.090 Å.
- `:methanol` — C, 3 H, O, H(O), rigid at the OPLS-AA vacuum minimum (see the code for the coordinates).
United-atom TraPPE-UA (Lorentz–Berthelot mixing), kept for reference:
- `:methane_ua` — one pseudo-atom CH4, σ = 3.73 Å, ϵ/k = 148 K, q = 0
  (Martin & Siepmann, J. Phys. Chem. B 102 (1998) 2569, Table 1).
- `:methanol_ua` — CH3 (α-carbon bonded to O) σ = 3.75 Å, ϵ/k = 98 K, q = +0.265;
  hydroxyl O σ = 3.02 Å, ϵ/k = 93 K, q = −0.700; hydroxyl H no LJ, q = +0.435;
  bonds C–O 1.43 Å, O–H 0.945 Å; angle C–O–H 108.5°
  (Chen, Potoff, Siepmann, J. Phys. Chem. B 105 (2001) 3093, Tables 1–2; charges from OPLS-UA,
  Jorgensen, J. Phys. Chem. 90 (1986) 1276). Checked against the TraPPE-UA GROMACS implementation
  (jpotoff/trappeua: CH4 0.373 nm/1.23054 kJ/mol; CH3_O 0.375/0.814817; O 0.302/0.773245; H 0/0;
  bonds 0.1430/0.0945 nm; angle 108.5°).
Masses enter only the centre of mass: CH4 16.043, CH3 15.035, O 15.9994, H 1.008
(IUPAC standard atomic weights, as in the TraPPE files).
`lj` is the list of (σ_Å, ϵ_K) of the solute's LJ site types in the order of `types` (0 = no LJ).
"""
function trappe_ua_solute(solute::Symbol)
    if solute === :methane_ua
        sites = zeros(Float64, 3, 1)
        return (lj=[(3.73, 148.0)], sites=sites, types=[1], q=[0.0], mass=[16.043], name="CH4 (TraPPE-UA)",
                mixing=:lorentz_berthelot)
    elseif solute === :methanol_ua
        θ = 108.5 * π / 180
        sites = zeros(Float64, 3, 3)
        sites[:, 1] .= (1.43, 0.0, 0.0)                         # CH3 (α)
        sites[:, 2] .= (0.0, 0.0, 0.0)                          # O
        sites[:, 3] .= (0.945 * cos(θ), 0.945 * sin(θ), 0.0)    # H, angle C–O–H = θ at O
        return (lj=[(3.75, 98.0), (3.02, 93.0)], sites=sites, types=[1, 2, 0],
                q=[0.265, -0.700, 0.435], mass=[15.035, 15.9994, 1.008], name="CH3OH (TraPPE-UA)",
                mixing=:lorentz_berthelot)
    elseif solute === :methane
        # OPLS-AA methane: C opls_138 σ 3.50 Å, ϵ 0.066 kcal/mol = 33.2125 K, q −0.24; H opls_140 σ 2.50 Å,
        # ϵ 0.030 kcal/mol = 15.0966 K, q +0.06; tetrahedral, C–H 1.090 Å (the OPLS-AA vacuum minimum)
        h = 1.090 / sqrt(3.0)
        sites = zeros(Float64, 3, 5)
        sites[:, 2] .= (h, h, h); sites[:, 3] .= (h, -h, -h); sites[:, 4] .= (-h, h, -h); sites[:, 5] .= (-h, -h, h)
        return (lj=[(3.50, 33.2125), (2.50, 15.0966)], sites=sites, types=[1, 2, 2, 2, 2],
                q=[-0.24, 0.06, 0.06, 0.06, 0.06], mass=[12.011, 1.008, 1.008, 1.008, 1.008],
                name="CH4 (OPLS-AA, rigid)", mixing=:geometric)
    elseif solute === :methanol
        # OPLS-AA methanol: C opls_157 (3.50 Å, 33.2125 K, q 0.145), H(C) opls_156 (2.50 Å, 15.0966 K, q 0.040),
        # O opls_154 (3.12 Å, 0.170 kcal/mol = 85.5473 K, q −0.683), H(O) opls_155 (no LJ, q 0.418).
        # Rigid at the OPLS-AA vacuum minimum (GROMACS oplsaa.ff, cg to machine precision, 2026-10-12,
        # gromacs_fe/opls_geom): C–H 1.0904/1.0905, C–O 1.4119, O–H 0.9457 Å; H–C–H 108.4–108.6°,
        # H–C–O 110.2–110.5°, C–O–H 109.0°; the methyl is staggered with respect to the hydroxyl H.
        sites = zeros(Float64, 3, 6)
        sites[:, 1] .= (0.37930, -0.01687, 0.0)        # C
        sites[:, 2] .= (-0.03212, 0.46844, 0.88554)    # H1
        sites[:, 3] .= (0.06180, -1.06008, 0.0)        # H2
        sites[:, 4] .= (-0.03212, 0.46844, -0.88554)   # H3
        sites[:, 5] .= (1.78883, 0.06478, 0.0)         # O
        sites[:, 6] .= (2.04432, 0.97530, 0.0)         # HO
        return (lj=[(3.50, 33.2125), (2.50, 15.0966), (3.12, 85.5473)], sites=sites, types=[1, 2, 2, 2, 3, 0],
                q=[0.145, 0.040, 0.040, 0.040, -0.683, 0.418], mass=[12.011, 1.008, 1.008, 1.008, 15.9994, 1.008],
                name="CH3OH (OPLS-AA, rigid)", mixing=:geometric)
    elseif solute === :methanol_flex
        # fully flexible OPLS-AA methanol: the rigid geometry as reference, bonded terms from opls_methanol_bonded
        # (GROMACS units; converted to reduced units in spce_solute_model). DESIGN §7, first flexible target.
        r = trappe_ua_solute(:methanol)
        return (lj=r.lj, sites=r.sites, types=r.types, q=r.q, mass=r.mass, name="CH3OH (OPLS-AA, flexible)",
                mixing=:geometric, flexible=true)
    elseif solute in keys(OPLS_FLEX_ITP)
        # flexible OPLS-AA molecule read from the GROMACS topology (OPLSTopology.jl): every parameter from oplsaa.ff
        m = opls_itp_molecule(joinpath(OPLS_FFDIR, OPLS_FLEX_ITP[solute]))
        return (lj=[(σ * 10.0, ϵ / 0.0083144626) for (σ, ϵ) in m.lj], sites=m.X0 .* 10.0, types=m.types, q=m.q, mass=m.mass,
                name="$(OPLS_FLEX_ITP[solute]) (OPLS-AA, flexible)", mixing=:geometric, flexible=true, bonded=m.bt)
    else
        error("trappe_ua_solute: unknown solute $solute (use :methane, :methanol, :methanol_flex, :methane_ua, :methanol_ua, $(join(keys(OPLS_FLEX_ITP), ", ")))")
    end
end

"Flexible OPLS-AA solutes built from the GROMACS topologies shipped in oplsaa.ff (OPLSTopology.jl)."
const OPLS_FLEX_ITP = Dict(:methanol_itp => "methanol.itp", :ethanol_flex => "ethanol.itp", :propanol_flex => "1propanol.itp")

"""
    spce_solute_model(solute; α_per_angstrom=0.20, rc_angstrom=10.0, cutoff=:molecule,
                      cap_factor=1000.0, use_lrc=true, cap_overlaps=true, shift_lj=false)

SPC/E water (`spce_model`, same keyword arguments and defaults as the production model of JOURNAL
entry 8: molecule-based cutoff, α = 0.20 Å⁻¹, r_c = 10 Å, overlap cap 1000 ϵ) plus one TraPPE-UA
solute template (`trappe_ua_solute`: OPLS-AA `:methane`/`:methanol`, TraPPE-UA `:methane_ua`/`:methanol_ua`)
in the same reduced units. Returns a NamedTuple with `templates = [water, solute]` (template indices 1
and 2), the shared `ff` (LJ types: 1 = water O, 2… = solute sites; the solute model's mixing rule,
geometric for OPLS-AA, applied to the water–solute cross terms too), and the conversion constants of `spce_model`.
"""
function spce_solute_model(solute::Symbol; α_per_angstrom::Float64=0.20, rc_angstrom::Float64=10.0,
                           use_lrc::Bool=true, cap_overlaps::Bool=true, shift_lj::Bool=false,
                           cutoff::Symbol=:molecule, cap_factor::Float64=1000.0, coulomb::Symbol=:wolf,
                           epsilon_rf::Float64=Inf)
    m = spce_model(; α_per_angstrom=α_per_angstrom, rc_angstrom=rc_angstrom, use_lrc=use_lrc,
                   cap_overlaps=cap_overlaps, shift_lj=shift_lj, cutoff=cutoff, cap_factor=cap_factor,
                   coulomb=coulomb, epsilon_rf=epsilon_rf)
    σ_A, ϵ_K, qs = m.σ_angstrom, m.ϵ_kelvin, m.q_scale
    p = trappe_ua_solute(solute)
    types = [t > 0 ? t + 1 : 0 for t in p.types]          # shift past the water O type
    bonded = nothing
    if get(p, :flexible, false)
        bt = haskey(p, :bonded) ? p.bonded : first(opls_methanol_bonded())   # kJ/mol, nm, rad
        bonded = bonded_to_reduced(bt, σ_A, ϵ_K * 0.0083144626)
    end
    tmpl = RigidTemplate(p.name, p.sites ./ σ_A, types, p.q .* qs, p.mass; bonded=bonded)
    σ_types = vcat(1.0, [σ / σ_A for (σ, _) in p.lj])
    ϵ_types = vcat(1.0, [ϵ / ϵ_K for (_, ϵ) in p.lj])
    ff = SiteForceField(σ_types, ϵ_types; rc=rc_angstrom / σ_A, coulomb=coulomb,
                        α=α_per_angstrom * σ_A, ke=1.0, use_lrc=use_lrc, cap_overlaps=cap_overlaps,
                        shift_lj=shift_lj, cutoff=cutoff, cap_factor=cap_factor, mixing=p.mixing,
                        epsilon_rf=epsilon_rf)
    return (templates=[m.template, tmpl], water=m.template, solute=tmpl, ff=ff,
            σ_angstrom=σ_A, ϵ_kelvin=ϵ_K, q_scale=qs, mixing=p.mixing)
end

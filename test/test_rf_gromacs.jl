"""
Reaction-field electrostatics against GROMACS: 500 SPC/E + 1 OPLS-AA rigid methanol (test/data/
rf_reference_spce_methanol.gro, 6-decimal coordinates) evaluated with `gmx mdrun -rerun` for
coulombtype = Reaction-Field with ε_rf = ∞ and 78, and Cut-off (ε_rf = 1); LJ unshifted with the dispersion
tail (test/data/rf_reference_energies.txt). MolSim's flat-array `rf_energy` and the `RigidSystem` pair path
(site-based cutoff) must reproduce Coulomb (SR) and LJ (SR) to 2e-5 relative (GROMACS mixed precision over
~10⁶ pair terms); the tail to 1e-3 (GROMACS corrects dispersion only, MolSim also the r⁻¹² term). DESIGN §2.
"""

using Test
using MolSim
using Random
const MCm = MolSim.MC

@testset "Reaction field vs GROMACS: 500 SPC/E + OPLS-AA methanol" begin
    gro = readlines(joinpath(@__DIR__, "data", "rf_reference_spce_methanol.gro"))
    nat = parse(Int, strip(gro[2]))
    @test nat == 1506
    box = parse.(Float64, split(gro[3 + nat]))
    pos_nm = zeros(3, nat)
    for k in 1:nat
        l = gro[2 + k]
        pos_nm[:, k] .= (parse(Float64, l[21:31]), parse(Float64, l[32:42]), parse(Float64, l[43:53]))
    end
    ref = Dict{String,NTuple{3,Float64}}()
    for l in eachline(joinpath(@__DIR__, "data", "rf_reference_energies.txt"))
        startswith(l, "#") && continue
        f = split(l)
        ref[f[1]] = (parse(Float64, f[3]), parse(Float64, f[4]), parse(Float64, f[5]))
    end
    kJ_per_eps(m) = m.ϵ_kelvin * 0.0083144626
    for (key, εrf) in (("rfinf", Inf), ("rf78", 78.0), ("cut", 1.0))
        m = MCm.spce_solute_model(:methanol; coulomb=:rf, epsilon_rf=εrf, cutoff=:site, cap_overlaps=true)
        ff = m.ff; σ_A = m.σ_angstrom; kJ = kJ_per_eps(m)
        L = box[1] * 10 / σ_A
        # molecule order in the file: methanol (6 sites) first, then 500 waters
        sys = MCm.RigidSystem(m.templates, vcat(2, fill(1, 500)), L, 1 / (300.0 / m.ϵ_kelvin), Xoshiro(1))
        k = 0
        for i in 1:sys.n_mol
            t = sys.templates[sys.tmpl[i]]
            lab = pos_nm[:, k+1:k+t.n_sites] .* (10 / σ_A)
            k += t.n_sites
            MCm.molecule_from_sites!(sys, i, lab)
        end
        # the rigid fit reproduces the file coordinates (6-decimal nm ⇒ 1e-5 Å)
        sp = MCm.site_positions(sys)
        maxdev = 0.0
        for s in 1:nat, d in 1:3
            δ = sp[d, s] - pos_nm[d, s] * 10 / σ_A; δ -= L * round(δ / L)
            maxdev = max(maxdev, abs(δ) * σ_A)
        end
        @test maxdev < 1e-4
        lj_g, tail_g, coul_g = ref[key]
        # flat routine
        pos, q, mol, σ, ϵ = MCm.site_arrays(sys, ff)
        E, Ex, Es = MCm.rf_energy(pos, q, mol, L, ff.rf)
        @test isapprox((E + Ex + Es) * kJ, coul_g; rtol=2e-5)
        # RigidSystem path: Coulomb = (LJ + Coulomb pair sum) − (LJ-only pair sum), plus the per-molecule constants
        ffn = MCm.SiteForceField([ff.σ_mix[a, a] for a in 1:ff.n_types], [ff.ϵ_mix[a, a] for a in 1:ff.n_types];
                                 rc=ff.rc, coulomb=:none, cutoff=:site, mixing=:geometric, cap_overlaps=true)
        U = MCm.total_pair_energy(sys, ff); ULJ = MCm.total_pair_energy(sys, ffn)
        @test isapprox((U - ULJ + MCm.wolf_constant_energy(sys, ff)) * kJ, coul_g; rtol=2e-5)
        @test isapprox(ULJ * kJ, lj_g; rtol=1e-5)
        @test isapprox(MCm.lrc_energy(sys, ffn) * kJ, tail_g; rtol=1e-3)
        Uv, W = MCm.total_pair_energy_virial(sys, ff)
        @test isapprox(Uv, U; rtol=1e-12)
        # per-molecule RF constant of a neutral molecule = −ke k_rf |μ|²
        t = m.solute; μ = t.sites_body * t.site_q
        @test isapprox(MCm.coulomb_molecule_constant(t, ff), -ff.rf.k_rf * sum(abs2, μ); rtol=1e-10, atol=1e-10)
        εrf == 1.0 && @test abs(MCm.wolf_constant_energy(sys, ff)) < 1e-8      # plain shifted cut-off: constants cancel
    end
    # virial consistency: molecular virial equals −dU/dV under COM scaling for the RF model
    m = MCm.spce_solute_model(:methanol; coulomb=:rf, cutoff=:site, cap_overlaps=false, use_lrc=false)
    sys = MCm.RigidSystem(m.templates, vcat(2, fill(1, 60)), 20.0, 1.0 / 3.8, Xoshiro(5))
    MCm.random_rigid_system!(sys, m.ff)
    U0, W = MCm.total_pair_energy_virial(sys, m.ff)
    V = sys.L^3
    function U_at_scale(s)
        s2 = MCm.RigidSystem(sys.templates, sys.tmpl, sys.L * s, sys.β, Xoshiro(0))
        for i in 1:sys.n_mol
            MCm.set_molecule!(s2, i, sys.com[:, i] .* s, sys.quat[:, i])
        end
        MCm.total_pair_energy(s2, m.ff)
    end
    h = 1e-5
    dUdV = (U_at_scale(1 + h) - U_at_scale(1 - h)) / (V * ((1 + h)^3 - (1 - h)^3))
    @test abs(W / (3V) + dUdV) <= 1e-6 * max(abs(W / (3V)), 1e-8)
end

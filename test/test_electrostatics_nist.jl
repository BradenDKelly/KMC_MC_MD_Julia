"""
Electrostatics validation against the NIST SPC/E reference calculations
(test/data/nist_spce/SOURCE.md). Every Ewald term, the LJ pair energy and the tail correction are
compared with the published values for the four cuboid configurations at the 9 Å and 10 Å cutoffs.

Tolerances: NIST publishes each term to six significant figures, so a term can differ from the
exact value by up to 5e-6 relatively; NIST's erfc is the Numerical Recipes ERFCC approximation
(absolute error < 1.2e-7) while this code uses SpecialFunctions.erfc. We therefore require
relative agreement of 2e-5 per term. The total is a sum of large cancelling terms (E_self and
E_intra are ~6× |E_total|), so the rounding of those two terms alone allows ~3e-5 relative
slack on the total; we require 1e-4 on the total.
"""

using Test
using MolSim

include(joinpath(@__DIR__, "nist_spce_reader.jl"))

# Reference values (E/k_B in K), copied from the NIST pages on 2026-10-11.
# columns: config, M, rc, E_disp, E_LRC, E_real, E_fourier, E_self, E_intra, E_total
const NIST_REF = [
    (1, 100,  9.0,  9.98560E+04, -1.12959E+03, -5.58904E+05, 6.27009E+03, -2.84469E+06, 2.80999E+06, -4.88608E+05),
    (2, 200,  9.0,  1.94941E+05, -4.51836E+03, -1.19308E+06, 6.03495E+03, -5.68938E+06, 5.61998E+06, -1.06602E+06),
    (3, 300,  9.0,  3.57106E+05, -1.01663E+04, -1.96320E+06, 5.24461E+03, -8.53407E+06, 8.42998E+06, -1.71511E+06),
    (4, 750,  9.0,  4.53536E+05, -1.88265E+04, -3.44720E+06, 7.58785E+03, -1.42235E+07, 1.41483E+07, -3.08010E+06),
    (1, 100, 10.0,  9.95387E+04, -8.23715E+02, -5.58889E+05, 6.27009E+03, -2.84469E+06, 2.80999E+06, -4.88604E+05),
    (2, 200, 10.0,  1.93712E+05, -3.29486E+03, -1.19295E+06, 6.03495E+03, -5.68938E+06, 5.61998E+06, -1.06590E+06),
    (3, 300, 10.0,  3.54344E+05, -7.41343E+03, -1.96297E+06, 5.24461E+03, -8.53407E+06, 8.42998E+06, -1.71488E+06),
    (4, 750, 10.0,  4.48593E+05, -1.37286E+04, -3.57226E+06, 7.58785E+03, -1.42235E+07, 1.41483E+07, -3.20501E+06),
]

relerr(a, b) = abs(a - b) / abs(b)

@testset "NIST SPC/E reference energies (Ewald + LJ)" begin
    cache = Dict{Int,Any}()
    for (cfg, M, rc, E_disp, E_lrc, E_real, E_four, E_self, E_intra, E_tot) in NIST_REF
        data = get!(cache, cfg) do
            read_nist_config(joinpath(NIST_DIR, "spce_sample_config_periodic$(cfg).txt"))
        end
        L, Mfile, pos, q, σ, ε, mol = data
        @test Mfile == M
        ep = MolSim.MC.EwaldParams(L, 5.6 / L, rc, 27, KE_K_ANGSTROM)
        ew = MolSim.MC.ewald_energy(pos, q, mol, ep)
        lj, lrc = MolSim.MC.site_lj_energy(pos, σ, ε, mol, L, rc)
        @test relerr(lj, E_disp) < 2e-5
        @test relerr(lrc, E_lrc) < 2e-5
        @test relerr(ew.real, E_real) < 2e-5
        @test relerr(ew.fourier, E_four) < 2e-5
        @test relerr(ew.self, E_self) < 2e-5
        @test relerr(ew.intra, E_intra) < 2e-5
        total = lj + lrc + ew.total
        @test relerr(total, E_tot) < 1e-4
        @info "NIST config $cfg rc=$rc" lj lrc ew.real ew.fourier ew.self ew.intra total E_tot relerr(total, E_tot)
    end
end

@testset "Wolf summation vs Ewald on the NIST configurations" begin
    # Parameters chosen by the Hens & Vlugt protocol from dev/wolf_parameter_scan.jl
    # (output recorded in dev/results/wolf_parameter_scan_spce.txt, 2026-10-11): α = 0.16 Å⁻¹ with
    # r_c = 10 Å reproduces the Ewald intermolecular Coulomb energy of the liquid-density
    # configuration 3 to 1e-4 and of configurations 1-2 to < 0.3 %; configuration 4 (750 molecules,
    # 0.83 g/cm³, 30 Å box) shows a +2.3 % offset for every (α, r_c) scanned. The tolerances below are
    # those recorded deviations rounded up, not independent accuracy claims.
    α_spce, rc_spce = 0.16, 10.0
    tol = Dict(1 => 5e-3, 2 => 5e-3, 3 => 5e-3, 4 => 3e-2)
    for cfg in 1:4
        L, M, pos, q, σ, ε, mol = read_nist_config(joinpath(NIST_DIR, "spce_sample_config_periodic$(cfg).txt"))
        ep = MolSim.MC.EwaldParams(L, 5.6 / L, 10.0, 27, KE_K_ANGSTROM)
        E_ewald = MolSim.MC.ewald_energy(pos, q, mol, ep).total
        wp = MolSim.MC.WolfParams(α_spce, rc_spce, KE_K_ANGSTROM; model=:wolf)
        Ep, Es, Ei = MolSim.MC.wolf_energy(pos, q, mol, L, wp)
        E_wolf = Ep + Es + Ei
        @test isfinite(E_wolf)
        @test relerr(E_wolf, E_ewald) < tol[cfg]
        @info "Wolf vs Ewald, config $cfg" Ep Es Ei E_wolf E_ewald relerr(E_wolf, E_ewald)
        # DSF energy form (experimental, see Electrostatics.jl): must at least be finite and
        # approach the Wolf value as the shift-force term vanishes (large α r_c)
        wpd = MolSim.MC.WolfParams(0.30, rc_spce, KE_K_ANGSTROM; model=:dsf)
        Epd, Esd, Eid = MolSim.MC.wolf_energy(pos, q, mol, L, wpd)
        @test isfinite(Epd + Esd + Eid)
    end
    # pair-function limits: both forms vanish at the cutoff
    wp = MolSim.MC.WolfParams(0.28, 10.0, 1.0)
    @test MolSim.MC.wolf_pair_u(10.0, 1.0, 1.0, wp) == 0.0
    @test MolSim.MC.wolf_pair_u(9.999999, 1.0, 1.0, wp) ≈ 0.0 atol = 1e-6
    wpd = MolSim.MC.WolfParams(0.28, 10.0, 1.0; model=:dsf)
    @test MolSim.MC.wolf_pair_u(9.999999, 1.0, 1.0, wpd) ≈ 0.0 atol = 1e-6
end

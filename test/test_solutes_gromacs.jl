# 1-butanol (OPLS-AA, data/opls/1butanol.itp) and caffeine (LigParGen, data/ligpargen/caffeine) in 500 SPC/E with reaction
# field (ε_rf = ∞, site cut-off 10 Å): the MolSim Hamiltonian (intermolecular LJ + RF, flexible-solute intramolecular terms incl.
# GROMACS's RF exclusion and self terms, rigid-water RF constants) against a GROMACS 2024.3 rerun of the same coordinates
# (test/data/gromacs_<solute>_spce500.gro, 6 decimals; topology from dev/gromacs_solute_itp.jl, the one used for the
# free-energy reference). Tolerance as test_rf_gromacs.jl: 2e-5 relative (GROMACS mixed precision over ~10⁶ pair terms).
using Test, Random
using MolSim

include(joinpath(@__DIR__, "test_solutes_gromacs_loader.jl"))

@testset "Butanol and caffeine in SPC/E (RF): MolSim energy = GROMACS rerun" begin
    ref = Dict{Symbol,Float64}()
    for l in eachline(joinpath(@__DIR__, "data", "solute_gromacs_reference.txt"))
        startswith(l, "#") && continue
        f = split(l); ref[Symbol(f[1])] = parse(Float64, f[2])
    end
    for sym in (:butanol_flex, :caffeine_lpg)
        m, sys, nw = load_solute_water_gro(sym, joinpath(@__DIR__, "data", "gromacs_$(sym)_spce500.gro"))
        ff = m.ff; kJ = m.ϵ_kelvin * 0.0083144626
        @test nw == 500
        U = (MCq.total_pair_energy(sys, ff) + MCq.total_intra_energy(sys, ff) + nw * MCq.coulomb_molecule_constant(m.templates[1], ff)) * kJ
        @test isapprox(U, ref[sym]; rtol=2e-5)
    end
end

@testset "Butanol and caffeine in SPC/E (Ewald): MolSim energy = GROMACS rerun (flexible exclusions, JOURNAL 121)" begin
    # Ewald with flexible solutes: real-space erfc between molecules, Fourier + image term e_m over ALL intramolecular pairs
    # (EwaldRecip recip_corr), vacuum 1/r of the non-excluded and fudge-scaled 1-4 pairs in flex_intra_energy; against
    # GROMACS Ewald and fine PME (0.04 nm, order 8), coulomb-modifier None. Tolerance as above (2e-5 relative).
    ref = Dict{Tuple{Symbol,String},Float64}()
    for l in eachline(joinpath(@__DIR__, "data", "solute_gromacs_ewald_reference.txt"))
        startswith(l, "#") && continue
        f = split(l); ref[(Symbol(f[1]), f[2])] = parse(Float64, f[3])
    end
    α_A = MCq.ewald_alpha_for(10.0, 1e-5)
    for sym in (:butanol_flex, :caffeine_lpg)
        m, sys, nw = load_solute_water_gro(sym, joinpath(@__DIR__, "data", "gromacs_$(sym)_spce500.gro"); coulomb=:ewald, α_A=α_A)
        MCq.attach_ewald!(sys, m.ff; n2max=401, precision=:double)     # |n| ≤ 20: Fourier sum converged far below 1e-5
        kJ = m.ϵ_kelvin * 0.0083144626
        U = (MCq.total_pair_energy(sys, m.ff) + MCq.total_intra_energy(sys, m.ff)) * kJ
        @test isapprox(U, ref[(sym, "ewald_nomod")]; rtol=2e-5)
        @test isapprox(U, ref[(sym, "pmefine_nomod")]; rtol=2e-5)
    end
end

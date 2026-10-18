# 1-butanol (OPLS-AA, data/opls/1butanol.itp) and caffeine (LigParGen, data/ligpargen/caffeine) in 500 SPC/E with reaction
# field (ε_rf = ∞, site cut-off 10 Å): the MolSim Hamiltonian (intermolecular LJ + RF, flexible-solute intramolecular terms incl.
# GROMACS's RF exclusion and self terms, rigid-water RF constants) against a GROMACS 2024.3 rerun of the same coordinates
# (test/data/gromacs_<solute>_spce500.gro, 6 decimals; topology from dev/gromacs_solute_itp.jl, the one used for the
# free-energy reference). Tolerance as test_rf_gromacs.jl: 2e-5 relative (GROMACS mixed precision over ~10⁶ pair terms).
using Test, Random
using MolSim
const MCq = MolSim.MC

function load_solute_water_gro(sym, path)
    m = MCq.spce_solute_model(sym; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=10.0)
    σA = m.σ_angstrom; lines = readlines(path); nat = parse(Int, strip(lines[2]))
    box = parse.(Float64, split(lines[end])[1:3]); L = box[1] * 10 / σA
    X = zeros(3, nat)
    for k in 1:nat
        l = lines[2 + k]; X[:, k] .= parse.(Float64, (l[21:31], l[32:42], l[43:53])) .* (10 / σA)
    end
    tw, ts = m.templates[1], m.templates[2]; nw = (nat - ts.n_sites) ÷ 3
    sys = MCq.RigidSystem(m.templates, vcat(2, fill(1, nw)), L, 1.0, Xoshiro(1))
    Y = X[:, 1:ts.n_sites]
    for a in 2:ts.n_sites, d in 1:3; Y[d, a] = Y[d, 1] + MCq._min_image(Y[d, a] - Y[d, 1], L); end
    MCq.set_conformation!(sys, 1, Y); MCq.set_molecule!(sys, 1, (Y * ts.site_mass) ./ sum(ts.site_mass), sys.quat[:, 1])
    for w in 1:nw
        W = X[:, ts.n_sites + 3(w - 1) .+ (1:3)]
        for a in 2:3, d in 1:3; W[d, a] = W[d, 1] + MCq._min_image(W[d, a] - W[d, 1], L); end
        MCq.molecule_from_sites!(sys, 1 + w, W)
    end
    return m, sys, nw
end

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

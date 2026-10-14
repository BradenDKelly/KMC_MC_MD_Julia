# Flexible OPLS-AA molecules built from the GROMACS topologies (OPLSTopology.jl): the builder reproduces the
# hand-coded methanol Hamiltonian exactly, and the intramolecular reaction-field energy of ethanol and 1-propanol
# equals GROMACS single points (test/data/opls_itp_rf_reference.txt) to the mixed-precision level (1e-3 kJ/mol).
using Test
using MolSim
const MCm = MolSim.MC

@testset "OPLS-AA itp builder: methanol = hand-coded; ethanol, 1-propanol = GROMACS (RF)" begin
    m = MCm.opls_itp_molecule(joinpath(MCm.OPLS_FFDIR, "methanol.itp"))
    bt0, types0, q0, σ0, ϵ0 = MCm.opls_methanol_bonded()
    @test m.types == types0 && m.q == q0 && m.lj == collect(zip(σ0, ϵ0))
    σm = [sqrt(σ0[a] * σ0[b]) for a in 1:3, b in 1:3]; ϵm = [sqrt(ϵ0[a] * ϵ0[b]) for a in 1:3, b in 1:3]
    X = m.X0 .+ 0.003 .* [sin(7i + 3j) for i in 1:3, j in 1:6]
    @test MCm.intra_energy(m.bt, X, m.types, m.q, σm, ϵm, MCm.KE_KJMOL_NM) ≈ MCm.intra_energy(bt0, X, types0, q0, σm, ϵm, MCm.KE_KJMOL_NM) rtol=1e-12
    models = Dict("ethanol" => MCm.spce_solute_model(:ethanol_flex; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=10.0),
                  "1propanol" => MCm.spce_solute_model(:propanol_flex; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=10.0))
    @test models["ethanol"].solute.n_sites == 9 && models["1propanol"].solute.n_sites == 12
    @test length(models["1propanol"].solute.bonded.nonbonded) == 15            # pairs beyond 1-4 (RF pair term exercised)
    n = 0
    for line in eachline(joinpath(@__DIR__, "data", "opls_itp_rf_reference.txt"))
        startswith(line, "#") && continue
        f = split(line); mdl = models[f[1]]; U_g = parse(Float64, f[3])
        v = parse.(Float64, f[4:end]); ns = length(v) ÷ 3
        X = reshape(v, 3, ns) .* 10 ./ mdl.σ_angstrom
        U = MCm.flex_intra_energy(mdl.solute, mdl.ff, X) * mdl.ϵ_kelvin * 0.0083144626
        @test isapprox(U, U_g; atol=1e-3)
        n += 1
    end
    @test n == 12
end

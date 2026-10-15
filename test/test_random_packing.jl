# Random starting configurations place the largest molecules first. Regression: 1-propanol in 500 SPC/E at
# 1000 kg/m³ (the driver's seed-1 setup) failed with "could not place molecule 501" when the solute went last.
using Test
using MolSim
using Random
const MCp = MolSim.MC

@testset "random_rigid_system!: largest molecule first (1-propanol in 500 SPC/E, seed 1)" begin
    m = MCp.spce_solute_model(:propanol_flex; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=10.0)
    ff = first(MCp.kT_overlap_cap(m.ff, m.templates, 300.0 / m.ϵ_kelvin; U_kT=100.0))
    N = 500
    ρstar = 1000.0 * 1e3 / 18.0153 * 6.02214129e23 / 1e30 * m.σ_angstrom^3
    L = cbrt(N / ρstar)
    sys = MCp.RigidSystem(m.templates, vcat(fill(1, N), 2), L, 300.0 / m.ϵ_kelvin |> inv, Xoshiro(1))
    MCp.random_rigid_system!(sys, ff)
    capped = 0
    for i in 1:sys.n_mol, j in i+1:sys.n_mol
        MCp.pair_energy(sys, ff, i, j) >= minimum(ff.ucap_mix) * 0.999 && (capped += 1)
    end
    @test capped == 0
end

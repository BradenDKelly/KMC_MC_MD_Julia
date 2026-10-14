# Bennett estimator from the eKMC jump chain (DESIGN §10.3): the landing energies of a species are Widom
# insertion samples into the (N−1)-system, so landing-Widom = MC Widom, and the Bennett acceptance ratio over
# landings + time-weighted energies agrees with both; the removal estimator from the same accumulator equals
# the species μ of the run.
using Test, Random
using MolSim
const MCm = MolSim.MC

@testset "Bennett/landing-Widom from the eKMC jump chain (LJ solvent + flexible dimer solute)" begin
    sites = zeros(3, 2); sites[1, 1] = -0.3; sites[1, 2] = 0.3
    solv = MCm.RigidTemplate("LJ2", sites, [1, 1], [0.0, 0.0], [1.0, 1.0])
    bt = MCm.BondedTerms(2; bonds=[MCm.HarmonicBond(1, 2, 0.6, 400.0)])
    dimer = MCm.RigidTemplate("flexdimer", sites, [2, 2], [0.0, 0.0], [1.0, 1.0]; bonded=bt)
    ff = MCm.SiteForceField([1.0, 1.1], [1.0, 1.0]; rc=2.5, coulomb=:none)        # ε_solute = ε_solvent: the solute relocates as often as a solvent molecule (many landings; the identity is tested, not the hard case)
    N = 32; T = 1.5; β = 1 / T; L = cbrt(N / 0.4); Nt = N + 1
    res = MCm.build_reservoir([solv, dimer], 2, ff, β, Xoshiro(3); M=5000, burnin_sweeps=500, stride_sweeps=10, δ=0.05)
    # Widom reference: ideal-gas dimer into the PURE solvent (Metropolis)
    sys0 = MCm.RigidSystem([solv, dimer], fill(1, N), L, β, Xoshiro(11)); MCm.random_rigid_system!(sys0, ff)
    MCm.minimize_rigid!(sys0, ff; nsteps=200)
    st = MCm.RigidMCStats()
    for _ in 1:300, _ in 1:N
        MCm.mc_translate!(sys0, ff, rand(sys0.rng, 1:N), 0.15, st); MCm.mc_rotate!(sys0, ff, rand(sys0.rng, 1:N), 0.4, st)
    end
    mc0 = MCm.run_rigid_mc!(sys0, ff; sweeps=3000, δ=0.15, δθ=0.4, sample_every=1, block_size=100,
                            widom_template=2, widom_per_sample=50, reservoirs=[nothing, res])
    w0, ew0 = MCm.mean(mc0.widom), MCm.stderr(mc0.widom)
    μ_w, eμ_w = -T * log(w0), T * ew0 / w0
    # solvent + 1 solute, stock eKMC with the Bennett accumulator on the solute species
    sys = MCm.RigidSystem([solv, dimer], vcat(fill(1, N), 2), L, β, Xoshiro(11)); MCm.random_rigid_system!(sys, ff)
    MCm.minimize_rigid!(sys, ff; nsteps=200)
    st = MCm.RigidMCStats()
    for _ in 1:300, _ in 1:Nt
        MCm.mc_translate!(sys, ff, rand(sys.rng, 1:Nt), 0.15, st); MCm.mc_rotate!(sys, ff, rand(sys.rng, 1:Nt), 0.4, st)
        MCm.mc_site_move!(sys, ff, Nt, 0.05, st); MCm.mc_conformer_swap!(sys, ff, Nt, res, st)
    end
    events = 1_500_000
    @test_throws ErrorException MCm.run_rigid_ekmc!(deepcopy(sys), ff; events=100, block_events=50, reservoirs=[nothing, res], bar_species=3)
    ek = MCm.run_rigid_ekmc!(deepcopy(sys), ff; events=events, sample_every=1, block_events=events ÷ 25, species_mu=true,
                             reservoirs=[nothing, res], bar_species=2)
    b = ek.bar
    @test b isa MCm.BennettAccumulator && MCm.nblocks(b) == 25
    d = MCm.bennett_diagnostics(b)
    @test d.n_landings > 1000 && isfinite(d.n_eff_time) && 0 < d.overlap <= 1
    r2 = ek.RN_species[2]
    μ_rm, eμ_rm = T * log(MCm.ratio_mean(r2)), T * MCm.ratio_jackknife_stderr(r2; f=log)
    μ_rm2, eμ_rm2 = MCm.removal_mu(b, T)
    @test isapprox(μ_rm, μ_rm2; rtol=1e-8)                 # same sums as the species estimator of the run
    μ_lw, eμ_lw = MCm.landing_widom_mu(b, T)
    μ_bar, eμ_bar = MCm.bennett_mu(b, T)
    μ_bar0, _ = MCm.bennett_mu(b, T; M=0.0)                 # any M is consistent
    @info "Bennett test (reduced units)" μ_w eμ_w μ_lw eμ_lw μ_rm eμ_rm μ_bar eμ_bar μ_bar0 d
    within3σ(a, ea, c, ec) = abs(a - c) <= 3 * sqrt(ea^2 + ec^2)
    @test within3σ(μ_lw, eμ_lw, μ_w, eμ_w)                  # landings ARE Widom samples into the pure solvent
    @test within3σ(μ_bar, eμ_bar, μ_w, eμ_w)
    @test within3σ(μ_bar, eμ_bar, μ_rm, eμ_rm)
    @test abs(μ_bar - μ_bar0) <= 3 * eμ_bar
    @test isfinite(eμ_bar) && eμ_bar <= 2 * max(eμ_lw, eμ_rm)   # sanity only: which side is better depends on the case (JOURNAL entry 37)
end

# Coupling-ladder eKMC (DESIGN §10.4): μ from rung occupancies and from MBAR agree with the Widom reference and with
# the stock removal estimator; U at the coupled rung is the stock U; the ladder makes a strongly bound solute switch
# between bound and absent far more often than stock relocation does.
using Test, Random
using MolSim
const MCm = MolSim.MC

within3σ(a, ea, c, ec) = abs(a - c) <= 3 * sqrt(ea^2 + ec^2)

function dimer_system(ε_s, N, T, ρ, seed)
    sites = zeros(3, 2); sites[1, 1] = -0.3; sites[1, 2] = 0.3
    solv = MCm.RigidTemplate("LJ2", sites, [1, 1], [0.0, 0.0], [1.0, 1.0])
    bt = MCm.BondedTerms(2; bonds=[MCm.HarmonicBond(1, 2, 0.6, 400.0)])
    dimer = MCm.RigidTemplate("flexdimer", sites, [2, 2], [0.0, 0.0], [1.0, 1.0]; bonded=bt)
    ff = MCm.SiteForceField([1.0, 1.1], [1.0, ε_s]; rc=2.5, coulomb=:none)
    β = 1 / T; L = cbrt(N / ρ); Nt = N + 1
    res = MCm.build_reservoir([solv, dimer], 2, ff, β, Xoshiro(3); M=5000, burnin_sweeps=500, stride_sweeps=10, δ=0.05)
    sys = MCm.RigidSystem([solv, dimer], vcat(fill(1, N), 2), L, β, Xoshiro(seed)); MCm.random_rigid_system!(sys, ff)
    MCm.minimize_rigid!(sys, ff; nsteps=200)
    st = MCm.RigidMCStats()
    for _ in 1:300, _ in 1:Nt
        MCm.mc_translate!(sys, ff, rand(sys.rng, 1:Nt), 0.15, st); MCm.mc_rotate!(sys, ff, rand(sys.rng, 1:Nt), 0.4, st)
        MCm.mc_site_move!(sys, ff, Nt, 0.05, st); MCm.mc_conformer_swap!(sys, ff, Nt, res, st)
    end
    return (; solv, dimer, ff, res, sys, L, β, Nt)
end

@testset "Coupling ladder: μ (occupancy, MBAR) = Widom = stock; U unchanged" begin
    N = 32; T = 1.5; ρ = 0.4
    S = dimer_system(1.0, N, T, ρ, 11)
    # Widom reference into the pure solvent
    sys0 = MCm.RigidSystem([S.solv, S.dimer], fill(1, N), S.L, S.β, Xoshiro(5)); MCm.random_rigid_system!(sys0, S.ff)
    MCm.minimize_rigid!(sys0, S.ff; nsteps=200)
    st = MCm.RigidMCStats()
    for _ in 1:300, _ in 1:N
        MCm.mc_translate!(sys0, S.ff, rand(sys0.rng, 1:N), 0.15, st); MCm.mc_rotate!(sys0, S.ff, rand(sys0.rng, 1:N), 0.4, st)
    end
    mc0 = MCm.run_rigid_mc!(sys0, S.ff; sweeps=3000, δ=0.15, δθ=0.4, sample_every=1, block_size=100,
                            widom_template=2, widom_per_sample=50, reservoirs=[nothing, S.res])
    w0, ew0 = MCm.mean(mc0.widom), MCm.stderr(mc0.widom)
    μ_w, eμ_w = -T * log(w0), T * ew0 / w0
    events = 3_000_000
    stock = MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=events, sample_every=1, block_events=events ÷ 25, species_mu=true,
                                reservoirs=[nothing, S.res])
    r2 = stock.RN_species[2]
    μ_st, eμ_st = T * log(MCm.ratio_mean(r2)), T * MCm.ratio_jackknife_stderr(r2; f=log)
    ν = 1.0 * MCm.estimate_relocation_rate(deepcopy(S.sys), S.ff; events=20_000)   # hops as frequent (in kMC time) as relocations; hops are O(N) and cheap
    @test_throws ErrorException MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=100, block_events=50, reservoirs=[nothing, S.res], ladder_solute=S.Nt)   # ν required
    @test_throws ErrorException MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=100, block_events=50, reservoirs=[nothing, S.res], ladder_solute=S.Nt, ladder_nu=ν, Pext=1.0)
    lad = MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=events, sample_every=1, block_events=events ÷ 25, species_mu=true,
                              reservoirs=[nothing, S.res], ladder_solute=S.Nt, ladder_nu=ν, ladder_wl_events=events ÷ 3,
                              ladder_eta_ramp=μ_st / T, debug_check_every=events ÷ 4)
    Ld = lad.ladder
    sm = MCm.ladder_summary(Ld)
    @test Ld.wl_f == 0.0 && Ld.ev_frozen > 0                    # η frozen before production
    @test all(sm.t_rung .> 0) && sm.n_recouple > 20
    μ_oc, eμ_oc = MCm.ladder_mu_occupancy(Ld, T)
    μ_mb, eμ_mb = MCm.ladder_mu_mbar(Ld, T)
    rK = lad.RN_species[2]
    μ_K, eμ_K = T * log(MCm.ratio_mean(rK)), T * MCm.ratio_jackknife_stderr(rK; f=log)
    @info "ladder (easy solute)" μ_w eμ_w μ_st eμ_st μ_oc eμ_oc μ_mb eμ_mb μ_K eμ_K length(lad.U.block_means) MCm.mean(lad.U) MCm.stderr(lad.U) MCm.mean(stock.U) MCm.stderr(stock.U) sm.frac_rung sm.n_recouple sm.η sm.wl_stages
    @test within3σ(μ_oc, eμ_oc, μ_w, eμ_w)
    @test within3σ(μ_mb, eμ_mb, μ_w, eμ_w)
    @test within3σ(μ_oc, eμ_oc, μ_st, eμ_st)
    # the coupled rung holds ≈ 1/K of the kMC time but far fewer events (bursts crowd the soft rungs), so its blocks are short
    @test length(lad.U.block_means) >= 5 && length(rK.block_sum_x) >= 5
    @test within3σ(μ_K, eμ_K, μ_st, eμ_st)                      # stock estimator restricted to the coupled rung
    @test within3σ(MCm.mean(lad.U), MCm.stderr(lad.U), MCm.mean(stock.U), MCm.stderr(stock.U))
    @test within3σ(MCm.mean(lad.P), MCm.stderr(lad.P), MCm.mean(stock.P), MCm.stderr(stock.P))
end

@testset "Coupling ladder: production re-flattening from bad η (no Wang–Landau)" begin
    N = 32; T = 1.5; ρ = 0.4
    S = dimer_system(1.0, N, T, ρ, 11)
    events = 2_000_000
    stock = MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=events, sample_every=1, block_events=events ÷ 25, species_mu=true,
                                reservoirs=[nothing, S.res])
    r2 = stock.RN_species[2]
    μ_st, eμ_st = T * log(MCm.ratio_mean(r2)), T * MCm.ratio_jackknife_stderr(r2; f=log)
    ν = MCm.estimate_relocation_rate(deepcopy(S.sys), S.ff; events=20_000)
    K = length(MCm.LADDER_LAMBDA_DEFAULT)
    lad = MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=events, sample_every=1, block_events=events ÷ 50, species_mu=true,
                              reservoirs=[nothing, S.res], ladder_solute=S.Nt, ladder_nu=ν, ladder_eta=zeros(K), ladder_eta_ramp=μ_st / T,
                              ladder_adapt_blocks=true)
    Ld = lad.ladder; sm = MCm.ladder_summary(Ld)
    @test length(Ld.η_blocks) == MCm.nblocks(Ld) && Ld.η_blocks[1] == zeros(K) && Ld.η != zeros(K)
    μ_oc, eμ_oc = MCm.ladder_mu_occupancy(Ld, T)
    μ_mb, eμ_mb = MCm.ladder_mu_mbar(Ld, T)
    # flatness improves: spread of the rung times in the last 5 blocks vs the first 5
    spread(bs) = (t = sum(Ld.t_rung[b] for b in bs); maximum(t) / max(minimum(t), eps()))
    nb = MCm.nblocks(Ld)
    @info "ladder adaptive η (easy solute, η₀ = 0, no WL)" μ_st eμ_st μ_oc eμ_oc μ_mb eμ_mb spread(1:5) spread(nb-4:nb) sm.η
    @test spread(nb-4:nb) < spread(1:5)
    @test within3σ(μ_oc, eμ_oc, μ_st, eμ_st)
    @test within3σ(μ_mb, eμ_mb, μ_st, eμ_st)
end

@testset "Coupling ladder + MC kernel (full sweeps and local sweeps around the fractional molecule)" begin
    N = 32; T = 1.5; ρ = 0.4
    S = dimer_system(1.0, N, T, ρ, 11)
    events = 1_500_000
    stock = MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=events, sample_every=1, block_events=events ÷ 25, species_mu=true,
                                reservoirs=[nothing, S.res])
    r2 = stock.RN_species[2]
    μ_st, eμ_st = T * log(MCm.ratio_mean(r2)), T * MCm.ratio_jackknife_stderr(r2; f=log)
    ν = MCm.estimate_relocation_rate(deepcopy(S.sys), S.ff; events=20_000)
    for (name, rloc) in (("full sweeps", 0.0), ("local sweeps r=1.8", 1.8))
        lad = MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=events, sample_every=1, block_events=events ÷ 25, species_mu=true,
                                  reservoirs=[nothing, S.res], ladder_solute=S.Nt, ladder_nu=ν, ladder_wl_events=events ÷ 3,
                                  ladder_eta_ramp=μ_st / T, mc_kernel_rate=ν, mc_δ=0.15, mc_δθ=0.4, mc_local_radius=rloc,
                                  debug_check_every=events ÷ 5)
        Ld = lad.ladder; sm = MCm.ladder_summary(Ld)
        μ_oc, eμ_oc = MCm.ladder_mu_occupancy(Ld, T)
        μ_mb, eμ_mb = MCm.ladder_mu_mbar(Ld, T)
        @info "ladder + MC kernel ($name)" μ_st eμ_st μ_oc eμ_oc μ_mb eμ_mb lad.mc_events lad.mc_acceptance sm.n_recouple
        @test lad.mc_events > 100 && 0.05 < lad.mc_acceptance < 0.95
        @test within3σ(μ_oc, eμ_oc, μ_st, eμ_st)
        @test within3σ(μ_mb, eμ_mb, μ_st, eμ_st)
        @test within3σ(MCm.mean(lad.U), MCm.stderr(lad.U), MCm.mean(stock.U), MCm.stderr(stock.U))
    end
end

@testset "Coupling ladder: a strongly bound solute switches bound ↔ absent; occupancy = MBAR" begin
    N = 32; T = 1.5; ρ = 0.4
    S = dimer_system(2.0, N, T, ρ, 11)                           # stock: ~90 solute landings per 1.5e6 events (JOURNAL entry 37)
    events = 1_000_000
    stock = MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=events, sample_every=1, block_events=events ÷ 25, species_mu=true,
                                reservoirs=[nothing, S.res], bar_species=2)
    n_land_stock = MCm.bennett_diagnostics(stock.bar).n_landings
    r2 = stock.RN_species[2]
    ν = 1.0 * MCm.estimate_relocation_rate(deepcopy(S.sys), S.ff; events=20_000)
    lad = MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=events, sample_every=1, block_events=events ÷ 25, species_mu=true,
                              reservoirs=[nothing, S.res], ladder_solute=S.Nt, ladder_nu=ν, ladder_wl_events=events ÷ 3,
                              ladder_eta_ramp=T * log(MCm.ratio_mean(r2)) / T)
    Ld = lad.ladder; sm = MCm.ladder_summary(Ld)
    μ_oc, eμ_oc = MCm.ladder_mu_occupancy(Ld, T)
    μ_mb, eμ_mb = MCm.ladder_mu_mbar(Ld, T)
    @info "ladder (bound solute)" n_land_stock sm.n_recouple sm.frac_rung sm.η μ_oc eμ_oc μ_mb eμ_mb T * log(MCm.ratio_mean(r2)) T * MCm.ratio_jackknife_stderr(r2; f=log)
    @test sm.n_recouple > 10 * max(n_land_stock, 1)
    @test all(sm.t_rung .> 0)
    @test within3σ(μ_oc, eμ_oc, μ_mb, eμ_mb)
end

@testset "Coupling ladder with the λ = 0 hub (direct jumps 1 ↔ m, no relocation of the fractional molecule at m ≥ 3)" begin
    N = 32; T = 1.5; ρ = 0.4
    for ε_s in (1.0, 2.0)
        S = dimer_system(ε_s, N, T, ρ, 11)
        events = 2_000_000
        stock = MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=events, sample_every=1, block_events=events ÷ 25, species_mu=true,
                                    reservoirs=[nothing, S.res])
        r2 = stock.RN_species[2]
        μ_st, eμ_st = T * log(MCm.ratio_mean(r2)), T * MCm.ratio_jackknife_stderr(r2; f=log)
        ν = MCm.estimate_relocation_rate(deepcopy(S.sys), S.ff; events=20_000)
        lad = MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=events, sample_every=1, block_events=events ÷ 25, species_mu=true,
                                  reservoirs=[nothing, S.res], ladder_solute=S.Nt, ladder_nu=ν, ladder_wl_events=events ÷ 3,
                                  ladder_eta_ramp=μ_st / T, ladder_jump=true, debug_check_every=events ÷ 4)
        Ld = lad.ladder; sm = MCm.ladder_summary(Ld)
        μ_oc, eμ_oc = MCm.ladder_mu_occupancy(Ld, T)
        μ_mb, eμ_mb = MCm.ladder_mu_mbar(Ld, T)
        rK = lad.RN_species[2]
        μ_K, eμ_K = T * log(MCm.ratio_mean(rK)), T * MCm.ratio_jackknife_stderr(rK; f=log)
        @info "ladder + λ=0 hub (ε_s = $ε_s)" μ_st eμ_st μ_oc eμ_oc μ_mb eμ_mb μ_K eμ_K sm.n_jump sm.n_hop sm.frac_rung sm.η
        @test Ld.jump && all(Ld.csol[3:end] .== 0.0) && Ld.csol[2] > 0.0
        @test sum(sm.n_jump[3:end]) > 50 && sm.n_jump[1] > 50 && all(sm.t_rung .> 0)
        @test within3σ(μ_oc, eμ_oc, μ_st, eμ_st)
        @test within3σ(μ_mb, eμ_mb, μ_st, eμ_st)
        @test within3σ(μ_K, eμ_K, μ_st, eμ_st)                  # removal estimator at the coupled rung (c = 0: exp(βu) used directly)
        @test within3σ(MCm.mean(lad.U), MCm.stderr(lad.U), MCm.mean(stock.U), MCm.stderr(stock.U))
    end
end

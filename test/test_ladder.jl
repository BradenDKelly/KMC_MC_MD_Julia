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

# time-averaged intramolecular energy of the flexible solute by Metropolis MC (block standard error, 20 blocks)
function mc_uintra(S, sweeps)
    sys = deepcopy(S.sys); st = MCm.RigidMCStats(); Ui = Float64[]
    for sw in 1:sweeps
        for _ in 1:S.Nt
            MCm.mc_translate!(sys, S.ff, rand(sys.rng, 1:S.Nt), 0.15, st); MCm.mc_rotate!(sys, S.ff, rand(sys.rng, 1:S.Nt), 0.4, st)
        end
        MCm.mc_site_move!(sys, S.ff, S.Nt, 0.05, st); MCm.mc_conformer_swap!(sys, S.ff, S.Nt, S.res, st)
        sw > sweeps ÷ 10 && push!(Ui, MCm.intra_energy_molecule(sys, S.ff, S.Nt))
    end
    nb = 20; n = length(Ui) ÷ nb; bm = [sum(Ui[(b-1)*n+1:b*n]) / n for b in 1:nb]; m = sum(bm) / nb
    return sum(Ui) / length(Ui), sqrt(sum(abs2, bm .- m) / (nb - 1) / nb)
end

@testset "Coupling ladder + rejection-free local moves of the fractional molecule (with and without the λ = 0 hub)" begin
    N = 32; T = 1.5; ρ = 0.4
    for ε_s in (1.0, 2.0)
        S = dimer_system(ε_s, N, T, ρ, 11)
        events = 2_000_000
        stock = MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=events, sample_every=1, block_events=events ÷ 25, species_mu=true,
                                    reservoirs=[nothing, S.res])
        r2 = stock.RN_species[2]
        μ_st, eμ_st = T * log(MCm.ratio_mean(r2)), T * MCm.ratio_jackknife_stderr(r2; f=log)
        ν = MCm.estimate_relocation_rate(deepcopy(S.sys), S.ff; events=20_000)
        # ⟨U_intra⟩ reference: Metropolis with site moves and conformer swaps (stock eKMC changes the conformer only at
        # relocations of the solute, which are rare at ε_s = 2: stock gave 0.40 ± 0.08 vs MC 0.80 ± 0.04, job 207603)
        Ui_mc, eUi_mc = mc_uintra(S, 20_000)
        for hub in (true, false)
            lad = MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=events, sample_every=1, block_events=events ÷ 25, species_mu=true,
                                      reservoirs=[nothing, S.res], ladder_solute=S.Nt, ladder_nu=ν, ladder_wl_events=events ÷ 3,
                                      ladder_eta_ramp=μ_st / T, ladder_jump=hub, ladder_local=true, debug_check_every=events ÷ 4)
            Ld = lad.ladder; sm = MCm.ladder_summary(Ld); lm = lad.lmoves
            μ_oc, eμ_oc = MCm.ladder_mu_occupancy(Ld, T)
            μ_mb, eμ_mb = MCm.ladder_mu_mbar(Ld, T)
            @info "ladder + L-moves (ε_s = $ε_s, hub = $hub)" μ_st eμ_st μ_oc eμ_oc μ_mb eμ_mb lm.n_move lm.n_move_rung lm.n_refresh sm.frac_rung sm.res_frac sm.pose_frac
            @test all(lm.n_move .> 100) && all(lm.n_move_rung[2:end] .> 0) && lm.n_move_rung[1] == 0
            @test all(sm.t_rung .> 0)
            @test within3σ(μ_oc, eμ_oc, μ_st, eμ_st)
            @test within3σ(μ_mb, eμ_mb, μ_st, eμ_st)
            @test within3σ(MCm.mean(lad.U), MCm.stderr(lad.U), MCm.mean(stock.U), MCm.stderr(stock.U))
            @test within3σ(MCm.mean(lad.U_intra), MCm.stderr(lad.U_intra), Ui_mc, eUi_mc)
        end
    end
end

@testset "Coupling ladder + hub + solute and solvent L-moves: μ = stock, U = stock" begin
    N = 32; T = 1.5; ρ = 0.4
    S = dimer_system(2.0, N, T, ρ, 11)
    events = 1_000_000
    stock = MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=2 * events, sample_every=1, block_events=events ÷ 12, species_mu=true,
                                reservoirs=[nothing, S.res])
    r2 = stock.RN_species[2]
    μ_st, eμ_st = T * log(MCm.ratio_mean(r2)), T * MCm.ratio_jackknife_stderr(r2; f=log)
    ν = MCm.estimate_relocation_rate(deepcopy(S.sys), S.ff; events=20_000)
    lad = MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=events, sample_every=1, block_events=events ÷ 25, species_mu=true,
                              reservoirs=[nothing, S.res], ladder_solute=S.Nt, ladder_nu=ν, ladder_wl_events=events ÷ 3,
                              ladder_eta_ramp=μ_st / T, ladder_jump=true, ladder_local=true, local_solvent=true, solvent_nu=ν / N,
                              mc_δ=0.15, mc_δθ=0.4, debug_check_every=events ÷ 4)
    Ld = lad.ladder; sm = MCm.ladder_summary(Ld)
    μ_oc, eμ_oc = MCm.ladder_mu_occupancy(Ld, T); μ_mb, eμ_mb = MCm.ladder_mu_mbar(Ld, T)
    @info "ladder + hub + L-moves + solvent L-moves (ε_s = 2)" μ_st eμ_st μ_oc eμ_oc μ_mb eμ_mb lad.slmoves.n_move lad.lmoves.n_move sm.frac_rung
    @test sum(lad.slmoves.n_move) > 1000 && all(sm.t_rung .> 0)
    @test within3σ(μ_oc, eμ_oc, μ_st, eμ_st)
    @test within3σ(μ_mb, eμ_mb, μ_st, eμ_st)
    @test within3σ(MCm.mean(lad.U), MCm.stderr(lad.U), MCm.mean(stock.U), MCm.stderr(stock.U))
end

@testset "Coupling ladder: 1/t Wang–Landau schedule (Belardinelli & Pereyra 2007)" begin
    λ = MCm.LADDER_LAMBDA_DEFAULT; K = length(λ)
    # switch at a halving: flat visits with dt = 1; the flatness check at event 1000 halves f to 0.5, and with
    # ν = 0.002 the 1/t value is K/(ν T) = 10/(0.002·1000) = 5 ≥ 0.5, so the run switches to f = 1/t there
    ν = 0.002
    L = MCm.CouplingLadder(1, λ, 100; ν=ν, wl_events=2000, wl_1t=true)
    @test L.wl_1t && L.wl_f == 1.0 && !MCm.ladder_production(L)
    for ev in 1:999
        MCm.wl_update!(L, ev, 1.0, (ev - 1) % K + 1)
    end
    @test !L.wl_phase1t && L.wl_f == 1.0 && L.η[end] == 0.0
    MCm.wl_update!(L, 1000, 1.0, 10)
    @test L.wl_phase1t && L.wl_stages == 1 && L.wl_f ≈ K / (ν * 1000.0)
    @test MCm.ladder_production(L) && MCm.adapting(L) && L.ev_frozen == 1000     # production starts at the switch, η keeps adapting
    for ev in 1001:1999
        MCm.wl_update!(L, ev, 1.0, (ev - 1) % K + 1)
    end
    @test L.wl_f ≈ K / (ν * 1999.0) && MCm.ladder_production(L)
    for ev in 2000:2001                                                         # past the budget: unchanged schedule, no freeze
        MCm.wl_update!(L, ev, 1.0, (ev - 1) % K + 1)
    end
    @test L.wl_f ≈ K / (ν * 2001.0) && MCm.adapting(L) && L.ev_frozen == 1000
    # no halving before the budget: f = 1/t from the budget on
    L2 = MCm.CouplingLadder(1, λ, 100; ν=2.0, wl_events=50, wl_1t=true)
    for ev in 1:49
        MCm.wl_update!(L2, ev, 1.0, (ev - 1) % K + 1)
    end
    @test !L2.wl_phase1t && L2.wl_f == 1.0 && !MCm.ladder_production(L2)
    MCm.wl_update!(L2, 50, 1.0, 10)
    @test L2.wl_phase1t && L2.wl_f ≈ K / (2.0 * 50.0) && MCm.ladder_production(L2) && L2.ev_frozen == 50
    @test_throws ErrorException MCm.CouplingLadder(1, λ, 100; ν=2.0, wl_events=50, wl_1t=true, adapt_blocks=true)
end

@testset "Coupling ladder: 1/t Wang–Landau adapting through production; μ (occupancy, MBAR) = stock" begin
    N = 32; T = 1.5; ρ = 0.4
    S = dimer_system(1.0, N, T, ρ, 11)
    events = 2_000_000
    stock = MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=events, sample_every=1, block_events=events ÷ 25, species_mu=true,
                                reservoirs=[nothing, S.res])
    r2 = stock.RN_species[2]
    μ_st, eμ_st = T * log(MCm.ratio_mean(r2)), T * MCm.ratio_jackknife_stderr(r2; f=log)
    ν = MCm.estimate_relocation_rate(deepcopy(S.sys), S.ff; events=20_000)
    K = length(MCm.LADDER_LAMBDA_DEFAULT)
    # start from flat η = 0 (far from the converged weights) so the adaptation has work to do
    lad = MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=events, sample_every=1, block_events=events ÷ 25, species_mu=true,
                              reservoirs=[nothing, S.res], ladder_solute=S.Nt, ladder_nu=ν, ladder_wl_events=events ÷ 4,
                              ladder_eta=zeros(K), ladder_wl_1t=true, debug_check_every=events ÷ 4)
    Ld = lad.ladder; sm = MCm.ladder_summary(Ld)
    @test Ld.wl_phase1t && MCm.adapting(Ld) && Ld.wl_f ≈ K / (Ld.ν * Ld.wl_T)    # still adapting at the end, on the 1/t schedule
    @test Ld.ev_frozen > 0 && MCm.nblocks(Ld) >= 15 && all(sm.t_rung .> 0)
    μ_oc, eμ_oc = MCm.ladder_mu_occupancy(Ld, T)
    μ_mb, eμ_mb = MCm.ladder_mu_mbar(Ld, T)
    @info "ladder 1/t Wang–Landau (easy solute, η₀ = 0)" μ_st eμ_st μ_oc eμ_oc μ_mb eμ_mb sm.frac_rung sm.η sm.wl_stages Ld.wl_f
    @test within3σ(μ_oc, eμ_oc, μ_st, eμ_st)
    @test within3σ(μ_mb, eμ_mb, μ_st, eμ_st)
    @test within3σ(MCm.mean(lad.U), MCm.stderr(lad.U), MCm.mean(stock.U), MCm.stderr(stock.U))
end

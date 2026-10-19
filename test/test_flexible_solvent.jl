# Flexible solvent L-moves (DESIGN §14): every molecule of a liquid of flexible molecules carries translation, rotation,
# single-site and torsion slots. V1: involution of every slot kind and exact incremental bookkeeping (RF, Ewald, Wolf);
# V2: flexible LJ-dimer liquid, eKMC vs Metropolis MC with site moves (U, P, ⟨U_intra⟩ within 3 combined standard errors).
using Test, Random
using MolSim
const MCf = MolSim.MC

fs_within3σ(a, ea, b, eb) = abs(a - b) <= 3 * sqrt(ea^2 + eb^2)

"Dense fluid of OPLS-AA ethanol (flexible, 9 sites) at `density` g/cm³ (random start without overlaps, then relaxed)."
function ethanol_liquid(N, seed; coulomb=:rf, rc=7.0, density=0.4)
    α = MCf.ewald_alpha_for(rc, 1e-5)
    m = MCf.spce_solute_model(:ethanol_flex; coulomb=coulomb, epsilon_rf=Inf, cutoff=:site, rc_angstrom=rc,
                              α_per_angstrom=α)
    t = m.templates[2]
    ρ = density / 46.07 * 6.02214076e23 / 1e24                 # molecules / Å³
    L = cbrt(N / ρ) / m.σ_angstrom
    sys = MCf.RigidSystem([t], fill(1, N), L, m.ϵ_kelvin / 300.0, Xoshiro(seed))
    MCf.random_rigid_system!(sys, m.ff); MCf.minimize_rigid!(sys, m.ff; nsteps=300)
    coulomb === :ewald && MCf.attach_ewald!(sys, m.ff; rtol=1e-5, precision=:double)
    return m, sys
end

@testset "Flexible solvent L-moves: every slot kind is an involution; incremental bookkeeping = rebuild (RF, Ewald, Wolf)" begin
    for coul in (:rf, :ewald, :wolf)
        m, sys = ethanol_liquid(30, 4; coulomb=coul)
        ff = m.ff; N = sys.n_mol
        ek = MCf.init_rigid_ekmc(sys, ff)
        # T/R after S/D slots: the saved body must be restored for the rigid-body candidates
        kinds = [MCf.LM_S, MCf.LM_T, MCf.LM_D, MCf.LM_R, MCf.LM_S, MCf.LM_D]
        S = MCf.SolventLocalMoves(sys, ff, collect(1:N); kinds=kinds, a=0.05, θmax=0.2, ν=1.0, νD=1.0,
                                  site_a=0.02, φmax=0.6)
        @test MCf.slm_check!(S, sys, ff) < 1e-9
        @test all(S.cin[[2, 4], :] .== 0.0) && any(S.cin[[1, 3, 5, 6], :] .!= 0.0)
        for (ii, j) in ((3, 1), (11, 2), (17, 3), (24, 4))
            i = S.mols[ii]
            com0 = copy(sys.com); q0 = copy(sys.quat); B0 = copy(sys.body[i]); φ0 = copy(ek.phi)
            Ui0 = MCf.flex_intra_energy(sys.templates[1], ff, sys.body[i])
            MCf.slm_total_rate!(S, ek, sys.β)
            ξ = sum(S.Wm[1:ii-1]) + sum(S.rates[1:j-1, ii]) + 0.5 * S.rates[j, ii]
            @test MCf.slm_fire!(S, sys, ff, ek, ξ) == i
            @test !(sys.com[:, i] ≈ com0[:, i] && sys.quat[:, i] ≈ q0[:, i] && sys.body[i] ≈ B0)
            # the internal slot changed U_intra by the stored ΔU_intra; T/R kept the body
            Ui1 = MCf.flex_intra_energy(sys.templates[1], ff, sys.body[i])
            kinds[j] in (MCf.LM_T, MCf.LM_R) ? (@test sys.body[i] == B0) : (@test Ui1 != Ui0)
            @test MCf.slm_check!(S, sys, ff) < 1e-9                  # incremental columns (and ΔU_intra) = rebuild
            MCf.slm_total_rate!(S, ek, sys.β)
            @test S.cin[j, ii] ≈ Ui0 - Ui1 atol = 1e-9               # the slot now holds the way back
            ξ = sum(S.Wm[1:ii-1]) + sum(S.rates[1:j-1, ii]) + 0.5 * S.rates[j, ii]
            MCf.slm_fire!(S, sys, ff, ek, ξ)
            @test isapprox(sys.com, com0; atol=1e-12) && isapprox(sys.quat, q0; atol=1e-12) && isapprox(sys.body[i], B0; atol=1e-12)
            @test isapprox(ek.phi, φ0; atol=1e-9)
        end
        # moves of other molecules followed by column updates = full rebuild
        for _ in 1:15
            k = rand(sys.rng, 1:N)
            MCf.set_molecule!(sys, k, sys.com[:, k] .+ 0.05 .* randn(sys.rng, 3), sys.quat[:, k])
            MCf.update_rigid_ekmc_after_move!(ek, sys, ff, k)
            MCf.slm_moved!(S, sys, ff, k)
        end
        @test MCf.slm_check!(S, sys, ff) < 1e-9
        @test all(S.n_move[[MCf.LM_T, MCf.LM_R, MCf.LM_S, MCf.LM_D]] .> 0)
    end
end

@testset "Flexible solvent L-moves in the eKMC loop: the U_intra observable follows site and torsion events (regression)" begin
    # Before the fix the per-molecule U_intra (r.Uin) was refreshed at relocations but not after a solvent S/D event, so
    # ⟨U_intra⟩ of eKMC with solvent_flexible = true read stale conformations (slow test V3: 31.61 vs MC 30.39 kJ/mol).
    m, sys = ethanol_liquid(20, 9)
    ff = m.ff; t = m.templates[2]
    res = MCf.build_reservoir([t], 1, ff, sys.β, Xoshiro(2); M=200, burnin_sweeps=50, stride_sweeps=5, δ=0.02)
    r = MCf.init_rigid_ekmc_run(sys, ff; events=20_000, sample_every=20, block_events=2_000, reservoirs=[res],
                                local_solvent=true, solvent_flexible=true, solvent_nu=1e-3, mc_δ=0.05, mc_δθ=0.15,
                                solvent_site_a=0.01, solvent_φ=0.5)
    MCf.advance_rigid_ekmc!(r, sys, ff)
    @test r.slmoves.n_move[MCf.LM_S] > 0 && r.slmoves.n_move[MCf.LM_D] > 0
    fresh = [MCf.flex_intra_energy(t, ff, sys.body[i]) for i in 1:sys.n_mol]
    @test maximum(abs.(r.Uin .- fresh)) <= 1e-12 * maximum(abs.(fresh))
end

@testset "Flexible solvent L-moves in a rigid + flexible mixture (ethanol + SPC/E): inactive slots, bookkeeping = rebuild" begin
    # Before: solvent_flexible refused a mixture of rigid and flexible movers. Now rigid movers keep only T/R slots (their
    # LM_S/LM_D slots are inactive, rate 0); the incremental candidate energies equal a rebuild; no inactive slot ever fires.
    m = MCf.spce_solute_model(:ethanol_flex; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=7.0)
    ff = m.ff; tw, te = m.templates[1], m.templates[2]
    ne, nw = 12, 12
    L = cbrt((ne * 46.07 + nw * 18.02) / 0.5 / 6.02214076e23 * 1e24) / m.σ_angstrom      # 0.5 g/cm³: random placement works
    sys = MCf.RigidSystem([tw, te], vcat(fill(2, ne), fill(1, nw)), L, m.ϵ_kelvin / 300.0, Xoshiro(13))
    MCf.random_rigid_system!(sys, ff); MCf.minimize_rigid!(sys, ff; nsteps=200)
    res = MCf.build_reservoir([te], 1, ff, sys.β, Xoshiro(3); M=200, burnin_sweeps=50, stride_sweeps=5, δ=0.02)
    r = MCf.init_rigid_ekmc_run(sys, ff; events=20_000, sample_every=20, block_events=2_000, reservoirs=[nothing, res],
                                local_solvent=true, solvent_flexible=true, solvent_nu=1e-3, mc_δ=0.05, mc_δθ=0.15,
                                solvent_site_a=0.01, solvent_φ=0.5, debug_check_every=5_000)
    S = r.slmoves
    @test length(S.mols) == ne + nw
    for (ii, i) in enumerate(S.mols)
        fl = MCf.isflexible(sys.templates[sys.tmpl[i]])
        @test all(S.act[j, ii] == (fl || !(S.kinds[j] in (MCf.LM_S, MCf.LM_D))) for j in eachindex(S.kinds))
    end
    @test MCf.slm_check!(S, sys, ff) < 1e-9
    nS0 = copy(S.n_move)
    @test_logs min_level=Base.CoreLogging.Warn MCf.advance_rigid_ekmc!(r, sys, ff)
    @test S.n_move[MCf.LM_S] > nS0[MCf.LM_S] && S.n_move[MCf.LM_D] > nS0[MCf.LM_D] && S.n_move[MCf.LM_T] > nS0[MCf.LM_T]
    MCf.slm_total_rate!(S, r.ek, sys.β)
    @test all(S.rates[.!S.act] .== 0.0)
    @test MCf.slm_check!(S, sys, ff) < 1e-9
    fresh = [MCf.intra_energy_molecule(sys, ff, i) for i in 1:sys.n_mol]
    @test maximum(abs.(r.Uin .- fresh)) <= 1e-12 * maximum(abs.(fresh))
end

@testset "Flexible LJ-dimer liquid: eKMC + flexible solvent L-moves vs Metropolis MC (U, P, ⟨U_intra⟩)" begin
    N = 32; ρ = 0.3; T = 2.0; β = 1 / T
    sites = zeros(3, 2); sites[1, 1] = -0.3; sites[1, 2] = 0.3
    bt = MCf.BondedTerms(2; bonds=[MCf.HarmonicBond(1, 2, 0.6, 400.0)])
    dimer = MCf.RigidTemplate("flexdimer", sites, [1, 1], [0.0, 0.0], [1.0, 1.0]; bonded=bt)
    ff = MCf.SiteForceField([1.0], [1.0]; rc=2.5, coulomb=:none)
    res = MCf.build_reservoir([dimer], 1, ff, β, Xoshiro(3); M=5000, burnin_sweeps=500, stride_sweeps=10, δ=0.05)
    sys = MCf.RigidSystem([dimer], fill(1, N), cbrt(N / ρ), β, Xoshiro(7)); MCf.random_rigid_system!(sys, ff)
    st = MCf.RigidMCStats()
    for _ in 1:300, _ in 1:N
        i = rand(sys.rng, 1:N)
        MCf.mc_translate!(sys, ff, i, 0.15, st); MCf.mc_rotate!(sys, ff, i, 0.4, st); MCf.mc_site_move!(sys, ff, i, 0.05, st)
    end
    mc = MCf.run_rigid_mc!(deepcopy(sys), ff; sweeps=4000, δ=0.15, δθ=0.4, sample_every=1, block_size=100,
                           reservoirs=[res], δ_site=0.05)
    ν = MCf.estimate_relocation_rate(deepcopy(sys), ff; events=20_000) / N
    events = 1_000_000
    ek = @test_logs min_level=Base.CoreLogging.Warn MCf.run_rigid_ekmc!(sys, ff; events=events, sample_every=10,
            block_events=events ÷ 25, debug_check_every=events ÷ 10, reservoirs=[res], local_solvent=true,
            solvent_flexible=true, solvent_nu=ν, mc_δ=0.15, mc_δθ=0.4, solvent_site_a=0.05)
    S = ek.slmoves
    @info "flexible LJ-dimer liquid: MC vs eKMC + flexible solvent L-moves" MCf.mean(mc.U) MCf.stderr(mc.U) MCf.mean(ek.U) MCf.stderr(ek.U) MCf.mean(mc.P) MCf.stderr(mc.P) MCf.mean(ek.P) MCf.stderr(ek.P) MCf.mean(mc.U_intra) MCf.stderr(mc.U_intra) MCf.mean(ek.U_intra) MCf.stderr(ek.U_intra) S.n_move ek.relocations
    @test S.kinds == vcat(fill(MCf.LM_T, 3), fill(MCf.LM_R, 3), fill(MCf.LM_S, 4))      # no rotatable bond: no torsion slots
    @test S.n_move[MCf.LM_S] > events ÷ 20
    @test fs_within3σ(MCf.mean(mc.U), MCf.stderr(mc.U), MCf.mean(ek.U), MCf.stderr(ek.U))
    @test fs_within3σ(MCf.mean(mc.P), MCf.stderr(mc.P), MCf.mean(ek.P), MCf.stderr(ek.P))
    @test fs_within3σ(MCf.mean(mc.U_intra), MCf.stderr(mc.U_intra), MCf.mean(ek.U_intra), MCf.stderr(ek.U_intra))
end

@testset "Flexible solvent L-moves: column updates reach every candidate pose (torsion slots swing fragments; regression)" begin
    # Before the fix a mover's candidates were assumed to stay within its extent + √3 a of its COM (true for T/R only).
    # A torsion candidate swings a fragment further out: put a one-site LJ probe within rc of such a swung site but beyond
    # that old reach — moving the probe there must update the candidate's stored pair energy.
    mi = MCf.spce_solute_model(:ibuprofen_lpg; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=8.0)
    probe = MCf.RigidTemplate("probe", zeros(3, 1), [1], [0.0], [16.0])           # LJ type 1 (water O), no charge
    ff = mi.ff; rc = ff.rc
    L = 60.0 / mi.σ_angstrom
    sys = MCf.RigidSystem([mi.templates[2], probe], [1, 2], L, mi.ϵ_kelvin / 400.0, Xoshiro(8))
    c0 = fill(L / 2, 3)
    MCf.set_molecule!(sys, 1, c0, [1.0, 0.0, 0.0, 0.0]); MCf.set_molecule!(sys, 2, c0 .+ [0.45L, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    ek = MCf.init_rigid_ekmc(sys, ff)
    ext = sys.ext[1]; a = 0.02
    # draw torsion slots until one candidate puts a site beyond ext + 0.1 rc + √3 a from the COM
    S = nothing; far = nothing
    for seed in 1:200
        sys.rng = Xoshiro(seed)
        S = MCf.SolventLocalMoves(sys, ff, [1]; kinds=[MCf.LM_D], a=a, θmax=0.1, ν=1.0, νD=1.0, φmax=Float64(π))
        MCf._slm_save!(S, sys, 1); MCf._slm_place!(S, sys, 1, 1, 1)
        d = [sqrt(sum(abs2, sys.site_off[:, n] .+ [MCf._min_image(sys.com[k, 1] - c0[k], L) for k in 1:3])) for n in 1:33]
        n = argmax(d)
        p = c0 .+ [MCf._min_image(sys.com[k, 1] - c0[k], L) for k in 1:3] .+ sys.site_off[:, n]
        MCf._slm_restore!(S, sys, 1)
        d[n] > ext + 0.1rc + sqrt(3) * a && (far = (p, d[n]); break)
    end
    @test far !== nothing
    p, dn = far
    u = (p .- c0) ./ sqrt(sum(abs2, p .- c0))
    target = p .+ 0.9rc .* u                                    # within rc of the swung site
    @test sqrt(sum(abs2, target .- c0)) > rc + ext + sqrt(3) * a   # but outside the old reach of the mover's candidates
    @test S.cp[1, 2, 1] == 0.0                                   # probe far: no stored interaction
    MCf.set_molecule!(sys, 2, target, [1.0, 0.0, 0.0, 0.0])
    MCf.update_rigid_ekmc_after_move!(ek, sys, ff, 2)
    MCf.slm_moved!(S, sys, ff, 2)
    stored = S.cp[1, 2, 1]
    MCf.slm_rebuild_all!(S, sys, ff)
    @test S.cp[1, 2, 1] != 0.0 && stored == S.cp[1, 2, 1]
end

# Rejection-free local moves (L-moves, DESIGN §11.2) and the vacuum eKMC conformer pool (§11.3 E3): fast checks.
# The statistical checks (ladder + L-moves μ = stock on the LJ dimer) are in test_ladder.jl (run separately, ~1 h).
using Test, Random
using MolSim
const MCm = MolSim.MC

lm_within3σ(a, ea, c, ec) = abs(a - c) <= 3 * sqrt(ea^2 + ec^2)

function lm_dimer_system(ε_s, N, T, ρ, seed)
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


@testset "L-moves: every slot is an involution (fire j twice → the same state)" begin
    S = lm_dimer_system(1.0, 32, 1.5, 0.4, 11)
    sys = deepcopy(S.sys); ff = S.ff; s = S.Nt
    ek = MCm.init_rigid_ekmc(sys, ff)
    ek.frac = s; ek.lam[s] = 0.5; MCm.rebuild_rigid_ekmc!(ek, sys, ff)
    lm = MCm.LocalMoves(sys, s, 10; kinds=[MCm.LM_T, MCm.LM_R, MCm.LM_C], a=0.2, θmax=0.4, ν=1.0, νD=1.0, reservoir=S.res)
    for j in 1:3
        com0 = copy(sys.com[:, s]); q0 = copy(sys.quat[:, s]); B0 = copy(sys.body[s]); u0 = ek.frac_u
        W = MCm.lm_total_rate!(lm, sys, ff, sys.β, 0.5, ek.frac_u)
        @test W > 0 && !lm.stale
        @test lm.cand_u[j] ≈ (MCm.lm_rebuild!(lm, sys, ff); lm.cand_u[j])
        ξ = sum(lm.rates[1:j-1]) + 0.5 * lm.rates[j]
        @test MCm.lm_fire!(lm, sys, ff, ek, ξ) == j
        @test !(sys.com[:, s] ≈ com0 && sys.quat[:, s] ≈ q0 && sys.body[s] ≈ B0)
        MCm.lm_total_rate!(lm, sys, ff, sys.β, 0.5, ek.frac_u)
        @test lm.cand_u[j] ≈ u0 atol = 1e-9                      # the slot now holds the way back
        ξ = sum(lm.rates[1:j-1]) + 0.5 * lm.rates[j]
        MCm.lm_fire!(lm, sys, ff, ek, ξ)
        @test isapprox(sys.com[:, s], com0; atol=1e-12) && isapprox(sys.quat[:, s], q0; atol=1e-12) && sys.body[s] == B0
        @test ek.frac_u ≈ u0 atol = 1e-9
    end
end

@testset "L-moves: T/R candidates after a C candidate keep the saved body (regression)" begin
    # set_molecule! keeps the current body: before the fix a T or R slot evaluated after a C slot in lm_rebuild! used the
    # C candidate's conformation, so its stored energy was not the energy of the move that fires.
    S = lm_dimer_system(1.0, 32, 1.5, 0.4, 11)
    sys = deepcopy(S.sys); ff = S.ff; s = S.Nt
    lm = MCm.LocalMoves(sys, s, 10; kinds=[MCm.LM_C, MCm.LM_T, MCm.LM_R], a=0.2, θmax=0.4, ν=1.0, νD=1.0, reservoir=S.res)
    MCm.lm_rebuild!(lm, sys, ff)
    @test !(lm.dB[1] ≈ sys.body[s])                     # the C slot really changes the conformation
    B0 = copy(sys.body[s]); com0 = copy(sys.com[:, s]); q0 = copy(sys.quat[:, s])
    for j in 2:3
        MCm._lm_save!(lm, sys); MCm._lm_place!(lm, sys, j)
        @test sys.body[s] == B0
        MCm.eval_row!(lm.row_u, lm.row_w, sys, ff, s; recip=false)
        @test lm.cand_u[j] ≈ sum(lm.row_u) rtol = 1e-12
        MCm._lm_restore!(lm, sys)
    end
    @test sys.com[:, s] == com0 && sys.quat[:, s] == q0 && sys.body[s] == B0
end

lm_mean(x) = sum(x) / length(x)
function blockse(x; nb=20)
    n = length(x) ÷ nb; bm = [sum(x[(b-1)*n+1:b*n]) / n for b in 1:nb]; m = lm_mean(bm)
    return sqrt(sum(abs2, bm .- m) / (nb - 1) / nb)
end

@testset "Conformer pool by vacuum rejection-free eKMC = vacuum Metropolis pool (<U_intra>, dimer and ethanol)" begin
    S = lm_dimer_system(1.0, 8, 1.5, 0.4, 3)
    for (tmpls, idx, ff, β, δ) in ((S.sys.templates, 2, S.ff, S.β, 0.05),
                                   (begin m = MCm.spce_solute_model(:ethanol_flex; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=10.0)
                                        (m.templates, 2, m.ff, m.ϵ_kelvin / 300.0, 0.03) end))
        r_mc = MCm.build_reservoir(tmpls, idx, ff, β, Xoshiro(5); M=4000, burnin_sweeps=500, stride_sweeps=20, δ=δ)
        r_ek = MCm.build_reservoir_ekmc(tmpls, idx, ff, β, Xoshiro(6); M=4000, burnin_sweeps=500, stride_sweeps=20, δ=δ)
        a, ea = lm_mean(r_mc.U_intra), blockse(r_mc.U_intra); b, eb = lm_mean(r_ek.U_intra), blockse(r_ek.U_intra)
        @info "vacuum <U_intra> (reduced): Metropolis vs rejection-free eKMC" tmpls[idx].name a ea b eb r_ek.acceptance
        @test length(r_ek.confs) == 4000 && all(c -> size(c) == size(r_mc.confs[1]), r_ek.confs)
        @test abs(a - b) <= 4 * sqrt(ea^2 + eb^2)
    end
end

@testset "Ladder + hub + L-moves: bookkeeping exact (debug rebuilds), moves at every rung with λ > 0, D0 diagnostics" begin
    S = lm_dimer_system(1.0, 32, 1.5, 0.4, 11)
    ν = MCm.estimate_relocation_rate(deepcopy(S.sys), S.ff; events=20_000)
    ev = 100_000
    lad = @test_logs min_level=Base.CoreLogging.Warn MCm.run_rigid_ekmc!(deepcopy(S.sys), S.ff; events=ev, sample_every=1,
            block_events=ev ÷ 10, species_mu=true, reservoirs=[nothing, S.res], ladder_solute=S.Nt, ladder_nu=ν,
            ladder_wl_events=ev ÷ 3, ladder_jump=true, ladder_jump_max=7, ladder_local=true, debug_check_every=ev ÷ 50)
    lm = lad.lmoves; sm = MCm.ladder_summary(lad.ladder)
    @test lm.kinds == [1, 1, 1, 2, 2, 2, 3, 3]
    @test all(lm.n_move .> 0) && lm.n_move_rung[1] == 0 && sum(lm.n_move_rung) > 0 && lm.n_refresh > 0
    @test sum(sm.n_jump[8:end]) == 0 && sum(sm.n_jump[3:7]) > 0          # hub limited to rungs 3…7
    @test all(0 .< sm.res_frac[sm.t_rung .> 0] .<= 1) && all(sm.pose_frac .<= sm.res_frac .+ 1e-12)
end

# ---- internal moves of a flexible molecule (JOURNAL 79–80) ----
function lm_dihed(X, i, j, k, l)
    b1 = X[:, j] - X[:, i]; b2 = X[:, k] - X[:, j]; b3 = X[:, l] - X[:, k]
    n1 = [b1[2]*b2[3]-b1[3]*b2[2], b1[3]*b2[1]-b1[1]*b2[3], b1[1]*b2[2]-b1[2]*b2[1]]
    n2 = [b2[2]*b3[3]-b2[3]*b3[2], b2[3]*b3[1]-b2[1]*b3[3], b2[1]*b3[2]-b2[2]*b3[1]]
    u = b2 / sqrt(sum(abs2, b2)); m1 = [n1[2]*u[3]-n1[3]*u[2], n1[3]*u[1]-n1[1]*u[3], n1[1]*u[2]-n1[2]*u[1]]
    return atan(sum(m1 .* n2), sum(n1 .* n2))
end
lm_alcohol(sol) = MCm.spce_solute_model(sol; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=10.0)

@testset "Torsion moves: rotatable bonds, isometry of the fragment rotation, inverse" begin
    @test length(MCm.rotatable_bonds(lm_dimer_system(1.0, 4, 1.5, 0.4, 1).dimer)) == 0
    for (sol, nb) in ((:methanol_flex, 1), (:ethanol_flex, 2), (:propanol_flex, 3))
        t = lm_alcohol(sol).templates[2]; rb = MCm.rotatable_bonds(t)
        @test length(rb) == nb
        for (a, b, mov) in rb
            X = copy(t.sites_body); X0 = copy(X)
            MCm.rotate_fragment!(X, a, b, mov, 1.3)
            D(Y) = [sqrt(sum(abs2, Y[:, i] - Y[:, j])) for i in 1:t.n_sites, j in 1:t.n_sites]
            fixed = setdiff(1:t.n_sites, mov)
            @test D(X)[mov, mov] ≈ D(X0)[mov, mov] && D(X)[fixed, fixed] ≈ D(X0)[fixed, fixed] && D(X)[[a, b], mov] ≈ D(X0)[[a, b], mov]
            @test !(X ≈ X0)
            MCm.rotate_fragment!(X, a, b, mov, -1.3)
            @test maximum(abs.(X - X0)) < 1e-12
        end
    end
end

@testset "Conformer pools with torsion moves cross the 1-propanol C–C–C–O barrier; Metropolis = eKMC" begin
    m = lm_alcohol(:propanol_flex); β = m.ϵ_kelvin / 300.0
    trans(r) = (φ = [lm_dihed(c, 1, 5, 8, 11) for c in r.confs]; tr = abs.(φ) .> 2π / 3; (count(tr) / length(tr), count(i -> tr[i] != tr[i-1], 2:length(tr))))
    r0 = MCm.build_reservoir(m.templates, 2, m.ff, β, Xoshiro(5); M=2000, burnin_sweeps=200, stride_sweeps=20, δ=0.03, torsion=false)
    @test trans(r0)[2] < 20                                         # single-site moves alone: (almost) no flips (the bug)
    r_mc = MCm.build_reservoir(m.templates, 2, m.ff, β, Xoshiro(5); M=4000, burnin_sweeps=500, stride_sweeps=20, δ=0.03, torsion=true)
    r_ek = MCm.build_reservoir_ekmc(m.templates, 2, m.ff, β, Xoshiro(6); M=4000, burnin_sweeps=500, stride_sweeps=20, δ=0.03, torsion=true)
    (fa, na), (fb, nb) = trans(r_mc), trans(r_ek)
    a, ea = lm_mean(r_mc.U_intra), blockse(r_mc.U_intra); b, eb = lm_mean(r_ek.U_intra), blockse(r_ek.U_intra)
    @info "1-propanol torsion pools" fa na fb nb a ea b eb
    @test na > 200 && nb > 200
    @test abs(a - b) <= 4 * sqrt(ea^2 + eb^2)
    @test abs(fa - fb) < 0.08
end

@testset "Conformer pools default to torsion moves for templates with rotatable bonds (JOURNAL 2026-10-20)" begin
    # Regression: the default was torsion=false, so drivers that left it out (liquid-ethanol tests) relocated
    # molecules into a pool that was not Boltzmann over the torsions. The default must equal torsion=true.
    m = lm_alcohol(:ethanol_flex); β = m.ϵ_kelvin / 298.0
    @test !isempty(MCm.rotatable_bonds(m.templates[2]))
    kw = (M=50, burnin_sweeps=20, stride_sweeps=2, δ=0.03)
    d = MCm.build_reservoir(m.templates, 2, m.ff, β, Xoshiro(11); kw...)
    e = MCm.build_reservoir(m.templates, 2, m.ff, β, Xoshiro(11); kw..., torsion=true)
    @test d.confs == e.confs
    d = MCm.build_reservoir_ekmc(m.templates, 2, m.ff, β, Xoshiro(12); kw...)
    e = MCm.build_reservoir_ekmc(m.templates, 2, m.ff, β, Xoshiro(12); kw..., torsion=true)
    @test d.confs == e.confs
end

# one flexible solute and a few waters (SPC/E + alcohol), for the slot mechanics
function lm_alcohol_system(sol, nw, seed; L=nothing)
    m = lm_alcohol(sol); ff = first(MCm.kT_overlap_cap(m.ff, m.templates, 300.0 / m.ϵ_kelvin; U_kT=100.0))
    ρ = 1000.0 * 1e3 / 18.0153 * 6.02214129e23 / 1e30 * m.σ_angstrom^3
    Lb = L === nothing ? cbrt((nw + 1) / ρ) : L
    sys = MCm.RigidSystem(m.templates, vcat(fill(1, nw), 2), Lb, m.ϵ_kelvin / 300.0, Xoshiro(seed)); MCm.random_rigid_system!(sys, ff)
    MCm.minimize_rigid!(sys, ff; nsteps=200)
    return (; m, ff, sys, s=nw + 1)
end

@testset "L-moves LM_S / LM_D: involution, U_intra in the rate, bookkeeping" begin
    S = lm_alcohol_system(:propanol_flex, 30, 4; L=7.0)
    sys = deepcopy(S.sys); ff = S.ff; s = S.s; t = sys.templates[sys.tmpl[s]]
    ek = MCm.init_rigid_ekmc(sys, ff)
    ek.frac = s; ek.lam[s] = 0.35; MCm.rebuild_rigid_ekmc!(ek, sys, ff)
    @test_throws ErrorException MCm.LocalMoves(sys, s, 10; kinds=[MCm.LM_S], a=0.2, θmax=0.3, ν=1.0, νD=1.0)   # site_a required
    lm = MCm.LocalMoves(sys, s, 10; kinds=[MCm.LM_S, MCm.LM_S, MCm.LM_D, MCm.LM_D], a=0.2, θmax=0.3, ν=1.0, νD=1.0, site_a=0.03, φmax=1.0)
    for j in 1:4
        com0 = copy(sys.com[:, s]); q0 = copy(sys.quat[:, s]); B0 = copy(sys.body[s]); u0 = ek.frac_u
        Ui0 = MCm.flex_intra_energy(t, ff, B0)
        W = MCm.lm_total_rate!(lm, sys, ff, sys.β, 0.35, ek.frac_u)
        @test W > 0 && lm.uin0 ≈ Ui0
        x = 0.35 * (lm.cand_u[j] - u0) + lm.cand_intra[j] - Ui0
        @test lm.rates[j] ≈ exp(min(-0.5 * sys.β * x, MCm.MOBILITY_LOG_MAX))
        ξ = sum(lm.rates[1:j-1]) + 0.5 * lm.rates[j]
        @test MCm.lm_fire!(lm, sys, ff, ek, ξ) == j
        @test MCm.flex_intra_energy(t, ff, sys.body[s]) ≈ lm.cand_intra[j]
        @test abs(sum(sys.body[s] * t.site_mass)) < 1e-10                       # body stays COM-centred
        @test sys.quat[:, s] ≈ q0                                               # orientation kept
        MCm.lm_total_rate!(lm, sys, ff, sys.β, 0.35, ek.frac_u)
        @test lm.cand_u[j] ≈ u0 atol = 1e-9
        @test lm.cand_intra[j] ≈ Ui0 atol = 1e-9                                 # the slot holds the way back
        ξ = sum(lm.rates[1:j-1]) + 0.5 * lm.rates[j]
        MCm.lm_fire!(lm, sys, ff, ek, ξ)
        @test isapprox(sys.com[:, s], com0; atol=1e-12) && isapprox(sys.body[s], B0; atol=1e-12)
        @test ek.frac_u ≈ u0 atol = 1e-9
    end
    @test lm.n_move == [0, 0, 0, 4, 4]
end

@testset "LM_S + LM_D alone sample the isolated 1-propanol: ⟨U_intra⟩ and trans fraction = Metropolis torsion pool" begin
    S = lm_alcohol_system(:propanol_flex, 1, 7; L=30.0)                         # one water, far beyond the cutoff
    sys = S.sys; ff = S.ff; s = S.s; t = sys.templates[sys.tmpl[s]]
    MCm.set_molecule!(sys, 1, [1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]); MCm.set_molecule!(sys, s, [15.0, 15.0, 15.0], sys.quat[:, s])
    ek = MCm.init_rigid_ekmc(sys, ff); ek.frac = s; ek.lam[s] = 1.0; MCm.rebuild_rigid_ekmc!(ek, sys, ff)
    @test ek.frac_u == 0.0
    lm = MCm.LocalMoves(sys, s, 10; kinds=vcat(fill(MCm.LM_S, 6), fill(MCm.LM_D, 3)), a=0.2, θmax=0.3, ν=1.0, νD=1.0, site_a=0.03, φmax=Float64(π))
    rng = Xoshiro(17); nev = 300_000; burn = 30_000
    tU = 0.0; tT = 0.0; ttot = 0.0; Us = Float64[]; ws = Float64[]
    for ev in 1:nev
        W = MCm.lm_total_rate!(lm, sys, ff, sys.β, 1.0, ek.frac_u); R = W + lm.νD; dt = 1 / R
        if ev > burn
            Ui = MCm.flex_intra_energy(t, ff, sys.body[s]); tr = abs(lm_dihed(sys.body[s], 1, 5, 8, 11)) > 2π / 3
            push!(Us, Ui); push!(ws, dt); tT += tr * dt; ttot += dt
        end
        ξ = rand(rng) * R
        ξ < W ? MCm.lm_fire!(lm, sys, ff, ek, ξ) : MCm.lm_refresh!(lm, sys, nothing)
    end
    # time-weighted mean with a block error over equal-event blocks
    nb = 20; n = length(Us) ÷ nb
    bm = [sum(Us[(b-1)*n+1:b*n] .* ws[(b-1)*n+1:b*n]) / sum(ws[(b-1)*n+1:b*n]) for b in 1:nb]
    a = sum(Us .* ws) / sum(ws); ea = sqrt(sum(abs2, bm .- lm_mean(bm)) / (nb - 1) / nb)
    m = S.m
    r_mc = MCm.build_reservoir(m.templates, 2, m.ff, sys.β, Xoshiro(5); M=6000, burnin_sweeps=500, stride_sweeps=20, δ=0.03, torsion=true)
    b = lm_mean(r_mc.U_intra); eb = blockse(r_mc.U_intra)
    fb = count(c -> abs(lm_dihed(c, 1, 5, 8, 11)) > 2π / 3, r_mc.confs) / length(r_mc.confs)
    @info "isolated 1-propanol: LM_S + LM_D vs Metropolis pool" a ea b eb tT / ttot fb lm.n_move
    @test abs(a - b) <= 4 * sqrt(ea^2 + eb^2)
    @test abs(tT / ttot - fb) < 0.08
end

@testset "Ladder with fixed η (no Wang–Landau) and per-rung dihedral histograms (diagnostic): time bookkeeping exact" begin
    m = MCm.spce_solute_model(:ethanol_flex; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=7.0)
    ff = m.ff; N = 30; Nt = N + 1
    L = cbrt(N / 0.02) / m.σ_angstrom                                   # dilute enough for random placement
    sys = MCm.RigidSystem(m.templates, vcat(fill(1, N), 2), L, m.ϵ_kelvin / 300.0, Xoshiro(17)); MCm.random_rigid_system!(sys, ff)
    res = MCm.build_reservoir(m.templates, 2, ff, sys.β, Xoshiro(4); M=300, burnin_sweeps=50, stride_sweeps=5, δ=0.02, torsion=true)
    λs = [0.0, 0.1, 0.4, 1.0]; η = [3.0, 2.0, 1.0, 0.0]
    dih = [(1, 5, 8, 9), (2, 1, 5, 8)]
    ν = MCm.estimate_relocation_rate(deepcopy(sys), ff; events=5_000)
    r = MCm.init_rigid_ekmc_run(sys, ff; events=30_000, sample_every=10, block_events=3_000, reservoirs=[nothing, res],
                                ladder_solute=Nt, ladder_lambda=λs, ladder_nu=ν, ladder_eta=η, ladder_wl_events=0,
                                ladder_jump=true, ladder_local=true, track_dihedrals=dih)
    MCm.advance_rigid_ekmc!(r, sys, ff)
    sm = MCm.ladder_summary(r.ladder)
    @test sm.η == η && sm.wl_stages == 0
    @test length(r.dih_hist) == length(λs) && all(length(h) == length(dih) for h in r.dih_hist)
    @test count(>(0), sm.t_rung) >= 2
    for k in eachindex(λs), d in eachindex(dih)
        @test isapprox(sum(r.dih_hist[k][d]), sm.t_rung[k]; rtol=1e-12)
    end
end

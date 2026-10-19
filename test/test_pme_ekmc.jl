# SPME mesh path inside rigid/flexible-molecule MC and eKMC (PMERecip.jl, DESIGN §13.3 E5, JOURNAL 122): the pairwise split
# Σ_{m<n} Q_m·Φ_n + Σ_m ½Q_m·Φ_m equals the flat SPME energy (`pme_energy`); the energy of flexible solutes in SPC/E equals
# the GROMACS PME rerun (fine mesh); every incremental update (MC commits, eKMC rows/φ/frac_u, solute and solvent L-move
# candidates incl. flexible site/torsion slots) equals a rebuild.
using Test, Random, Serialization
using MolSim
const MCp = MolSim.MC
isdefined(@__MODULE__, :load_solute_water_gro) || include(joinpath(@__DIR__, "test_solutes_gromacs_loader.jl"))

function pme_water(N, seed; rc_angstrom=4.5, K=16, p=4, equil=30)
    α = MCp.ewald_alpha_for(rc_angstrom, 1e-5)
    m = MCp.spce_model(; α_per_angstrom=α, rc_angstrom=rc_angstrom, coulomb=:ewald)
    L = cbrt(N / 0.0334) / m.σ_angstrom
    sys = MCp.RigidSystem([m.template], fill(1, N), L, m.ϵ_kelvin / 300.0, Xoshiro(seed))
    MCp.random_rigid_system!(sys, m.ff)
    MCp.attach_pme!(sys, m.ff; K=K, order=p)
    st = MCp.RigidMCStats()
    for _ in 1:equil, _ in 1:N
        MCp.mc_translate!(sys, m.ff, rand(sys.rng, 1:N), 0.1, st); MCp.mc_rotate!(sys, m.ff, rand(sys.rng, 1:N), 0.3, st)
    end
    return m, sys, st
end

"Largest change of the committed mesh state (Q, Φ, image energies, stencils) under a rebuild."
function pme_committed_drift(sys)
    R = sys.recip
    Q = copy(R.Q); Φ = copy(R.Φ); e = copy(R.eimg); b = copy(R.base); θ = copy(R.θ); ws = copy(R.wself); wi = copy(R.wimg)
    MCp.recip_rebuild!(R, sys)
    return max(maximum(abs.(Q .- R.Q)), maximum(abs.(Φ .- R.Φ)), maximum(abs.(e .- R.eimg)), maximum(abs.(θ .- R.θ)),
               maximum(abs.(ws .- R.wself)), maximum(abs.(wi .- R.wimg)), b == R.base ? 0.0 : Inf)
end

function pme_ek_drift(ek, sys, ff)
    φ = copy(ek.phi); P = copy(ek.pair_u); fu = ek.frac_u
    MCp.rebuild_rigid_ekmc!(ek, sys, ff)
    return max(maximum(abs.(φ .- ek.phi)), maximum(abs.(P .- ek.pair_u)), abs(fu - ek.frac_u))
end

@testset "SPME mesh path: pairwise split = flat SPME energy (Essmann 1995), cross terms Q_m·Φ_n" begin
    m, sys, _ = pme_water(30, 3)
    R = sys.recip; ff = m.ff
    pos, q, mol, _, _ = MCp.site_arrays(sys, ff)
    ep = MCp.EwaldParams(sys.L, R.α, ff.rc, R.ew.n2max, R.ke)
    e = MCp.pme_energy(pos, q, mol, ep, R.K, R.p)
    # recip_energy_total = Σ_{m<n} ψ_mn + Σ_m e_m; adding back the mesh self energies and removing the exact image terms
    # leaves ½ Q·Φ − Σ corr = the SPME Fourier sum + self + intramolecular erf terms of every charge pair
    Etot = MCp.recip_energy_total(R, sys)
    @test isapprox(Etot - sum(R.eimg) + sum(R.mself) - sum(R.corr), e.fourier + e.self + e.intra; rtol=1e-11)
    Ψ = MCp.recip_pair_matrix(R, sys)
    @test isapprox(0.5 * sum(Ψ) + sum(R.mself), e.fourier; rtol=1e-11)
    @test Ψ ≈ transpose(Ψ) rtol = 1e-10                        # ψ_mn = ψ_nm (G symmetric)
    # the image term of the mesh path is the exact Ewald one
    MCp.attach_ewald!(sys, ff; n2max=R.ew.n2max, precision=:double)
    Re = sys.recip
    @test maximum(abs.(Re.eimg .- R.eimg)) < 1e-12
    # fine mesh vs Ewald: the tolerance of the Ewald / fine-PME comparison in test_ewald_ekmc.jl (2e-5 relative)
    MCp.attach_pme!(sys, ff; K=48, order=8, n2max=R.ew.n2max)
    Ufine = MCp.total_pair_energy(sys, ff)
    MCp.attach_ewald!(sys, ff; n2max=R.ew.n2max, precision=:double)
    @test isapprox(Ufine, MCp.total_pair_energy(sys, ff); rtol=2e-5)
end

@testset "SPME mesh path vs GROMACS PME: butanol and caffeine in 500 SPC/E (fine mesh, coulomb-modifier None)" begin
    # the GROMACS PME rerun of test_solutes_gromacs.jl (0.04 nm, order 8; data/solute_gromacs_ewald_reference.txt);
    # tolerance as the other GROMACS comparisons (2e-5 relative: GROMACS mixed precision over ~10⁶ pair terms)
    ref = Dict{Tuple{Symbol,String},Float64}()
    for l in eachline(joinpath(@__DIR__, "data", "solute_gromacs_ewald_reference.txt"))
        startswith(l, "#") && continue
        f = split(l); ref[(Symbol(f[1]), f[2])] = parse(Float64, f[3])
    end
    α_A = MCp.ewald_alpha_for(10.0, 1e-5)
    for sym in (:butanol_flex, :caffeine_lpg)
        m, sys, nw = load_solute_water_gro(sym, joinpath(@__DIR__, "data", "gromacs_$(sym)_spce500.gro"); coulomb=:ewald, α_A=α_A)
        kJ = m.ϵ_kelvin * 0.0083144626
        MCp.attach_pme!(sys, m.ff; spacing=0.4 / m.σ_angstrom, order=8)      # GROMACS fourier-spacing 0.04 nm, pme-order 8
        U = (MCp.total_pair_energy(sys, m.ff) + MCp.total_intra_energy(sys, m.ff)) * kJ
        @test isapprox(U, ref[(sym, "pmefine_nomod")]; rtol=2e-5)
    end
end

@testset "SPME mesh path: MC commits and eKMC bookkeeping (rows, φ, fractional coupling): incremental = rebuild" begin
    m, sys, st = pme_water(40, 11)
    ff = m.ff
    @test st.trans_acc > 0 && st.rot_acc > 0
    @test pme_committed_drift(sys) < 1e-10
    ek = MCp.init_rigid_ekmc(sys, ff)
    @test 0.5 * sum(ek.phi) + 0.5 * MCp._image_energy_sum(ek, sys) ≈ MCp.total_pair_energy(sys, ff) rtol = 1e-11
    acc = MCp.ChemicalPotentialAccumulator()
    for _ in 1:200
        MCp.rigid_ekmc_step!(ek, sys, ff, acc)
    end
    for _ in 1:2
        MCp.ekmc_mc_sweep!(ek, sys, ff, 0.1, 0.3)
    end
    @test pme_ek_drift(ek, sys, ff) < 1e-9
    @test pme_committed_drift(sys) < 1e-10
    s = 5
    ek.frac = s; ek.lam[s] = 0.4
    MCp.rebuild_rigid_ekmc!(ek, sys, ff)
    @test ek.frac_u ≈ MCp.removal_energy(sys, ff, s) rtol = 1e-11
    for _ in 1:200
        k = MCp.select_by_mobility(ek, sys.rng)
        MCp.set_molecule!(sys, k, sys.L .* rand(sys.rng, 3), MCp.uniform_random_quaternion(sys.rng))
        MCp.update_rigid_ekmc_after_move!(ek, sys, ff, k)
    end
    MCp.set_fractional_coupling!(ek, sys.β, 0.7)
    MCp.set_molecule!(sys, s, sys.L .* rand(sys.rng, 3), MCp.uniform_random_quaternion(sys.rng))
    MCp.update_rigid_ekmc_after_move!(ek, sys, ff, s)
    MCp.ekmc_mc_sweep!(ek, sys, ff, 0.1, 0.3)
    @test pme_ek_drift(ek, sys, ff) < 1e-9
    @test ek.frac_u ≈ MCp.removal_energy(sys, ff, s) rtol = 1e-10
end

@testset "SPME mesh path: solvent and solute L-move candidates (Q_c·ΔΦ updates) = rebuild" begin
    m, sys, _ = pme_water(36, 12)
    ff = m.ff
    N = sys.n_mol; s = N
    ek = MCp.init_rigid_ekmc(sys, ff)
    ek.frac = s; ek.lam[s] = 0.6
    MCp.rebuild_rigid_ekmc!(ek, sys, ff)
    S = MCp.SolventLocalMoves(sys, ff, collect(1:N-1); kinds=[MCp.LM_T, MCp.LM_R, MCp.LM_T], a=0.1, θmax=0.3, ν=1.0,
                              νD=1.0, frac=s)
    lm = MCp.LocalMoves(sys, s, 2; kinds=[MCp.LM_T, MCp.LM_R, MCp.LM_T], a=0.1, θmax=0.3, ν=1.0, νD=1.0)
    MCp.lm_rebuild!(lm, sys, ff)
    @test any(!=(0.0), S.cr_u) && any(!=(0.0), S.cr_s) && any(!=(0.0), lm.cand_rec)
    for it in 1:45
        if it % 3 == 0
            MCp.slm_total_rate!(S, ek, sys.β)
            i = MCp.slm_fire!(S, sys, ff, ek, rand(sys.rng) * S.W)
            MCp.lm_molecule_moved!(lm, sys, ff, i)
        elseif it % 3 == 1
            k = rand(sys.rng, 1:N-1)
            MCp.set_molecule!(sys, k, sys.L .* rand(sys.rng, 3), MCp.uniform_random_quaternion(sys.rng))
            MCp.update_rigid_ekmc_after_move!(ek, sys, ff, k)
            MCp.slm_moved!(S, sys, ff, k)
            MCp.lm_molecule_moved!(lm, sys, ff, k)
        else
            MCp.lm_total_rate!(lm, sys, ff, sys.β, ek.lam[s], ek.frac_u)
            MCp.lm_fire!(lm, sys, ff, ek, rand(sys.rng) * sum(lm.rates))
            MCp.slm_moved!(S, sys, ff, s)
            MCp.lm_rebuild!(lm, sys, ff)
        end
    end
    @test MCp.slm_check!(S, sys, ff) < 1e-9
    cu = copy(lm.cand_u); MCp.lm_rebuild!(lm, sys, ff)
    @test maximum(abs.(cu .- lm.cand_u)) < 1e-9
    @test pme_ek_drift(ek, sys, ff) < 1e-9
    j = 2                                                      # a candidate energy is the removal energy at that pose
    MCp._lm_save!(lm, sys); MCp._lm_place!(lm, sys, j)
    u_place = MCp.removal_energy(sys, ff, s)
    MCp._lm_restore!(lm, sys)
    @test lm.cand_u[j] ≈ u_place rtol = 1e-10
end

@testset "SPME mesh path: flexible solvent site and torsion slots (ethanol liquid): incremental = rebuild" begin
    rc = 7.0
    α = MCp.ewald_alpha_for(rc, 1e-5)
    m = MCp.spce_solute_model(:ethanol_flex; coulomb=:ewald, cutoff=:site, rc_angstrom=rc, α_per_angstrom=α)
    t = m.templates[2]; ff = m.ff; N = 24
    L = cbrt(N / (0.4 / 46.07 * 6.02214076e23 / 1e24)) / m.σ_angstrom
    sys = MCp.RigidSystem([t], fill(1, N), L, m.ϵ_kelvin / 300.0, Xoshiro(4))
    MCp.random_rigid_system!(sys, ff); MCp.minimize_rigid!(sys, ff; nsteps=200)
    MCp.attach_pme!(sys, ff; K=20, order=4)
    ek = MCp.init_rigid_ekmc(sys, ff)
    S = MCp.SolventLocalMoves(sys, ff, collect(1:N); kinds=[MCp.LM_S, MCp.LM_T, MCp.LM_D, MCp.LM_R], a=0.05, θmax=0.2,
                              ν=1.0, νD=1.0, site_a=0.02, φmax=0.6)
    @test MCp.slm_check!(S, sys, ff) < 1e-9
    for _ in 1:30
        MCp.slm_total_rate!(S, ek, sys.β)
        MCp.slm_fire!(S, sys, ff, ek, rand(sys.rng) * S.W)
    end
    @test all(S.n_move[[MCp.LM_T, MCp.LM_R, MCp.LM_S, MCp.LM_D]] .> 0) || all(S.n_move[[MCp.LM_S, MCp.LM_D]] .> 0)
    @test MCp.slm_check!(S, sys, ff) < 1e-9
    @test pme_ek_drift(ek, sys, ff) < 1e-9
    @test pme_committed_drift(sys) < 1e-10
end

"SPME mesh path with the gather engine on `backend` (rows and solvent-candidate updates on the device): = host rebuild."
function pme_device_check(backend::Symbol)
    m, sys, _ = pme_water(36, 13)
    ff = m.ff
    N = sys.n_mol; s = N
    MCp.pme_attach_device!(sys.recip, sys; backend=backend)
    ek = MCp.init_rigid_ekmc(sys, ff)
    ek.frac = s; ek.lam[s] = 0.6
    MCp.rebuild_rigid_ekmc!(ek, sys, ff)
    S = MCp.SolventLocalMoves(sys, ff, collect(1:N-1); kinds=[MCp.LM_T, MCp.LM_R, MCp.LM_T], a=0.1, θmax=0.3, ν=1.0,
                              νD=1.0, frac=s)
    for it in 1:60
        if it % 2 == 0
            MCp.slm_total_rate!(S, ek, sys.β); MCp.slm_fire!(S, sys, ff, ek, rand(sys.rng) * S.W)
        else
            k = it % 6 == 1 ? s : rand(sys.rng, 1:N-1)                # the fractional molecule too (cr_s updates)
            MCp.set_molecule!(sys, k, sys.L .* rand(sys.rng, 3), MCp.uniform_random_quaternion(sys.rng))
            MCp.update_rigid_ekmc_after_move!(ek, sys, ff, k); MCp.slm_moved!(S, sys, ff, k)
        end
        it == 30 && (k = rand(sys.rng, 1:N-1);                         # the same molecule twice in a row
                     for _ in 1:2
                         MCp.set_molecule!(sys, k, sys.L .* rand(sys.rng, 3), MCp.uniform_random_quaternion(sys.rng))
                         MCp.update_rigid_ekmc_after_move!(ek, sys, ff, k); MCp.slm_moved!(S, sys, ff, k)
                     end)
    end
    @test S.crP.dev !== nothing
    @test MCp.slm_check!(S, sys, ff) < 1e-9
    @test pme_ek_drift(ek, sys, ff) < 1e-9
end

@testset "SPME mesh path, gather engine on the KernelAbstractions CPU backend: incremental = host rebuild" begin
    pme_device_check(:cpu)
end

@testset "SPME mesh path: reciprocal molecular virial = −dE_rec/d ln L (COM scaling; E6, JOURNAL 125)" begin
    # the mesh-path Hamiltonian (SPME cross terms + exact image terms); central difference in ln L, h = 1e-5: truncation
    # ≈ h²/6·|E‴| and rounding ≈ ε|E|/h with |E‴| ~ |E| ⇒ ≈ 4e-11 |E| — tolerance 1e-8 |E| (as the Ewald virial test)
    for (N, K, p) in ((30, 16, 4), (40, 20, 6))
        α = MCp.ewald_alpha_for(4.5, 1e-5)
        m = MCp.spce_model(; α_per_angstrom=α, rc_angstrom=4.5, coulomb=:ewald)
        sys = MCp.RigidSystem([m.template], fill(1, N), cbrt(N / 0.0334) / m.σ_angstrom, m.ϵ_kelvin / 300.0, Xoshiro(N))
        MCp.random_rigid_system!(sys, m.ff)
        MCp.attach_pme!(sys, m.ff; K=K, order=p)
        W = MCp.recip_virial(sys.recip, sys)
        E(f) = (s2 = deepcopy(sys); s2.L *= f;
                for i in 1:s2.n_mol; MCp.set_molecule!(s2, i, s2.com[:, i] .* f, s2.quat[:, i]); end;
                MCp.recip_rebuild!(s2.recip, s2); MCp.recip_energy_total(s2.recip, s2))
        h = 1e-5
        @test abs(-(E(1 + h) - E(1 - h)) / (2h) - W) <= 1e-8 * abs(E(1.0))
    end
end

@testset "SPME state restored from a checkpoint in a NEW process (FFTW plans recreated; regression, JOURNAL 126)" begin
    # FFTW plans are pointers into the process that made them; a checkpointed PMERecip used them after deserialization
    # in the next chain piece and segfaulted on the first FFT
    m, sys, _ = pme_water(20, 5)
    U0 = MCp.total_pair_energy(sys, m.ff)
    f = tempname()
    open(io -> Serialization.serialize(io, (sys, m.ff)), f, "w")
    proj = dirname(Base.active_project())
    code = "using MolSim, Serialization; sys, ff = deserialize(\"$f\"); print(MolSim.MC.total_pair_energy(sys, ff))"
    out = read(`$(Base.julia_cmd()) --startup-file=no --project=$proj -e $code`, String)
    @test isapprox(parse(Float64, out), U0; rtol=1e-12)
    s2 = deepcopy(sys)                                          # a deepcopy makes its own plans
    @test s2.recip.fplan === nothing && MCp.total_pair_energy(s2, m.ff) == U0
    rm(f; force=true)
end

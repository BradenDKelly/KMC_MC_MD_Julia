# Reciprocal-space Ewald inside rigid-molecule MC and eKMC (DESIGN §13, JOURNAL 96): the RigidSystem energy equals the
# NIST-validated flat-array `ewald_energy`; removal energies equal energy differences; every incremental update (MC
# commits, eKMC rows, image energies in φ, fractional coupling, solute and solvent L-move candidates) equals a rebuild.
using Test, Random
using MolSim
const MCe = MolSim.MC

include(joinpath(@__DIR__, "nist_spce_reader.jl"))

function ewald_spce_template_angstrom()
    θ = 109.47 * π / 180
    sites = zeros(3, 3)
    sites[:, 2] .= (sin(θ / 2), cos(θ / 2), 0.0)
    sites[:, 3] .= (-sin(θ / 2), cos(θ / 2), 0.0)
    MCe.RigidTemplate("SPC/E-A", sites, [1, 0, 0], [-2 * SPCE_QH, SPCE_QH, SPCE_QH], [15.9994, 1.00794, 1.00794])
end

function nist_rigid_system(cfg, rc)
    L, M, pos, q, σ, ε, mol = read_nist_config(joinpath(NIST_DIR, "spce_sample_config_periodic$(cfg).txt"))
    ff = MCe.SiteForceField([SPCE_SIGMA], [SPCE_EPS]; rc=rc, coulomb=:ewald, α=5.6 / L, ke=KE_K_ANGSTROM,
                            use_lrc=true, cap_overlaps=false)
    sys = MCe.RigidSystem([ewald_spce_template_angstrom()], fill(1, M), L, 1.0 / 300.0, Xoshiro(1))
    for i in 1:M
        lab = pos[:, 3i-2:3i]
        for h in 2:3, d in 1:3
            lab[d, h] -= L * round((lab[d, h] - lab[d, 1]) / L)
        end
        MCe.molecule_from_sites!(sys, i, lab)
    end
    MCe.attach_ewald!(sys, ff; n2max=27)                      # the NIST wave-vector set (|n|² < 27)
    return sys, ff, (L=L, M=M, pos=pos, q=q, σ=σ, ε=ε, mol=mol)
end

function flat_energy(d, rc; drop::Int=0)
    keep = drop == 0 ? collect(1:3d.M) : [s for s in 1:3d.M if d.mol[s] != drop]
    ep = MCe.EwaldParams(d.L, 5.6 / d.L, rc, 27, KE_K_ANGSTROM)
    lj, _ = MCe.site_lj_energy(d.pos[:, keep], d.σ[keep], d.ε[keep], d.mol[keep], d.L, rc)
    return lj + MCe.ewald_energy(d.pos[:, keep], d.q[keep], d.mol[keep], ep).total
end

@testset "Ewald in RigidSystem = NIST-validated ewald_energy (total and removal energies)" begin
    rc = 8.0                                                   # < L/2 − extent: COM image = site minimum image (as the Wolf chain test)
    for cfg in (1, 2)
        sys, ff, d = nist_rigid_system(cfg, rc)
        U = MCe.total_pair_energy(sys, ff)
        Uref = flat_energy(d, rc)
        @test abs(U - Uref) / abs(Uref) < 1e-10
        for i in (1, 7, d.M)
            u = MCe.removal_energy(sys, ff, i)
            uref = Uref - flat_energy(d, rc; drop=i)
            @test abs(u - uref) < 1e-6                         # K; |U| ≈ 5e5 K, so a 1e-12 relative difference
        end
        @test isnan(MCe.pressure(sys, ff))                     # reciprocal virial not implemented: no silent pressure
        @test MCe.total_pair_energy_virial(sys, ff)[1] ≈ U rtol = 1e-12   # the MC sampling path (regression: it lacked the Fourier part)
        @info "Ewald RigidSystem vs flat, NIST config $cfg" U Uref MCe.n_kvectors(sys.recip)
    end
end

# small SPC/E box in reduced units with Ewald parameters from rtol = 1e-5 (GROMACS default ewald-rtol)
function ewald_water(N, seed; rc_angstrom=4.5, equil=50)
    ρ = 0.0334                                                 # molecules / Å³
    α = MCe.ewald_alpha_for(rc_angstrom, 1e-5)
    m = MCe.spce_model(; α_per_angstrom=α, rc_angstrom=rc_angstrom, coulomb=:ewald)
    L = cbrt(N / ρ) / m.σ_angstrom
    sys = MCe.RigidSystem([m.template], fill(1, N), L, m.ϵ_kelvin / 300.0, Xoshiro(seed))
    MCe.random_rigid_system!(sys, m.ff)
    MCe.attach_ewald!(sys, m.ff; rtol=1e-5)
    st = MCe.RigidMCStats()
    for _ in 1:equil, _ in 1:N
        MCe.mc_translate!(sys, m.ff, rand(sys.rng, 1:N), 0.1, st); MCe.mc_rotate!(sys, m.ff, rand(sys.rng, 1:N), 0.3, st)
    end
    return m, sys, st
end

committed_drift(sys) = (S = copy(sys.recip.S); e = copy(sys.recip.eimg); MCe.recip_rebuild!(sys.recip, sys);
                        max(maximum(abs.(S .- sys.recip.S)), maximum(abs.(e .- sys.recip.eimg))))

function ek_drift(ek, sys, ff)
    φ = copy(ek.phi); P = copy(ek.pair_u); fu = ek.frac_u
    MCe.rebuild_rigid_ekmc!(ek, sys, ff)
    return max(maximum(abs.(φ .- ek.phi)), maximum(abs.(P .- ek.pair_u)), abs(fu - ek.frac_u))
end

@testset "Ewald MC commits and eKMC bookkeeping: incremental = rebuild" begin
    m, sys, st = ewald_water(48, 11)
    ff = m.ff
    @test st.trans_acc > 0 && st.rot_acc > 0
    @test committed_drift(sys) < 1e-10                         # every accepted MC move committed its S_i
    ek = MCe.init_rigid_ekmc(sys, ff)
    U0 = 0.5 * sum(ek.phi) + 0.5 * MCe._image_energy_sum(ek, sys)
    @test U0 ≈ MCe.total_pair_energy(sys, ff) rtol = 1e-11      # ½Σφ + ½Σe = Σ pairs + Σ e
    acc = MCe.ChemicalPotentialAccumulator()
    for _ in 1:300
        MCe.rigid_ekmc_step!(ek, sys, ff, acc)
    end
    for _ in 1:3
        MCe.ekmc_mc_sweep!(ek, sys, ff, 0.1, 0.3)
    end
    @test ek_drift(ek, sys, ff) < 1e-9
    @test committed_drift(sys) < 1e-10
    # fractional molecule at λ = 0.4: pairs and image energy scaled, frac_u bare
    s = 5
    ek.frac = s; ek.lam[s] = 0.4
    MCe.rebuild_rigid_ekmc!(ek, sys, ff)
    @test ek.frac_u ≈ MCe.removal_energy(sys, ff, s) rtol = 1e-11
    for _ in 1:300
        k = MCe.select_by_mobility(ek, sys.rng)
        MCe.set_molecule!(sys, k, sys.L .* rand(sys.rng, 3), MCe.uniform_random_quaternion(sys.rng))
        MCe.update_rigid_ekmc_after_move!(ek, sys, ff, k)
    end
    MCe.set_fractional_coupling!(ek, sys.β, 0.7)
    MCe.set_molecule!(sys, s, sys.L .* rand(sys.rng, 3), MCe.uniform_random_quaternion(sys.rng))
    MCe.update_rigid_ekmc_after_move!(ek, sys, ff, s)
    MCe.ekmc_mc_sweep!(ek, sys, ff, 0.1, 0.3)
    @test ek_drift(ek, sys, ff) < 1e-9
    @test ek.frac_u ≈ MCe.removal_energy(sys, ff, s) rtol = 1e-10
end

@testset "Ewald: solvent and solute L-move candidates (Fourier part by ΔS products) = rebuild" begin
    m, sys, _ = ewald_water(40, 12)
    ff = m.ff
    N = sys.n_mol; s = N
    ek = MCe.init_rigid_ekmc(sys, ff)
    ek.frac = s; ek.lam[s] = 0.6
    MCe.rebuild_rigid_ekmc!(ek, sys, ff)
    S = MCe.SolventLocalMoves(sys, ff, collect(1:N-1); kinds=[MCe.LM_T, MCe.LM_R, MCe.LM_T], a=0.1, θmax=0.3, ν=1.0,
                              νD=1.0, frac=s)
    lm = MCe.LocalMoves(sys, s, 2; kinds=[MCe.LM_T, MCe.LM_R, MCe.LM_T], a=0.1, θmax=0.3, ν=1.0, νD=1.0)
    MCe.lm_rebuild!(lm, sys, ff)
    @test any(!=(0.0), S.cr_u) && any(!=(0.0), lm.cand_rec)
    for it in 1:60
        if it % 3 == 0                                         # solvent L-move
            MCe.slm_total_rate!(S, ek, sys.β)
            i = MCe.slm_fire!(S, sys, ff, ek, rand(sys.rng) * S.W)
            MCe.lm_molecule_moved!(lm, sys, ff, i)
        elseif it % 3 == 1                                     # relocation of a solvent molecule
            k = rand(sys.rng, 1:N-1)
            MCe.set_molecule!(sys, k, sys.L .* rand(sys.rng, 3), MCe.uniform_random_quaternion(sys.rng))
            MCe.update_rigid_ekmc_after_move!(ek, sys, ff, k)
            MCe.slm_moved!(S, sys, ff, k)
            MCe.lm_molecule_moved!(lm, sys, ff, k)
        else                                                   # L-move of the fractional molecule
            MCe.lm_total_rate!(lm, sys, ff, sys.β, ek.lam[s], ek.frac_u)
            MCe.lm_fire!(lm, sys, ff, ek, rand(sys.rng) * sum(lm.rates))
            MCe.slm_moved!(S, sys, ff, s)
            MCe.lm_rebuild!(lm, sys, ff)
        end
    end
    @test MCe.slm_check!(S, sys, ff) < 1e-9
    cu = copy(lm.cand_u); MCe.lm_rebuild!(lm, sys, ff)
    @test maximum(abs.(cu .- lm.cand_u)) < 1e-9
    @test ek_drift(ek, sys, ff) < 1e-9
    # the candidate energy of an L-move slot is the removal energy of s at that pose
    j = 2
    MCe._lm_save!(lm, sys); MCe._lm_place!(lm, sys, j)
    u_place = MCe.removal_energy(sys, ff, s)
    MCe._lm_restore!(lm, sys)
    @test lm.cand_u[j] ≈ u_place rtol = 1e-10
end

@testset "SPME reciprocal energy converges to the Ewald Fourier sum (Essmann 1995)" begin
    for cfg in (1, 4)
        L, M, pos, q, σ, ε, mol = read_nist_config(joinpath(NIST_DIR, "spce_sample_config_periodic$(cfg).txt"))
        α = 5.6 / L
        Eref = MCe.ewald_recip_energy(pos, q, MCe.EwaldParams(L, α, 10.0, 400, KE_K_ANGSTROM))   # converged k-sum
        err(K, p) = abs(MCe.pme_recip_energy(pos, q, MCe.PMEParams(L, α, KE_K_ANGSTROM, K, p)) - Eref) / abs(Eref)
        e1, e2, e3, e4 = err(16, 4), err(32, 6), err(48, 6), err(64, 8)
        @info "SPME vs Ewald, NIST config $cfg" e1 e2 e3 e4
        @test e1 > e2 > e3 > e4                               # spline/mesh error falls with K and p
        @test e4 < 1e-9                                       # (observed 2e-11 and 6e-11)
    end
    @test MCe.pme_grid_size(20.0 / 1.2) == 18 && MCe.pme_grid_size(25.0) == 25 && MCe.pme_grid_size(11) == 12
end

@testset "Ewald and SPME vs GROMACS: 500 SPC/E + OPLS-AA methanol (coulomb-modifier None, ewald-rtol 1e-5)" begin
    # tolerance 2e-5 relative: GROMACS mixed precision over ~10⁶ pair terms, as test_rf_gromacs.jl
    gro = readlines(joinpath(@__DIR__, "data", "rf_reference_spce_methanol.gro"))
    nat = parse(Int, strip(gro[2])); box = parse.(Float64, split(gro[3 + nat]))
    pos_nm = zeros(3, nat)
    for k in 1:nat
        l = gro[2 + k]; pos_nm[:, k] .= (parse(Float64, l[21:31]), parse(Float64, l[32:42]), parse(Float64, l[43:53]))
    end
    ref = Dict{String,Tuple{Float64,Float64}}()
    for l in eachline(joinpath(@__DIR__, "data", "ewald_reference_energies.txt"))
        startswith(l, "#") && continue
        f = split(l); ref[f[1]] = (parse(Float64, f[4]), parse(Float64, f[5]))
    end
    rc_A = 10.0; α_A = MCe.ewald_alpha_for(rc_A, 1e-5)
    @test 10α_A ≈ 3.123413 rtol = 1e-6                        # GROMACS ewald-rtol → β (grompp, 1 nm)
    m = MCe.spce_solute_model(:methanol; coulomb=:ewald, α_per_angstrom=α_A, rc_angstrom=rc_A, cutoff=:site, cap_overlaps=true)
    ff = m.ff; σ_A = m.σ_angstrom; kJ = m.ϵ_kelvin * 0.0083144626
    L = box[1] * 10 / σ_A
    sys = MCe.RigidSystem(m.templates, vcat(2, fill(1, 500)), L, m.ϵ_kelvin / 300.0, Xoshiro(1))
    k = 0
    for i in 1:sys.n_mol
        t = sys.templates[sys.tmpl[i]]
        MCe.molecule_from_sites!(sys, i, pos_nm[:, k+1:k+t.n_sites] .* (10 / σ_A)); k += t.n_sites
    end
    MCe.attach_ewald!(sys, ff; n2max=401)                      # |n| ≤ 20: Fourier sum converged far below 1e-5
    ffn = MCe.SiteForceField([ff.σ_mix[a, a] for a in 1:ff.n_types], [ff.ϵ_mix[a, a] for a in 1:ff.n_types]; rc=ff.rc,
                             coulomb=:none, cutoff=:site, mixing=:geometric, cap_overlaps=true)
    Uc = (MCe.total_pair_energy(sys, ff) - MCe.total_pair_energy(sys, ffn)) * kJ
    @test isapprox(Uc, sum(ref["ewald"]); rtol=2e-5)
    @test isapprox(Uc, sum(ref["pmefine"]); rtol=2e-5)
    pos, q, mol, _, _ = MCe.site_arrays(sys, ff)
    ep = MCe.EwaldParams(L, α_A * σ_A, ff.rc, 401, 1.0)
    e24 = MCe.pme_energy(pos, q, mol, ep, 24, 4)               # the GROMACS default grid for this box
    @test isapprox(e24.fourier * kJ, ref["pme"][2]; rtol=2e-5)
    @test isapprox(e24.total * kJ, sum(ref["pme"]); rtol=2e-5)
    @test isapprox((e24.real + e24.self + e24.intra) * kJ, ref["pme"][1]; rtol=2e-5)
    @info "Ewald/SPME vs GROMACS (kJ/mol)" Uc sum(ref["ewald"]) e24.fourier * kJ ref["pme"][2] e24.total * kJ sum(ref["pme"])
end

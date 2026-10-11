"""
Rigid molecules (LJ sites + Wolf charges): bookkeeping, detailed balance of the eKMC move, pressure
identity, and MC ↔ eKMC consistency. Tolerances: exact identities at round-off level; statistical
comparisons within 3 combined block-average standard errors (fixed seeds).
"""

using Test
using MolSim
using StaticArrays
using Random

const MCm = MolSim.MC
include(joinpath(@__DIR__, "nist_spce_reader.jl"))

# SPC/E in Å / K / e units (NIST conventions), for the energy chain test
function spce_template_angstrom()
    θ = 109.47 * π / 180
    sites = zeros(3, 3)
    sites[:, 2] .= (sin(θ / 2), cos(θ / 2), 0.0)
    sites[:, 3] .= (-sin(θ / 2), cos(θ / 2), 0.0)
    MCm.RigidTemplate("SPC/E-A", sites, [1, 0, 0], [-2 * SPCE_QH, SPCE_QH, SPCE_QH], [15.9994, 1.00794, 1.00794])
end

@testset "RigidSystem energy chain: NIST sites → RigidSystem → Electrostatics.jl routines" begin
    for cfg in (1, 2)
        L, M, pos, q, σ, ε, mol = read_nist_config(joinpath(NIST_DIR, "spce_sample_config_periodic$(cfg).txt"))
        tmpl = spce_template_angstrom()
        # r_c = 8 Å < L/2 − (molecular extent): the COM-consistent image used by RigidSystem and the
        # per-site minimum image used by Electrostatics.jl then select the same site pairs
        rc = 8.0
        ff = MCm.SiteForceField([SPCE_SIGMA], [SPCE_EPS]; rc=rc, coulomb=:wolf, α=0.16,
                                ke=KE_K_ANGSTROM, use_lrc=true, cap_overlaps=false)
        sys = MCm.RigidSystem([tmpl], fill(1, M), L, 1.0 / 300.0, Xoshiro(1))
        for i in 1:M
            lab = pos[:, 3i-2:3i]
            # unwrap H sites relative to O (same image)
            for h in 2:3, d in 1:3
                lab[d, h] -= L * round((lab[d, h] - lab[d, 1]) / L)
            end
            MCm.molecule_from_sites!(sys, i, lab)
        end
        # site positions reproduce the file (mod L) — checks quaternion fit and template geometry
        sp = MCm.site_positions(sys)
        maxdev = 0.0
        for s in 1:3M, d in 1:3
            δ = sp[d, s] - pos[d, s]; δ -= L * round(δ / L)
            maxdev = max(maxdev, abs(δ))
        end
        @test maxdev < 1e-8
        # energies against the NIST-validated flat-array routines
        lj, lrc = MCm.site_lj_energy(pos, σ, ε, mol, L, rc)
        wp = MCm.WolfParams(0.16, rc, KE_K_ANGSTROM; model=:wolf)
        Ep, Es, Ei = MCm.wolf_energy(pos, q, mol, L, wp)
        Upair = MCm.total_pair_energy(sys, ff)
        @test abs(Upair - (lj + Ep)) / abs(lj + Ep) < 1e-9
        @test abs(MCm.lrc_energy(sys, ff) - lrc) / abs(lrc) < 1e-9
        @test abs(MCm.wolf_constant_energy(sys, ff) - (Es + Ei)) / abs(Es + Ei) < 1e-9
        Utot = MCm.total_energy(sys, ff; with_constants=true)
        @test abs(Utot - (lj + lrc + Ep + Es + Ei)) / abs(Utot) < 1e-9
        # virial path gives the same energy
        Uv, W = MCm.total_pair_energy_virial(sys, ff)
        @test abs(Uv - Upair) / abs(Upair) < 1e-12
    end
end

function small_spce_system(N, ρstar, Tstar, seed; rc_angstrom=4.5, cap_overlaps=true)
    m = MCm.spce_model(; rc_angstrom=rc_angstrom, cap_overlaps=cap_overlaps)
    L = cbrt(N / ρstar)
    sys = MCm.RigidSystem([m.template], fill(1, N), L, 1.0 / Tstar, Xoshiro(seed))
    MCm.random_rigid_system!(sys, m.ff)
    return sys, m.ff, m
end

@testset "eKMC move: removal-energy invariant and detailed balance (SPC/E, Wolf)" begin
    sys, ff, m = small_spce_system(20, 0.6, 3.81, 7)
    β = sys.β
    for trial in 1:25
        k = rand(sys.rng, 1:sys.n_mol)
        U = MCm.total_pair_energy(sys, ff)
        uk = MCm.removal_energy(sys, ff, k)
        # independent removal energy: energy of the system without k
        sub = MCm.RigidSystem(sys.templates, deleteat!(copy(sys.tmpl), k), sys.L, β, Xoshiro(0))
        jj = 0
        for j in 1:sys.n_mol
            j == k && continue
            jj += 1
            MCm.set_molecule!(sub, jj, sys.com[:, j], sys.quat[:, j])
        end
        Uminus = MCm.total_pair_energy(sub, ff)
        @test abs((U - uk) - Uminus) <= 1e-9 * max(1.0, abs(U))
        # move k to a uniform random position and orientation (the eKMC move)
        L = sys.L
        com_new = MVector{3,Float64}(rand(sys.rng) * L, rand(sys.rng) * L, rand(sys.rng) * L)
        MCm.set_molecule!(sys, k, com_new, MCm.uniform_random_quaternion(sys.rng))
        U2 = MCm.total_pair_energy(sys, ff)
        uk2 = MCm.removal_energy(sys, ff, k)
        @test abs((U2 - uk2) - (U - uk)) <= 1e-9 * max(1.0, abs(U))
        # microscopic detailed balance: π(X) w(X→X') = π(X') w(X'→X) with w ∝ exp(β u_k)/(V 8π²)
        lhs = -β * U + β * uk
        rhs = -β * U2 + β * uk2
        @test abs(lhs - rhs) <= 1e-9 * max(1.0, abs(β * U))
    end
end

@testset "eKMC bookkeeping stays consistent over many events" begin
    sys, ff, m = small_spce_system(16, 0.6, 3.81, 11)
    ek = MCm.init_rigid_ekmc(sys, ff)
    acc = MCm.ChemicalPotentialAccumulator()
    for _ in 1:3000
        MCm.rigid_ekmc_step!(ek, sys, ff, acc)
    end
    phi = copy(ek.phi); R = ek.R
    MCm.rebuild_rigid_ekmc!(ek, sys, ff)
    @test maximum(abs.(phi .- ek.phi)) < 1e-8 * max(1.0, maximum(abs.(ek.phi)))
    @test abs(R - ek.R) / ek.R < 1e-8
    @test acc.t_total > 0 && isfinite(MCm.mu_ex(acc, 3.81))
end

@testset "Molecular virial pressure equals −dU/dV under COM scaling" begin
    # three water molecules close together in a large box, no cutoff crossings, no tail, no caps
    m = MCm.spce_model(; rc_angstrom=10.0, use_lrc=false, cap_overlaps=false)
    ff = m.ff
    L = 20.0
    sys = MCm.RigidSystem([m.template], [1, 1, 1], L, 1.0 / 3.81, Xoshiro(3))
    MCm.set_molecule!(sys, 1, [10.0, 10.0, 10.0], MCm.uniform_random_quaternion(sys.rng))
    MCm.set_molecule!(sys, 2, [10.9, 10.3, 10.1], MCm.uniform_random_quaternion(sys.rng))
    MCm.set_molecule!(sys, 3, [10.2, 10.95, 9.7], MCm.uniform_random_quaternion(sys.rng))
    U0, W = MCm.total_pair_energy_virial(sys, ff)
    V = L^3
    P_conf_virial = W / (3V)
    # central difference of U(V) with COM scaling (orientations fixed)
    function U_at_scale(s)
        s2 = MCm.RigidSystem(sys.templates, sys.tmpl, L * s, sys.β, Xoshiro(0))
        for i in 1:3
            MCm.set_molecule!(s2, i, sys.com[:, i] .* s, sys.quat[:, i])
        end
        MCm.total_pair_energy(s2, ff)
    end
    h = 1e-5
    Vp = V * (1 + h)^3; Vm = V * (1 - h)^3
    dUdV = (U_at_scale(1 + h) - U_at_scale(1 - h)) / (Vp - Vm)
    @test abs(P_conf_virial + dUdV) <= 1e-6 * abs(P_conf_virial)
end

function lj_diatomic_model(; bond=0.6, rc=2.5)
    sites = zeros(3, 2); sites[1, 1] = -bond / 2; sites[1, 2] = bond / 2
    tmpl = MCm.RigidTemplate("LJ2", sites, [1, 1], [0.0, 0.0], [1.0, 1.0])
    ff = MCm.SiteForceField([1.0], [1.0]; rc=rc, coulomb=:none)
    return tmpl, ff
end

function mc_vs_ekmc(templates, ff, N, ρ, T, seed; δ, δθ, equil_sweeps, sweeps, events,
                    sample_every=50, widom_per_sample=50)
    L = cbrt(N / ρ)
    sys = MCm.RigidSystem(templates, fill(1, N), L, 1 / T, Xoshiro(seed))
    MCm.random_rigid_system!(sys, ff)
    st = MCm.RigidMCStats()
    for _ in 1:equil_sweeps, _ in 1:N
        MCm.mc_translate!(sys, ff, rand(sys.rng, 1:N), δ, st)
        MCm.mc_rotate!(sys, ff, rand(sys.rng, 1:N), δθ, st)
    end
    mc = MCm.run_rigid_mc!(deepcopy(sys), ff; sweeps=sweeps, δ=δ, δθ=δθ, sample_every=1,
                           block_size=max(10, sweeps ÷ 40), widom_template=1, widom_per_sample=widom_per_sample)
    ek = MCm.run_rigid_ekmc!(sys, ff; events=events, sample_every=sample_every, block_events=events ÷ 25,
                             debug_check_every=events ÷ 10)
    w, ew = MCm.mean(mc.widom), MCm.stderr(mc.widom)
    rn, ern = MCm.mean(ek.RN), MCm.stderr(ek.RN)
    r = (U_mc=MCm.mean(mc.U), eU_mc=MCm.stderr(mc.U), U_ek=MCm.mean(ek.U), eU_ek=MCm.stderr(ek.U),
         P_mc=MCm.mean(mc.P), eP_mc=MCm.stderr(mc.P), P_ek=MCm.mean(ek.P), eP_ek=MCm.stderr(ek.P),
         μ_widom=-T * log(w), eμ_widom=T * ew / w, μ_ek=T * log(rn), eμ_ek=T * ern / rn,
         μ_ek_acc=MCm.mu_ex(ek.mu, T), t_total=ek.t_total)
    return r
end

within3σ(a, ea, b, eb) = abs(a - b) <= 3 * sqrt(ea^2 + eb^2)

@testset "Rigid LJ diatomic NVT: Metropolis MC vs eKMC (U, P) and Widom vs ⟨R/N⟩ (μ_ex)" begin
    # Supercritical state (ρ* = 0.3, T* = 2): eKMC residence times are not heavy-tailed here, so
    # 4e6 events suffice (dev experiment 2026-10-11, JOURNAL.md). Denser states need far longer runs
    # (slow suite / cluster validation).
    tmpl, ff = lj_diatomic_model()
    r = mc_vs_ekmc([tmpl], ff, 32, 0.3, 2.0, 3; δ=0.15, δθ=0.4, equil_sweeps=500, sweeps=4000,
                   events=4_000_000)
    @info "LJ diatomic N=32 ρ*=0.3 T*=2" r
    @test within3σ(r.U_mc, r.eU_mc, r.U_ek, r.eU_ek)
    @test within3σ(r.P_mc, r.eP_mc, r.P_ek, r.eP_ek)
    @test within3σ(r.μ_widom, r.eμ_widom, r.μ_ek, r.eμ_ek)
    @test abs(r.μ_ek_acc - r.μ_ek) <= 3 * r.eμ_ek
end

if get(ENV, "MOLSIM_SLOW_TESTS", "0") == "1"
    @testset "SPC/E (Wolf) NVT, supercritical: Metropolis MC vs eKMC (slow)" begin
        m = MCm.spce_model(; rc_angstrom=4.5)
        r = mc_vs_ekmc([m.template], m.ff, 64, 0.3, 10.0, 5; δ=0.15, δθ=0.4, equil_sweeps=500,
                       sweeps=6000, events=10_000_000)
        @info "SPC/E N=64 ρ*=0.3 T*=10" r
        @test within3σ(r.U_mc, r.eU_mc, r.U_ek, r.eU_ek)
        @test within3σ(r.P_mc, r.eP_mc, r.P_ek, r.eP_ek)
        @test within3σ(r.μ_widom, r.eμ_widom, r.μ_ek, r.eμ_ek)
    end
end

# ============================================================================
# NPT eKMC for rigid molecules
# ============================================================================

@testset "NPT eKMC volume moves keep the pair bookkeeping consistent" begin
    tmpl, ff = lj_diatomic_model()
    N = 32
    sys = MCm.RigidSystem([tmpl], fill(1, N), cbrt(N / 0.3), 1 / 2.5, Xoshiro(9))
    MCm.random_rigid_system!(sys, ff)
    ek = MCm.init_rigid_ekmc(sys, ff)
    # maintained virial equals the direct one
    _, W = MCm.total_pair_energy_virial(sys, ff)
    @test abs(MCm.ekmc_virial(ek) - W) <= 1e-10 * max(1.0, abs(W))
    acc = MCm.ChemicalPotentialAccumulator()
    V0 = sys.L^3
    for trial in 1:20
        for _ in 1:50
            MCm.rigid_ekmc_step!(ek, sys, ff, acc)
        end
        if isodd(trial)
            MCm.rigid_ekmc_volume_move!(ek, sys, ff, 0.5, 0.05 * V0)          # Tan Eq. 12 rule
        else
            p̄ = MCm.pressure(sys, ff, MCm.ekmc_virial(ek))
            MCm.rigid_ekmc_newton_volume_move!(ek, sys, ff, 0.5, 0.05 * V0, p̄, -1.0)   # Kelly Eq. 3.3 step
        end
        @test all(0.0 .<= sys.com .< sys.L)
        phi = copy(ek.phi); wsum = copy(ek.wsum); R = ek.R
        MCm.rebuild_rigid_ekmc!(ek, sys, ff)
        @test maximum(abs.(phi .- ek.phi)) <= 1e-9 * max(1.0, maximum(abs.(ek.phi)))
        @test maximum(abs.(wsum .- ek.wsum)) <= 1e-9 * max(1.0, maximum(abs.(ek.wsum)))
        @test abs(R - ek.R) / ek.R <= 1e-9
        Udirect = MCm.total_pair_energy(sys, ff)
        @test abs(0.5 * sum(ek.phi) - Udirect) <= 1e-9 * max(1.0, abs(Udirect))
    end
end

function mc_vs_ekmc_npt(templates, ff, N, T, Pext, seed; δ, δθ, equil_sweeps, sweeps, events,
                        ρ0=0.3, vol_every_sweeps=10, sample_every=20, widom_per_sample=50)
    sys = MCm.RigidSystem(templates, fill(1, N), cbrt(N / ρ0), 1 / T, Xoshiro(seed))
    MCm.random_rigid_system!(sys, ff)
    st = MCm.RigidMCStats()
    for _ in 1:equil_sweeps
        for _ in 1:N
            MCm.mc_translate!(sys, ff, rand(sys.rng, 1:N), δ, st)
            MCm.mc_rotate!(sys, ff, rand(sys.rng, 1:N), δθ, st)
        end
        U = MCm.total_energy(sys, ff)
        MCm.mc_volume_trial!(sys, ff, Pext, 0.1, st; U_current=U)
    end
    mc = MCm.run_rigid_mc!(deepcopy(sys), ff; sweeps=sweeps, δ=δ, δθ=δθ, Pext=Pext, max_dlnV=0.1,
                           vol_every=1, sample_every=1, block_size=max(10, sweeps ÷ 40),
                           widom_template=1, widom_per_sample=widom_per_sample)
    V0 = sys.L^3
    ek = MCm.run_rigid_ekmc!(sys, ff; events=events, Pext=Pext, max_dV=0.05 * V0,
                             vol_every_sweeps=vol_every_sweeps, sample_every=sample_every,
                             block_events=events ÷ 25, debug_check_every=events ÷ 10)
    V̄ = MCm.mean(mc.V); wV, ewV = MCm.mean(mc.widomV), MCm.stderr(mc.widomV)
    rnv, ernv = MCm.mean(ek.RNV), MCm.stderr(ek.RNV)
    Vm, eVm = MCm.mean(ek.V), MCm.stderr(ek.V)
    return (rho_mc=MCm.mean(mc.rho), erho_mc=MCm.stderr(mc.rho), rho_ek=MCm.mean(ek.rho), erho_ek=MCm.stderr(ek.rho),
            U_mc=MCm.mean(mc.U), eU_mc=MCm.stderr(mc.U), U_ek=MCm.mean(ek.U), eU_ek=MCm.stderr(ek.U),
            P_mc=MCm.mean(mc.P), eP_mc=MCm.stderr(mc.P), P_ek=MCm.mean(ek.P), eP_ek=MCm.stderr(ek.P),
            μ_widom=-T * log(wV / V̄), eμ_widom=T * ewV / wV,
            μ_ek=T * log(rnv * Vm), eμ_ek=T * sqrt((ernv / rnv)^2 + (eVm / Vm)^2),
            μ_ek_acc=MCm.mu_ex_npt(ek.mu_npt, T), vol_moves=ek.vol_moves)
end

@testset "Rigid LJ diatomic NPT: Metropolis MC vs eKMC (ρ, U, P) and NPT Widom vs Eq. A.132 μ" begin
    # Cut-and-shifted LJ sites, no tail correction: the virial pressure is then exact, so <P> = P*
    # in MC (the truncated-potential impulsive term and the energy-tail/virial-tail route
    # inconsistency in NPT acceptance are both avoided; cf. repo commit e3eea0d for atoms).
    # Supercritical state T* = 4, P* = 1 (dev experiment 2026-10-11: ρ* ≈ 0.21, both methods converge).
    tmpl, ff = lj_diatomic_model()
    ff = MCm.SiteForceField([1.0], [1.0]; rc=2.5, coulomb=:none, shift_lj=true, use_lrc=false)
    T = 4.0; Pext = 1.0
    r = mc_vs_ekmc_npt([tmpl], ff, 32, T, Pext, 3; δ=0.2, δθ=0.5, equil_sweeps=600, sweeps=4000,
                       events=4_000_000)
    @info "LJ diatomic NPT N=32 T*=4 P*=1" r
    @test within3σ(r.P_mc, r.eP_mc, Pext, 0.0)            # MC NPT self-consistency
    @test within3σ(r.P_ek, r.eP_ek, Pext, 0.0)            # eKMC volume controller pins <P> = P*
    @test within3σ(r.rho_mc, r.erho_mc, r.rho_ek, r.erho_ek)
    @test within3σ(r.U_mc, r.eU_mc, r.U_ek, r.eU_ek)
    @test within3σ(r.μ_widom, r.eμ_widom, r.μ_ek, r.eμ_ek)
    @test abs(r.μ_ek_acc - r.μ_ek) <= 3 * r.eμ_ek
    @test r.vol_moves == 4_000_000 ÷ (10 * 32)
end

if get(ENV, "MOLSIM_SLOW_TESTS", "0") == "1"
    @testset "SPC/E (Wolf) NPT, supercritical: Metropolis MC vs eKMC (slow)" begin
        # N = 64, T* = 12 (938 K), P* = 3.0, cut-and-shifted LJ, no tail, Wolf r_c = 4.5 Å.
        # Dev experiment 2026-10-11: MC ρ 0.3884(62), U/N −29.33(38), P 3.04(8), μ −12.46(36);
        # eKMC (Newton + Robbins–Monro) ρ 0.3869, U/N −29.31(7), P 2.90(5), μ −13.28(7).
        # The eKMC density after the Robbins–Monro phase has a controller uncertainty that the block
        # error does not capture, so ρ is compared with the MC error only (DESIGN.md §3c, open item:
        # a balanced volume move for eKMC).
        m = MCm.spce_model(; rc_angstrom=4.5, use_lrc=false, shift_lj=true)
        T = 12.0; Pext = 3.0; N = 64; events = 10_000_000
        burn = 200 * 10 * N + events ÷ 5
        sys = MCm.RigidSystem([m.template], fill(1, N), cbrt(N / 0.3), 1 / T, Xoshiro(7))
        MCm.random_rigid_system!(sys, m.ff)
        st = MCm.RigidMCStats()
        for _ in 1:600
            for _ in 1:N
                MCm.mc_translate!(sys, m.ff, rand(sys.rng, 1:N), 0.15, st)
                MCm.mc_rotate!(sys, m.ff, rand(sys.rng, 1:N), 0.4, st)
            end
            U = MCm.total_energy(sys, m.ff)
            MCm.mc_volume_trial!(sys, m.ff, Pext, 0.1, st; U_current=U)
        end
        mc = MCm.run_rigid_mc!(deepcopy(sys), m.ff; sweeps=6000, δ=0.15, δθ=0.4, Pext=Pext, max_dlnV=0.1,
                               vol_every=1, sample_every=1, block_size=150, widom_template=1, widom_per_sample=50)
        V0 = sys.L^3
        ek = MCm.run_rigid_ekmc!(sys, m.ff; events=events, Pext=Pext, max_dV=0.05 * V0, vol_every_sweeps=10,
                                 sample_every=20, block_events=(events - burn) ÷ 25, vol_equil_moves=200,
                                 burnin_events=burn, debug_check_every=events ÷ 10)
        V̄ = MCm.mean(mc.V); wV, ewV = MCm.mean(mc.widomV), MCm.stderr(mc.widomV)
        rnv, ernv = MCm.mean(ek.RNV), MCm.stderr(ek.RNV); Vm, eVm = MCm.mean(ek.V), MCm.stderr(ek.V)
        μ_mc, eμ_mc = -T * log(wV / V̄), T * ewV / wV
        μ_ek, eμ_ek = T * log(rnv * Vm), T * sqrt((ernv / rnv)^2 + (eVm / Vm)^2)
        @info "SPC/E NPT N=64 T*=12 P*=3" MCm.mean(mc.rho) MCm.stderr(mc.rho) MCm.mean(ek.rho) MCm.mean(mc.U) MCm.stderr(mc.U) MCm.mean(ek.U) MCm.stderr(ek.U) MCm.mean(mc.P) MCm.stderr(mc.P) MCm.mean(ek.P) MCm.stderr(ek.P) μ_mc eμ_mc μ_ek eμ_ek
        @test within3σ(MCm.mean(mc.P), MCm.stderr(mc.P), Pext, 0.0)
        @test within3σ(MCm.mean(ek.P), MCm.stderr(ek.P), Pext, 0.0)
        @test abs(MCm.mean(mc.rho) - MCm.mean(ek.rho)) <= 3 * MCm.stderr(mc.rho)
        @test within3σ(MCm.mean(mc.U), MCm.stderr(mc.U), MCm.mean(ek.U), MCm.stderr(ek.U))
        @test within3σ(μ_mc, eμ_mc, μ_ek, eμ_ek)
    end
end

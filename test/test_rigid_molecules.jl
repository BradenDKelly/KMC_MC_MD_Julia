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
    # The eKMC volume controller (Newton + Robbins–Monro) converges to the volume where the block-
    # averaged pressure equals P*; its residual is a controller error that the fixed-volume block error
    # of P does not measure (DESIGN.md §5b). The pinning precision cannot beat the precision with which
    # a pressure of this run length is measurable, so the yardstick is the MC pressure error of the
    # same-length MC run: |<P>_eKMC − P*| ≤ 3 σ_P(MC). Observed residuals: −1.1 % to +0.8 % of P*.
    @test abs(r.P_ek - Pext) <= 3 * r.eP_mc
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

# ============================================================================
# Global time-weighted ratio, jackknife error, per-species μ at infinite dilution (2026-10-12)
# ============================================================================

@testset "Time-weighted ratio estimator and delete-one-block jackknife" begin
    # equal block times: the jackknife error of Σx/Σt is exactly the standard error of the block means
    b = MCm.TimeWeightedBlockAverager(1)
    xs = 10.0 .+ [0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.15, -0.05]
    for x in xs
        push!(b, x, 1.0)
    end
    @test MCm.ratio_mean(b) ≈ sum(xs) / length(xs)
    @test MCm.ratio_jackknife_stderr(b) ≈ MCm.stderr(b)
    # f = log: agrees with the delta method to first order (relative errors here are ≈ 1 %)
    @test MCm.ratio_jackknife_stderr(b; f=log) ≈ MCm.ratio_jackknife_stderr(b) / MCm.ratio_mean(b) rtol=1e-2
    # constant x with very unequal block times: the ratio is exact and its error zero, whereas the
    # mean of block ratios is also exact here but would not be for x varying with t
    b2 = MCm.TimeWeightedBlockAverager(2)
    for (x, t) in zip([3.0, 3.0, 3.0, 3.0, 3.0, 3.0], [0.1, 2.0, 0.5, 7.0, 0.01, 1.0])
        push!(b2, x, t)
    end
    @test length(b2.block_sum_x) == 3 && MCm.ratio_mean(b2) ≈ 3.0
    @test MCm.ratio_jackknife_stderr(b2) ≈ 0.0 atol=1e-12
    # unequal block times: ratio of sums ≠ mean of block ratios (the biased blocked estimator)
    b3 = MCm.TimeWeightedBlockAverager(1)
    push!(b3, 1.0, 1.0); push!(b3, 3.0, 9.0)
    @test MCm.ratio_mean(b3) ≈ 2.8 && MCm.mean(b3) ≈ 2.0
    @test isnan(MCm.ratio_jackknife_stderr(MCm.TimeWeightedBlockAverager(1)))
end

@testset "TraPPE-UA solutes in SPC/E reduced units: geometry, charges, Lorentz–Berthelot" begin
    m = MCm.spce_solute_model(:methanol_ua)
    t = m.solute; σ = m.σ_angstrom
    d(a, b) = sqrt(sum(abs2, t.sites_body[:, a] .- t.sites_body[:, b])) * σ
    @test d(1, 2) ≈ 1.43 atol=1e-9                       # C–O
    @test d(2, 3) ≈ 0.945 atol=1e-9                      # O–H
    v1 = t.sites_body[:, 1] .- t.sites_body[:, 2]; v2 = t.sites_body[:, 3] .- t.sites_body[:, 2]
    @test acosd(sum(v1 .* v2) / sqrt(sum(abs2, v1) * sum(abs2, v2))) ≈ 108.5 atol=1e-9
    @test sum(t.site_q) ≈ 0.0 atol=1e-12
    @test t.site_q ./ m.q_scale ≈ [0.265, -0.700, 0.435]
    @test t.site_type == [2, 3, 0] && m.ff.n_types == 3
    @test m.ff.σ_mix[1, 2] * σ ≈ (3.16555789 + 3.75) / 2 && m.ff.σ_mix[1, 3] * σ ≈ (3.16555789 + 3.02) / 2
    @test m.ff.ϵ_mix[1, 2] * m.ϵ_kelvin ≈ sqrt(78.19743111 * 98.0) && m.ff.ϵ_mix[3, 3] * m.ϵ_kelvin ≈ 93.0
    mm = MCm.spce_solute_model(:methane_ua)
    @test mm.solute.n_sites == 1 && mm.solute.site_q == [0.0] && mm.ff.n_types == 2
    @test mm.ff.σ_mix[2, 2] * σ ≈ 3.73 && mm.ff.ϵ_mix[2, 2] * mm.ϵ_kelvin ≈ 148.0
    @test mm.ff.ucap_mix[1, 2] ≈ 1000.0 * mm.ff.ϵ_mix[1, 2]
    # the water template and force-field constants are those of spce_model
    m0 = MCm.spce_model(; α_per_angstrom=0.20, cutoff=:molecule)
    @test m.water.sites_body == m0.template.sites_body && m.ff.wolf.α == m0.ff.wolf.α && m.ff.rc == m0.ff.rc
    @test_throws ErrorException MCm.spce_solute_model(:ethanol)
end

@testset "OPLS-AA rigid all-atom methane and methanol in SPC/E units: geometry, charges, geometric mixing" begin
    kB = 0.0083144626
    m = MCm.spce_solute_model(:methane); t = m.solute; σ = m.σ_angstrom
    d(t, a, b) = sqrt(sum(abs2, t.sites_body[:, a] .- t.sites_body[:, b])) * σ
    @test t.n_sites == 5 && t.site_type == [2, 3, 3, 3, 3]
    @test all(d(t, 1, k) ≈ 1.090 for k in 2:5)
    @test all(d(t, a, b) ≈ 1.090 * sqrt(8 / 3) for a in 2:5, b in 2:5 if a < b)     # tetrahedral H–H
    @test sum(t.site_q) ≈ 0.0 atol=1e-12
    @test t.site_q[1] / m.q_scale ≈ -0.24
    @test isapprox(m.ff.ϵ_mix[2, 2] * m.ϵ_kelvin * kB, 0.276144; rtol=1e-4) && isapprox(m.ff.ϵ_mix[3, 3] * m.ϵ_kelvin * kB, 0.12552; rtol=1e-4)
    @test m.ff.σ_mix[1, 2] * σ ≈ sqrt(3.16555789 * 3.50) && m.ff.σ_mix[1, 3] * σ ≈ sqrt(3.16555789 * 2.50)   # geometric
    @test m.mixing === :geometric
    mo = MCm.spce_solute_model(:methanol); t = mo.solute; σ = mo.σ_angstrom
    @test t.n_sites == 6 && t.site_type == [2, 3, 3, 3, 4, 0]
    @test isapprox(d(t, 1, 5), 1.41189; atol=2e-4) && isapprox(d(t, 5, 6), 0.94569; atol=2e-4)   # C–O, O–H (OPLS minimum)
    @test all(isapprox(d(t, 1, k), 1.0904; atol=2e-4) for k in 2:4)
    @test sum(t.site_q) ≈ 0.0 atol=1e-12
    @test t.site_q[5] / mo.q_scale ≈ -0.683 && t.site_q[6] / mo.q_scale ≈ 0.418
    @test isapprox(mo.ff.ϵ_mix[4, 4] * mo.ϵ_kelvin * kB, 0.71128; rtol=1e-4) && mo.ff.σ_mix[4, 4] * σ ≈ 3.12
    @test mo.ff.σ_mix[1, 4] * σ ≈ sqrt(3.16555789 * 3.12)
    # the template is centred on the COM
    @test maximum(abs.(t.sites_body * t.site_mass)) <= 1e-12 * sum(t.site_mass)
    # Lorentz–Berthelot stays the default for the generic constructor
    ff = MCm.SiteForceField([1.0, 2.0], [1.0, 1.0]; rc=2.5)
    @test ff.σ_mix[1, 2] == 1.5
    @test_throws ErrorException MCm.SiteForceField([1.0], [1.0]; rc=2.5, mixing=:arithmetic)
end

@testset "Infinite dilution: one LJ solute in the LJ-diatomic solvent — per-species eKMC vs Widom (pure solvent)" begin
    # Supercritical state (ρ* = 0.3, T* = 2, 32 diatomics) as in the NVT test above. The eKMC per-species
    # estimator exp(βμ_s) = ⟨exp(βu_s)⟩ for the one real solute (particle removal with kMC time weights)
    # must equal Widom insertion of the solute into the PURE solvent (Frenkel & Smit §7.2.1): both are the
    # infinite-dilution μ_ex. Widom into the system that already holds the real solute is NOT the same
    # quantity at this size (the real solute excludes ≈ 5 % of the volume to the ghost: +0.2 in μ at
    # N = 32, dev diagnostic 2026-10-12), so the reference is a separate pure-solvent run. The Metropolis
    # particle-removal average of the same observable fails: 85 % of ⟨exp(βu_s)⟩ comes from overlap
    # configurations with ≈ 3e-12 of the canonical weight, which canonical MC never visits but the eKMC
    # relocation move samples with the correct time weights (Shing & Gubbins, Mol. Phys. 46 (1982) 1109).
    # No tail correction on either side (same constant).
    tmpl2, _ = lj_diatomic_model()
    sol = MCm.RigidTemplate("LJ1", zeros(3, 1), [2], [0.0], [1.0])
    ff = MCm.SiteForceField([1.0, 1.2], [1.0, 0.7]; rc=2.5, coulomb=:none)
    N = 32; ρ = 0.3; T = 2.0; L = cbrt(N / ρ); Nt = N + 1
    function equilibrated(tm, seed)
        sys = MCm.RigidSystem([tmpl2, sol], tm, L, 1 / T, Xoshiro(seed))
        MCm.random_rigid_system!(sys, ff)
        st = MCm.RigidMCStats(); n = length(tm)
        for _ in 1:500, _ in 1:n
            MCm.mc_translate!(sys, ff, rand(sys.rng, 1:n), 0.15, st)
            MCm.mc_rotate!(sys, ff, rand(sys.rng, 1:n), 0.4, st)
        end
        return sys
    end
    # reference: Widom insertion of the solute into the pure solvent
    mc0 = MCm.run_rigid_mc!(equilibrated(fill(1, N), 11), ff; sweeps=4000, δ=0.15, δθ=0.4, sample_every=1,
                            block_size=100, widom_template=2, widom_per_sample=50)
    w, ew = MCm.mean(mc0.widom), MCm.stderr(mc0.widom)
    μ_w, eμ_w = -T * log(w), T * ew / w
    # solvent + 1 real solute: MC (U, P, removal estimator) and eKMC (U, P, per-species μ)
    sys = equilibrated(vcat(fill(1, N), 2), 11)
    mc = MCm.run_rigid_mc!(deepcopy(sys), ff; sweeps=4000, δ=0.15, δθ=0.4, sample_every=1, block_size=100,
                           inverse_widom_molecule=Nt)
    iw, eiw = MCm.mean(mc.inv_widom), MCm.stderr(mc.inv_widom)
    μ_r, eμ_r = T * log(iw), T * eiw / iw
    events = 4_000_000
    ek = MCm.run_rigid_ekmc!(sys, ff; events=events, sample_every=1, block_events=events ÷ 25,
                             debug_check_every=events ÷ 10, species_mu=true, trace_molecule=Nt)
    rs = ek.RN_species[2]
    μ_ek, eμ_ek = T * log(MCm.ratio_mean(rs)), T * MCm.ratio_jackknife_stderr(rs; f=log)
    @info "LJ solute at infinite dilution, N=32+1 ρ*=0.3 T*=2" μ_w eμ_w μ_ek eμ_ek μ_r eμ_r MCm.mean(mc.U) MCm.mean(ek.U) MCm.mean(mc.P) MCm.mean(ek.P)
    @test within3σ(μ_w, eμ_w, μ_ek, eμ_ek)
    @test within3σ(MCm.mean(mc.U), MCm.stderr(mc.U), MCm.mean(ek.U), MCm.stderr(ek.U))
    @test within3σ(MCm.mean(mc.P), MCm.stderr(mc.P), MCm.mean(ek.P), MCm.stderr(ek.P))
    @test μ_r < μ_w - 1.0            # the Metropolis removal estimator misses the overlap region (see above)
    # bookkeeping identities: R·N_t = Σ_s N_s ⟨R_s/N_s⟩ sample by sample; the trace reproduces the estimator
    @test MCm.ratio_mean(ek.RN) * Nt ≈ N * MCm.ratio_mean(ek.RN_species[1]) + MCm.ratio_mean(rs) rtol=1e-10
    @test length(ek.trace_phi) == length(ek.trace_dt) == events
    @test MCm.ratio_mean(rs) ≈ sum(MCm._mobility(1 / T, ek.trace_phi[k]) * ek.trace_dt[k] for k in eachindex(ek.trace_phi)) /
                               sum(ek.trace_dt) rtol=1e-10
    @test isempty(MCm.run_rigid_ekmc!(deepcopy(sys), ff; events=100, sample_every=10, block_events=50).RN_species)
end

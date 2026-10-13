# Steepest-descent minimisation of random starting configurations (minimize_rigid!, JOURNAL entry 24/25).
using Test
using MolSim
using StaticArrays
using Random
const MCmin = MolSim.MC

function _ncapped(sys, ff)
    c = minimum(ff.ucap_mix) * 0.999
    return count(MCmin.pair_energy(sys, ff, i, j) >= c for i in 1:sys.n_mol-1 for j in i+1:sys.n_mol)
end

@testset "rigid forces and torques = −∇U (finite differences)" begin
    for (name, m) in (("LJ", nothing), ("SPC/E RF", MCmin.spce_model(; coulomb=:rf, epsilon_rf=Inf, cutoff=:site)))
        if m === nothing
            tmpl = MCmin.RigidTemplate("LJ", zeros(3, 1), [1], [0.0], [1.0])
            ff = MCmin.SiteForceField([1.0], [1.0]; rc=2.5, coulomb=:none); N = 40; L = cbrt(N / 0.6)
        else
            tmpl = m.template; ff = m.ff; N = 40; L = 3.0 * (m.ff.rc + 0.4)   # dilute enough for no caps
        end
        sys = MCmin.RigidSystem([tmpl], fill(1, N), L, 1.0, Xoshiro(4))
        MCmin.random_rigid_system!(sys, ff)
        ffu = MCmin.uncapped(ff)
        F = zeros(3, N); τ = zeros(3, N)
        U = MCmin.rigid_forces_torques!(F, τ, sys, ffu)
        @test U ≈ MCmin.total_pair_energy(sys, ffu) rtol = 1e-12
        h = 1e-6; maxerr = 0.0; scale = maximum(abs, F)
        for i in (1, 7, 23), k in 1:3
            com = [sys.com[1, i], sys.com[2, i], sys.com[3, i]]; q = sys.quat[:, i]
            cp = copy(com); cp[k] += h; MCmin.set_molecule!(sys, i, cp, q); Up = MCmin.total_pair_energy(sys, ffu)
            cm = copy(com); cm[k] -= h; MCmin.set_molecule!(sys, i, cm, q); Um = MCmin.total_pair_energy(sys, ffu)
            MCmin.set_molecule!(sys, i, com, q)
            maxerr = max(maxerr, abs(-(Up - Um) / 2h - F[k, i]))
            if sys.ext[i] > 0                       # torque: dU/dθ about lab axis e_k = −τ_k
                e = zeros(3); e[k] = 1.0
                qp = MCmin.quat_mul(MCmin._axis_angle_quaternion(e..., h), q); MCmin.set_molecule!(sys, i, com, qp)
                Up = MCmin.total_pair_energy(sys, ffu)
                qm = MCmin.quat_mul(MCmin._axis_angle_quaternion(e..., -h), q); MCmin.set_molecule!(sys, i, com, qm)
                Um = MCmin.total_pair_energy(sys, ffu)
                MCmin.set_molecule!(sys, i, com, q)
                maxerr = max(maxerr, abs(-(Up - Um) / 2h - τ[k, i]))
            end
        end
        @info "force/torque finite-difference check" name maxerr scale
        @test maxerr <= 1e-5 * max(1.0, scale)
    end
end

@testset "minimize_rigid! removes the overlap trap of random starts (LJ ρ*=0.85 T*=0.9)" begin
    tmpl = MCmin.RigidTemplate("LJ", zeros(3, 1), [1], [0.0], [1.0])
    ff = MCmin.SiteForceField([1.0], [1.0]; rc=2.5, coulomb=:none)
    N = 108; ρ = 0.85; T = 0.9; L = cbrt(N / ρ)
    # regression: seed 3 without minimisation falls into a capped pair in the first sweeps (cap 100 ε at
    # r = 0.75 σ is below U_LJ(0.75 σ) = 104 ε) and stays there; with minimisation no capped pair appears
    trapped = 0; clean = 0
    for seed in 1:16
        for minimize in (false, true)
            sys = MCmin.RigidSystem([tmpl], fill(1, N), L, 1 / T, Xoshiro(seed))
            MCmin.random_rigid_system!(sys, ff)
            if minimize
                r = MCmin.minimize_rigid!(sys, ff; nsteps=500)
                @test r.U_end < r.U_start && r.n_capped == 0
            end
            st = MCmin.RigidMCStats()
            for _ in 1:300, _ in 1:N
                MCmin.mc_translate!(sys, ff, rand(sys.rng, 1:N), 0.12, st)
            end
            n = _ncapped(sys, ff)
            minimize ? (clean += (n == 0)) : (trapped += (n > 0))
            seed == 3 && !minimize && @test n > 0          # the defect this fixes (fails if the trap disappears for another reason)
        end
    end
    @info "random LJ starts after 300 MC sweeps" trapped_without_minimiser = trapped clean_with_minimiser = clean
    @test clean == 16
end

@testset "minimize_rigid! on SPC/E (RF) liquid start" begin
    m = MCmin.spce_model(; coulomb=:rf, epsilon_rf=Inf, cutoff=:site)
    N = 64
    ρ = 1000.0 * 1e3 / 18.0153 * 6.02214129e23 / 1e30 * m.σ_angstrom^3
    sys = MCmin.RigidSystem([m.template], fill(1, N), cbrt(N / ρ), m.ϵ_kelvin / 300.0, Xoshiro(2))
    MCmin.random_rigid_system!(sys, m.ff)
    r = MCmin.minimize_rigid!(sys, m.ff; nsteps=300)
    @test r.U_end < r.U_start
    @test r.n_capped == 0
    @test MCmin.total_pair_energy(sys, m.ff) ≈ r.U_end rtol = 1e-10   # no caps left ⇒ capped = bare energy
end

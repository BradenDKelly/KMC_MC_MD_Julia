"""
Wolf with a molecule-based cutoff vs Ewald on the NIST SPC/E configurations: Coulomb energy and
configurational Coulomb pressure (−dE/dV under COM scaling) over a grid of α at r_c = 10 Å (and 8 Å),
using the RigidSystem path (cutoff=:molecule) with LJ switched off. Companion of
dev/wolf_parameter_scan.jl / dev/wolf_pressure_scan.jl (site-based cutoff).

Run:  julia --project=. dev/wolf_molcut_scan.jl
"""

using MolSim, Printf, Random, StaticArrays
const MCm = MolSim.MC
include(joinpath(@__DIR__, "..", "test", "nist_spce_reader.jl"))
const ATM_PER_K_A3 = 1.3806488e-23 / 1e-30 / 101325.0

function spce_template_A()
    θ = 109.47 * π / 180
    sites = zeros(3, 3); sites[:, 2] .= (sin(θ/2), cos(θ/2), 0.0); sites[:, 3] .= (-sin(θ/2), cos(θ/2), 0.0)
    MCm.RigidTemplate("SPC/E-A", sites, [1, 0, 0], [-2SPCE_QH, SPCE_QH, SPCE_QH], [15.9994, 1.00794, 1.00794])
end

function build(cfg)
    L, M, pos, q, σ, ε, mol = read_nist_config(joinpath(NIST_DIR, "spce_sample_config_periodic$(cfg).txt"))
    tmpl = spce_template_A()
    sys = MCm.RigidSystem([tmpl], fill(1, M), L, 1 / 300.0, Xoshiro(1))
    for i in 1:M
        lab = pos[:, 3i-2:3i]
        for h in 2:3, d in 1:3; lab[d, h] -= L * round((lab[d, h] - lab[d, 1]) / L); end
        MCm.molecule_from_sites!(sys, i, lab)
    end
    return sys, L, M, pos, q, mol
end

# Coulomb-only force field (ε = 0 switches LJ off; caps off)
coul_ff(α, rc, cutoff) = MCm.SiteForceField([SPCE_SIGMA], [0.0]; rc=rc, coulomb=:wolf, α=α, ke=KE_K_ANGSTROM,
                                            use_lrc=false, cap_overlaps=false, cutoff=cutoff)

function E_wolf(sys, ff)
    MCm.total_pair_energy(sys, ff) + MCm.wolf_constant_energy(sys, ff)
end

function P_fd(sys, ff; h=1e-4)
    function E_at(s)
        s2 = MCm.RigidSystem(sys.templates, sys.tmpl, sys.L * s, sys.β, Xoshiro(0))
        for i in 1:sys.n_mol; MCm.set_molecule!(s2, i, sys.com[:, i] .* s, sys.quat[:, i]); end
        E_wolf(s2, ff)
    end
    Vp = (sys.L * (1 + h))^3; Vm = (sys.L * (1 - h))^3
    -(E_at(1 + h) - E_at(1 - h)) / (Vp - Vm) * ATM_PER_K_A3
end

alphas = [0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.25, 0.30]
for cfg in 1:4
    sys, L, M, pos, q, mol = build(cfg)
    α_ew = cfg == 4 ? 0.28 : 5.6 / L; n2 = cfg == 4 ? 100 : 27     # converged Ewald also for the 30 Å box
    ep = MCm.EwaldParams(L, α_ew, 10.0, n2, KE_K_ANGSTROM)
    E_ew = MCm.ewald_energy(pos, q, mol, ep).total
    # Ewald pressure by COM scaling of the site arrays
    com = zeros(3, M); off = zeros(3, 3M)
    for i in 1:M
        p = pos[:, 3i-2:3i]; for h in 2:3, d in 1:3; p[d, h] -= L * round((p[d, h] - p[d, 1]) / L); end
        mm = [15.9994, 1.00794, 1.00794]; c = p * mm ./ sum(mm); com[:, i] = c; off[:, 3i-2:3i] = p .- c
    end
    fE(s) = MCm.ewald_energy((s .* repeat(com, inner=(1, 3))) .+ off, q, mol, MCm.EwaldParams(L * s, α_ew, 10.0, n2, KE_K_ANGSTROM)).total
    hh = 1e-4; P_ew = -(fE(1 + hh) - fE(1 - hh)) / ((L*(1+hh))^3 - (L*(1-hh))^3) * ATM_PER_K_A3
    @printf("\n# config %d: M=%d L=%.0f Å  Ewald(α=%.3f,|n|²<%d): E_coul=%.1f K  P_coul=%.0f atm\n", cfg, M, L, α_ew, n2, E_ew, P_ew)
    @printf("%7s | %-26s | %-26s | %-26s\n", "α", "molecule cutoff rc=10: ΔE%  ΔP", "site cutoff rc=10: ΔE%  ΔP", "molecule cutoff rc=8: ΔE%  ΔP")
    for α in alphas
        vals = String[]
        for (cut, rc) in ((:molecule, 10.0), (:site, 10.0), (:molecule, 8.0))
            (cfg <= 3 && rc + 2.0 > L / 2 && cut == :site) && (push!(vals, @sprintf("%26s", "(rc+ext > L/2)")); continue)
            ff = coul_ff(α, rc, cut)
            E = E_wolf(sys, ff); P = P_fd(sys, ff)
            push!(vals, @sprintf("%10.3f%% %12.0f   ", 100(E - E_ew)/abs(E_ew), P - P_ew))
        end
        @printf("%7.2f | %s | %s | %s\n", α, vals...)
    end
end

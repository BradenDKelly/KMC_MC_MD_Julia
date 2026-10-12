# Generate the reaction-field / LJ energy reference configuration for test/test_rf_gromacs.jl:
# 500 SPC/E + 1 OPLS-AA rigid methanol in the eKMC box (300 K, 1000 kg/m³ water), 100 Metropolis sweeps
# from a random start, written as a 6-decimal .gro for `gmx mdrun -rerun`, plus MolSim's own energies.
# Usage: julia --project=. dev/gromacs_rf_reference.jl <outdir>; then run the gmx reruns (see gromacs_fe/rf_ref/run.sh).

using MolSim, Random, StaticArrays, Printf
const MCm = MolSim.MC
outdir = length(ARGS) >= 1 ? ARGS[1] : "/shared/kmc_polyatomic/gromacs_fe/rf_ref"
m = MCm.spce_solute_model(:methanol; coulomb=:rf, cutoff=:site, cap_overlaps=true)
ff = m.ff; σ_A = m.σ_angstrom; T = 300.0 / m.ϵ_kelvin
N = 500
ρ_A3 = 1000.0 * 1e3 / 18.0153 * 6.02214129e23 / 1e30
L = cbrt(N / (ρ_A3 * σ_A^3))
sys = MCm.RigidSystem(m.templates, vcat(fill(1, N), 2), L, 1 / T, Xoshiro(2026))
MCm.random_rigid_system!(sys, ff)
st = MCm.RigidMCStats()
δ = 0.3 / σ_A; δθ = 0.3
for s in 1:100, _ in 1:N+1
    MCm.mc_translate!(sys, ff, rand(sys.rng, 1:N+1), δ, st)
    MCm.mc_rotate!(sys, ff, rand(sys.rng, 1:N+1), δθ, st)
end
# RigidSystem path vs flat routine (site cutoff); write the configuration in nm with 6 decimals
pos, q, mol, σ, ϵ = MCm.site_arrays(sys, ff)
E, Ex, Es = MCm.rf_energy(pos, q, mol, L, ff.rf)
ffn = MCm.SiteForceField([ff.σ_mix[a, a] for a in 1:ff.n_types], [ff.ϵ_mix[a, a] for a in 1:ff.n_types]; rc=ff.rc,
                         coulomb=:none, cutoff=:site, mixing=:geometric, cap_overlaps=true)
U = MCm.total_pair_energy(sys, ff); ULJ = MCm.total_pair_energy(sys, ffn)
@printf("RigidSystem Coulomb %.10f  flat RF pair %.10f  rel diff %.2e\n", U - ULJ, E, (U - ULJ - E) / E)
@printf("constants RigidSystem %.10f  flat %.10f\n", MCm.wolf_constant_energy(sys, ff), Ex + Es)
kJ = m.ϵ_kelvin * 0.0083144626
names = (["OW", "HW1", "HW2"], ["C", "H1", "H2", "H3", "O", "HO"])
open(joinpath(outdir, "conf.gro"), "w") do io
    println(io, "500 SPC/E + OPLS-AA rigid methanol, MolSim configuration (seed 2026, 100 sweeps)")
    @printf(io, "%5d\n", sys.n_sites_total)
    k = 0
    for i in 1:sys.n_mol
        t = sys.templates[sys.tmpl[i]]
        res = sys.tmpl[i] == 1 ? "SOL" : "MOH"
        for n in 1:t.n_sites
            k += 1
            x = (sys.com[1, i] + sys.site_off[1, sys.site_start[i] + n]) * σ_A / 10
            y = (sys.com[2, i] + sys.site_off[2, sys.site_start[i] + n]) * σ_A / 10
            z = (sys.com[3, i] + sys.site_off[3, sys.site_start[i] + n]) * σ_A / 10
            @printf(io, "%5d%-5s%5s%5d%11.6f%11.6f%11.6f\n", i, res, names[sys.tmpl[i]][n], k, x, y, z)
        end
    end
    @printf(io, "%11.6f%11.6f%11.6f\n", L * σ_A / 10, L * σ_A / 10, L * σ_A / 10)
end
# MolSim energies in kJ/mol for the record (RF ε_rf = ∞, ε_rf = 78, ε_rf = 1), LJ site cutoff, tail
open(joinpath(outdir, "molsim_energies.txt"), "w") do io
    for εrf in (Inf, 78.0, 1.0)
        p = MCm.RFParams(ff.rc, 1.0; ε_rf=εrf)
        E, Ex, Es = MCm.rf_energy(pos, q, mol, L, p)
        @printf(io, "eps_rf %s  RF pair %.6f  excluded %.6f  self %.6f  total %.6f kJ/mol\n", string(εrf), E * kJ, Ex * kJ, Es * kJ, (E + Ex + Es) * kJ)
    end
    @printf(io, "LJ pair (site cutoff, no shift) %.6f  LJ tail %.6f kJ/mol\n", ULJ * kJ, MCm.lrc_energy(sys, ffn) * kJ)
end
println(read(joinpath(outdir, "molsim_energies.txt"), String))
# row timing: RF vs Wolf (molecule cutoff production model) vs Wolf site cutoff, single thread
for (label, mm) in (("RF site", MCm.spce_solute_model(:methanol; coulomb=:rf, cutoff=:site)),
                    ("Wolf molecule (production)", MCm.spce_solute_model(:methanol; coulomb=:wolf, cutoff=:molecule)),
                    ("Wolf site", MCm.spce_solute_model(:methanol; coulomb=:wolf, cutoff=:site)))
    f = mm.ff
    MCm.removal_energy(sys, f, 1)
    t = @elapsed for k in 1:200; MCm.removal_energy(sys, f, 1 + (k % N)); end
    @printf("row (N=501, 1 thread) %-28s %.1f µs\n", label, t / 200 * 1e6)
end

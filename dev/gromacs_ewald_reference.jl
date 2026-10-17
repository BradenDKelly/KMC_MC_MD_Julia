# Ewald / PME energies of the GROMACS reference configuration (test/data/rf_reference_spce_methanol.gro: 500 SPC/E + 1
# OPLS-AA rigid methanol, L = 2.463884 nm) against `gmx mdrun -rerun` with coulombtype = PME (fourierspacing 0.12 nm,
# order 4 → 24³; and 0.04 nm, order 8 → 64³) and Ewald, rc = 1.0 nm, ewald-rtol 1e-5 (gromacs_fe/ewald_ref, DESIGN §13).
# Prints MolSim's Coulomb energy (real + Fourier + self + intramolecular) by the RigidSystem/EwaldRecip path and by the
# flat SPME routine. Usage: julia --project=. dev/gromacs_ewald_reference.jl
using MolSim, Random, Printf
const MCm = MolSim.MC
gro = readlines(joinpath(@__DIR__, "..", "test", "data", "rf_reference_spce_methanol.gro"))
nat = parse(Int, strip(gro[2])); box = parse.(Float64, split(gro[3 + nat]))
pos_nm = zeros(3, nat)
for k in 1:nat
    l = gro[2 + k]; pos_nm[:, k] .= (parse(Float64, l[21:31]), parse(Float64, l[32:42]), parse(Float64, l[43:53]))
end
rc_A = 10.0
α_A = MCm.ewald_alpha_for(rc_A, 1e-5)
m = MCm.spce_solute_model(:methanol; coulomb=:ewald, α_per_angstrom=α_A, rc_angstrom=rc_A, cutoff=:site, cap_overlaps=true)
ff = m.ff; σ_A = m.σ_angstrom; kJ = m.ϵ_kelvin * 0.0083144626
L = box[1] * 10 / σ_A
sys = MCm.RigidSystem(m.templates, vcat(2, fill(1, 500)), L, m.ϵ_kelvin / 300.0, Xoshiro(1))
k = 0
for i in 1:sys.n_mol
    t = sys.templates[sys.tmpl[i]]
    MCm.molecule_from_sites!(sys, i, pos_nm[:, k+1:k+t.n_sites] .* (10 / σ_A)); global k += t.n_sites
end
R = MCm.attach_ewald!(sys, ff; rtol=1e-5)
ffn = MCm.SiteForceField([ff.σ_mix[a, a] for a in 1:ff.n_types], [ff.ϵ_mix[a, a] for a in 1:ff.n_types]; rc=ff.rc,
                         coulomb=:none, cutoff=:site, mixing=:geometric, cap_overlaps=true)
Ucoul = (MCm.total_pair_energy(sys, ff) - MCm.total_pair_energy(sys, ffn)) * kJ
@printf("alpha = %.6f /A (GROMACS beta %.6f /nm)  Nk(half) = %d\n", α_A, 10α_A, MCm.n_kvectors(R))
@printf("MolSim Ewald (RigidSystem/EwaldRecip) Coulomb total = %.4f kJ/mol\n", Ucoul)
pos, q, mol, σ, ϵ = MCm.site_arrays(sys, ff)
ep = MCm.EwaldParams(L, α_A * σ_A, ff.rc, R.n2max, 1.0)
for (K, p) in ((24, 4), (64, 8))
    e = MCm.pme_energy(pos, q, mol, ep, K, p)
    @printf("MolSim SPME K=%d p=%d: real %.4f fourier %.4f self %.4f intra %.4f total %.4f kJ/mol\n", K, p,
            e.real * kJ, e.fourier * kJ, e.self * kJ, e.intra * kJ, e.total * kJ)
end
ew = MCm.ewald_energy(pos, q, mol, ep)
@printf("MolSim flat Ewald: real %.4f fourier %.4f self %.4f intra %.4f total %.4f kJ/mol\n",
        ew.real * kJ, ew.fourier * kJ, ew.self * kJ, ew.intra * kJ, ew.total * kJ)
for l in eachline("/shared/kmc_polyatomic/gromacs_fe/ewald_ref/gromacs_energies.txt")
    f = split(l); c = parse(Float64, f[4]) + parse(Float64, f[5])
    @printf("GROMACS %-8s Coulomb SR + recip = %.4f kJ/mol (SR %s, recip %s)\n", f[1], c, f[4], f[5])
end

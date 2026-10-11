"""
Wolf / DSF parameter selection for SPC/E by the Hens & Vlugt protocol
(R. Hens, T.J.H. Vlugt, J. Chem. Eng. Data 63 (2018) 1096, Section 2): on a liquid configuration,
compare the pairwise (Wolf) electrostatic energy with the Ewald intermolecular Coulomb energy over a
grid of damping parameters α and cutoffs R_c, and pick the smallest R_c whose best α reproduces Ewald.

Uses the four NIST SPC/E reference configurations (test/data/nist_spce); the Ewald value is the sum
real + fourier + self + intra at the NIST settings (α = 5.6/L, |n|² < 27, r_c = 10 Å), which the test
suite verifies to 6 significant figures.

Run:  julia --project=. dev/wolf_parameter_scan.jl
"""

using MolSim
using Printf

include(joinpath(@__DIR__, "..", "test", "nist_spce_reader.jl"))

alphas = [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.25, 0.28, 0.30]
cutoffs = [8.0, 9.0, 10.0, 12.0, 14.0]

for cfg in 1:4
    L, M, pos, q, σ, ε, mol = read_nist_config(joinpath(NIST_DIR, "spce_sample_config_periodic$(cfg).txt"))
    ep = MolSim.MC.EwaldParams(L, 5.6 / L, 10.0, 27, KE_K_ANGSTROM)
    E_ew = MolSim.MC.ewald_energy(pos, q, mol, ep).total
    println("\n# NIST config $cfg: M = $M, L = $L Å, Ewald intermolecular Coulomb E/k_B = $(round(E_ew; digits=1)) K")
    println("# relative deviation (E_wolf - E_ewald)/|E_ewald| for :wolf (left) and :dsf (right)")
    @printf("%8s", "α [1/Å]")
    for rc in cutoffs
        rc <= L / 2 || continue
        @printf("  %10s %10s", "wolf rc=$(rc)", "dsf")
    end
    println()
    for α in alphas
        @printf("%8.3f", α)
        for rc in cutoffs
            rc <= L / 2 || continue
            for model in (:wolf, :dsf)
                wp = MolSim.MC.WolfParams(α, rc, KE_K_ANGSTROM; model=model)
                Ep, Es, Ei = MolSim.MC.wolf_energy(pos, q, mol, L, wp)
                @printf("  %10.4f", (Ep + Es + Ei - E_ew) / abs(E_ew))
            end
        end
        println()
    end
end

"""
Configurational Coulomb pressure of Ewald vs Wolf/DSF on the NIST SPC/E configurations.

P_conf = −∂E/∂V at fixed scaled COM positions and fixed orientations (molecules rigid), by central
finite differences of the energies in Electrostatics.jl (Ewald: real + Fourier + self + intra at fixed
α and k-set; Wolf/DSF: pair + self + intra). This is the thermodynamic pressure of each model (the
quantity the NPT ensembles equilibrate), so it tests the Wolf *virial* against Ewald, which the
energy scan (dev/wolf_parameter_scan.jl) does not. Units: K/Å³ → atm (×k_B/Å³).

Run:  julia --project=. dev/wolf_pressure_scan.jl
"""

using MolSim, Printf
const MCm = MolSim.MC
include(joinpath(@__DIR__, "..", "test", "nist_spce_reader.jl"))

const ATM_PER_K_A3 = 1.3806488e-23 / 1e-30 / 101325.0
const MASS = Dict("O" => 15.9994, "H" => 1.00794)

# unwrap each molecule (H relative to O), compute COM and body offsets
function molecules(pos, L)
    n = size(pos, 2); M = n ÷ 3
    com = zeros(3, M); off = zeros(3, n)
    for i in 1:M
        o = 3i - 2
        p = pos[:, o:o+2]
        for h in 2:3, d in 1:3
            p[d, h] -= L * round((p[d, h] - p[d, 1]) / L)
        end
        m = [MASS["O"], MASS["H"], MASS["H"]]
        c = p * m ./ sum(m)
        com[:, i] = c
        off[:, o:o+2] = p .- c
    end
    return com, off
end

scaled_positions(com, off, s) = (s .* repeat(com, inner=(1, 3))) .+ off   # COM scaling (O,H,H per molecule), orientation fixed

function pressure_fd(f, com, off, L; h=1e-4)
    Lp, Lm = L * (1 + h), L * (1 - h)
    Ep = f(scaled_positions(com, off, 1 + h), Lp)
    Em = f(scaled_positions(com, off, 1 - h), Lm)
    return -(Ep - Em) / (Lp^3 - Lm^3)
end

alphas = [0.10, 0.14, 0.16, 0.20, 0.25, 0.30]
println("# P_conf(Coulomb) in atm, by −dE/dV under COM scaling; Ewald α = 5.6/L fixed, |n|² < 27, r_c = 10 Å")
for cfg in 1:4
    L, M, pos, q, σ, ε, mol = read_nist_config(joinpath(NIST_DIR, "spce_sample_config_periodic$(cfg).txt"))
    com, off = molecules(pos, L)
    α_ew = 5.6 / L
    f_ew = (p, Lx) -> MCm.ewald_energy(p, q, mol, MCm.EwaldParams(Lx, α_ew, 10.0, 27, KE_K_ANGSTROM)).total
    P_ew = pressure_fd(f_ew, com, off, L) * ATM_PER_K_A3
    # LJ part for reference (truncated, no impulsive term; tail excluded)
    f_lj = (p, Lx) -> MCm.site_lj_energy(p, σ, ε, mol, Lx, 10.0)[1]
    P_lj = pressure_fd(f_lj, com, off, L) * ATM_PER_K_A3
    @printf("\n# config %d: M=%d L=%.1f Å ρ=%.3f g/cm³   P_coul(Ewald) = %9.1f atm   P_LJ(pair, trunc.) = %9.1f atm\n",
            cfg, M, L, M * 18.0153 / 6.02214129e23 / (L * 1e-8)^3, P_ew, P_lj)
    @printf("%8s %10s %10s | %10s %10s\n", "α [1/Å]", "Wolf rc=10", "ΔP", "DSF rc=10", "ΔP")
    for α in alphas
        res = Float64[]
        for model in (:wolf, :dsf)
            wp = MCm.WolfParams(α, 10.0, KE_K_ANGSTROM; model=model)
            f_w = (p, Lx) -> sum(MCm.wolf_energy(p, q, mol, Lx, wp))
            push!(res, pressure_fd(f_w, com, off, L) * ATM_PER_K_A3)
        end
        @printf("%8.2f %10.1f %10.1f | %10.1f %10.1f\n", α, res[1], res[1] - P_ew, res[2], res[2] - P_ew)
    end
end

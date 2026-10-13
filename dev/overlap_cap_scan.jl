# Overlap-cap design scan (JOURNAL entry 26, DESIGN §9). Run: julia --project=. dev/overlap_cap_scan.jl
# For each intermolecular LJ site pair (a on molecule A, b on molecule B): min over random orientations of the
# bare (uncapped, RF) molecule-pair energy with |r_a − r_b| = r, on a grid of r; report r*(U) = the largest r
# below which every sampled orientation has u ≥ U (U = 25, 50, 100 kT), and the current trigger 0.75 σ_ab.
using MolSim, Random, Printf, StaticArrays
const MCm = MolSim.MC
m = MCm.spce_solute_model(:methanol; coulomb=:rf, epsilon_rf=Inf, cutoff=:site)
m2 = MCm.spce_solute_model(:methane; coulomb=:rf, epsilon_rf=Inf, cutoff=:site)
kT = 300.0 / m.ϵ_kelvin; σA = m.σ_angstrom
function scan(tA, tB, ff, a, b; nor=100_000)
    ffu = MCm.uncapped(ff)
    sys = MCm.RigidSystem([tA, tB], [1, 2], 30.0, 1 / kT, Xoshiro(1)); rng = Xoshiro(7)
    rs = collect(0.8:0.05:3.2) ./ σA; mins = Float64[]
    for r in rs
        best = Inf
        for _ in 1:nor
            MCm.set_molecule!(sys, 1, MVector(10.0, 10.0, 10.0), MCm.uniform_random_quaternion(rng))
            MCm.set_molecule!(sys, 2, MVector(10.0, 10.0, 10.0), MCm.uniform_random_quaternion(rng))
            oa = sys.site_off[:, sys.site_start[1] + a]; ob = sys.site_off[:, sys.site_start[2] + b]
            MCm.set_molecule!(sys, 1, MVector(10.0, 10.0, 10.0) .- oa, sys.quat[:, 1])
            MCm.set_molecule!(sys, 2, MVector(10.0 + r, 10.0, 10.0) .- ob, sys.quat[:, 2])
            best = min(best, MCm.pair_energy(sys, ffu, 1, 2))
        end
        push!(mins, best / kT)
    end
    rstar(U) = (i = findfirst(<(U), mins); i === nothing ? NaN : (i == 1 ? NaN : rs[i-1] * σA))
    return rstar
end
wat = m.water; meoh = m.templates[2]; ch4 = m2.templates[2]
cases = [("OW-OW", wat, wat, m.ff, 1, 1), ("OW-C(MeOH)", wat, meoh, m.ff, 1, 1), ("OW-HC(MeOH)", wat, meoh, m.ff, 1, 2),
         ("OW-OH(MeOH)", wat, meoh, m.ff, 1, 5), ("OW-C(CH4)", wat, ch4, m2.ff, 1, 1), ("OW-HC(CH4)", wat, ch4, m2.ff, 1, 2)]
@printf("%-12s %8s %8s %8s %8s   (Å)\n", "pair", "0.75σ", "r*(25kT)", "r*(50kT)", "r*(100kT)")
for (name, tA, tB, ff, a, b) in cases
    ta = tA.site_type[a]; tb = tB.site_type[b]
    rst = scan(tA, tB, ff, a, b)
    @printf("%-12s %8.3f %8.2f %8.2f %8.2f\n", name, sqrt(ff.rcap2_mix[ta, tb]) * σA, rst(25), rst(50), rst(100))
end

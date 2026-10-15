#=
Controlled test of the coupling-ladder variants on one system (JOURNAL 77): flexible LJ dimer solute (ε_s) in the LJ-diatomic
solvent of test/test_ladder.jl at a liquid density, where hub landings hit solvent. Same start (seed) for every variant:
  plain     ladder only (relocation of the fractional molecule at every rung; validated vs Metropolis+MBAR, JOURNAL 40–41)
  hub       + λ = 0 hub (no relocation at rungs ≥ 3)
  local     + rejection-free solute L-moves (no hub)
  hublocal  + hub + L-moves (the production setup of _lmmc2e8 / _sl30)
Usage: julia --project=. dev/ladder_variants.jl variant seed events [ε_s=2] [N=64] [ρ=0.5] [T=1.5] [outdir]
Prints μ (occupancy, MBAR; reduced units), the per-rung occupancy profile, frac_rung, hub/hop arrivals and D0 residences.
=#
using MolSim, Random, Printf
const MCm = MolSim.MC
variant = ARGS[1]; seed = parse(Int, ARGS[2]); events = parse(Int, ARGS[3])
ε_s = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 2.0
N = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 64
ρ = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 0.5
T = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 1.5
outdir = length(ARGS) >= 8 ? ARGS[8] : joinpath(@__DIR__, "results", "ladder_variants")
variant in ("plain", "hub", "local", "hublocal") || error("variant must be plain, hub, local or hublocal")
mkpath(outdir)

# the dimer system of test/test_ladder.jl
sites = zeros(3, 2); sites[1, 1] = -0.3; sites[1, 2] = 0.3
solv = MCm.RigidTemplate("LJ2", sites, [1, 1], [0.0, 0.0], [1.0, 1.0])
bt = MCm.BondedTerms(2; bonds=[MCm.HarmonicBond(1, 2, 0.6, 400.0)])
dimer = MCm.RigidTemplate("flexdimer", sites, [2, 2], [0.0, 0.0], [1.0, 1.0]; bonded=bt)
ff = MCm.SiteForceField([1.0, 1.1], [1.0, ε_s]; rc=2.5, coulomb=:none)
β = 1 / T; L = cbrt(N / ρ); Nt = N + 1
res = MCm.build_reservoir([solv, dimer], 2, ff, β, Xoshiro(3); M=5000, burnin_sweeps=500, stride_sweeps=10, δ=0.05)
sys = MCm.RigidSystem([solv, dimer], vcat(fill(1, N), 2), L, β, Xoshiro(seed)); MCm.random_rigid_system!(sys, ff)
MCm.minimize_rigid!(sys, ff; nsteps=200)
st = MCm.RigidMCStats()
for _ in 1:1000, _ in 1:Nt                                       # start configuration (Metropolis only as the start)
    MCm.mc_translate!(sys, ff, rand(sys.rng, 1:Nt), 0.15, st); MCm.mc_rotate!(sys, ff, rand(sys.rng, 1:Nt), 0.4, st)
    MCm.mc_site_move!(sys, ff, Nt, 0.05, st); MCm.mc_conformer_swap!(sys, ff, Nt, res, st)
end
ν = MCm.estimate_relocation_rate(deepcopy(sys), ff; events=20_000)
t0 = time()
lad = MCm.run_rigid_ekmc!(sys, ff; events=events, sample_every=10, block_events=events ÷ 25, species_mu=false,
                          reservoirs=[nothing, res], ladder_solute=Nt, ladder_nu=ν, ladder_wl_events=events ÷ 4,
                          ladder_jump=variant in ("hub", "hublocal"), ladder_local=variant in ("local", "hublocal"),
                          mc_δ=0.15, mc_δθ=0.4, debug_check_every=events ÷ 4)
Ld = lad.ladder; sm = MCm.ladder_summary(Ld)
μ_oc, eμ_oc = MCm.ladder_mu_occupancy(Ld, T); μ_mb, eμ_mb = MCm.ladder_mu_mbar(Ld, T)
t = sm.t_rung; f = [T * ((Ld.η[k] - Ld.η[1]) - log(t[k] / t[1])) for k in eachindex(t)]    # occupancy μ-profile: f_K = μ
out = joinpath(outdir, "lv_$(variant)_eps$(ε_s)_N$(N)_rho$(ρ)_seed$(seed).txt")
open(out, "w") do io
    @printf(io, "RESULT variant=%s seed=%d eps=%.2f N=%d rho=%.3f T=%.3f events=%d wall=%.0f  mu_occ=%.4f ± %.4f  mu_mbar=%.4f ± %.4f  U=%.4f ± %.4f\n",
            variant, seed, ε_s, N, ρ, T, events, time() - t0, μ_oc, eμ_oc, μ_mb, eμ_mb, MCm.mean(lad.U), MCm.stderr(lad.U))
    println(io, "# profile f_k (reduced, f_1 = 0, f_K = μ_occ): ", join([@sprintf("%.3f", x) for x in f], " "))
    println(io, "# frac_rung: ", join([@sprintf("%.4f", x) for x in sm.frac_rung], " "))
    println(io, "# hop arrivals: ", sm.n_hop, "  hub arrivals: ", sm.n_jump, "  WL stages ", sm.wl_stages)
    println(io, "# D0 res_frac: ", join([@sprintf("%.3f", x) for x in sm.res_frac], " "))
end
print(read(out, String))

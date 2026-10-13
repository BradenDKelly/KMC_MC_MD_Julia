#=
Test for a finite-run bias of the eKMC time average ⟨U⟩ = Σ U Δt / Σ Δt (Tan et al. 2017) on single-site LJ,
optionally with one strongly bound solute (LJ, ε_solute), using the same rigid-molecule engine as SPC/E.

Motivation (JOURNAL entry 23): stock eKMC water U/N with a solute present scattered 0.2–1 kJ/mol above
MC/GROMACS with block errors 30× smaller than the seed spread. Hypothesis: heavy-tailed residence times
(Δt ~ 1/R, low R correlated with low U) make finite runs read high.

Usage: julia --project=. dev/ekmc_time_average_bias.jl ρ T N events seed eps_solute mc_sweeps outdir
  eps_solute = 0 → pure LJ; otherwise one solute (σ=1, ε=eps_solute, Lorentz–Berthelot) among N−1 solvent.
Writes <outdir>/ljbias_r<ρ>_T<T>_N<N>_es<eps>_seed<seed>.txt: MC ⟨U/N⟩ ± block error, and per eKMC block
(100 blocks) the time-weighted sums ΣUΔt, ΣΔt, plus per-species ΣR_sΔt for μ, and Kish n_eff of the event Δt.
=#
using MolSim, Random, StaticArrays, Printf, Dates
const MCm = MolSim.MC

ρ = parse(Float64, ARGS[1]); T = parse(Float64, ARGS[2]); N = parse(Int, ARGS[3])
events = parse(Int, ARGS[4]); seed = parse(Int, ARGS[5]); eps_s = parse(Float64, ARGS[6])
mc_sweeps = parse(Int, ARGS[7]); outdir = ARGS[8]
mkpath(outdir)
L = cbrt(N / ρ)
solv = MCm.RigidTemplate("LJ", zeros(3, 1), [1], [0.0], [1.0])
if eps_s > 0
    sol = MCm.RigidTemplate("S", zeros(3, 1), [2], [0.0], [1.0])
    ff = MCm.SiteForceField([1.0, 1.0], [1.0, eps_s]; rc=2.5, coulomb=:none)
    templates = [solv, sol]; tm = vcat(fill(1, N - 1), 2)
else
    ff = MCm.SiteForceField([1.0], [1.0]; rc=2.5, coulomb=:none)
    templates = [solv]; tm = fill(1, N)
end
out = joinpath(outdir, @sprintf("ljbias_r%.2f_T%.2f_N%d_es%g_seed%d.txt", ρ, T, N, eps_s, seed))

sys = MCm.RigidSystem(templates, tm, L, 1 / T, Xoshiro(seed))
MCm.random_rigid_system!(sys, ff)
st = MCm.RigidMCStats(); δ = 0.12
t0 = time()
for _ in 1:2000, _ in 1:N                      # equilibration (translation only: single sites)
    MCm.mc_translate!(sys, ff, rand(sys.rng, 1:N), δ, st)
end
t_eq = time() - t0
t0 = time()
mc = MCm.run_rigid_mc!(deepcopy(sys), ff; sweeps=mc_sweeps, δ=δ, δθ=0.0, sample_every=1, block_size=max(10, mc_sweeps ÷ 50))
t_mc = time() - t0
t0 = time()
nb = 100
ek = MCm.run_rigid_ekmc!(sys, ff; events=events, sample_every=10, block_events=events ÷ nb,
                         debug_check_every=events ÷ 4, species_mu=true, trace_molecule=N)
t_ek = time() - t0
dt = ek.trace_dt
neff = sum(dt)^2 / sum(abs2, dt)
open(out, "w") do io
    println(io, "# ljbias $(Dates.now()) rho=$ρ T=$T N=$N L=$L events=$events seed=$seed eps_solute=$eps_s rc=2.5 LRC on")
    @printf(io, "MC sweeps=%d time=%.0fs (eq %.0fs) U/N=%.5f ± %.5f P=%.4f ± %.4f acc=%.3f\n", mc_sweeps, t_mc, t_eq,
            MCm.mean(mc.U), MCm.stderr(mc.U), MCm.mean(mc.P), MCm.stderr(mc.P), mc.stats.trans_acc / mc.stats.trans_att)
    @printf(io, "EKMC events=%d time=%.0fs U/N=%.5f ± %.5f ratio=%.5f ± %.5f P=%.4f kish_neff=%.1f max_dt_share=%.3e\n",
            events, t_ek, MCm.mean(ek.U), MCm.stderr(ek.U), MCm.ratio_mean(ek.U), MCm.ratio_jackknife_stderr(ek.U; f=identity),
            MCm.ratio_mean(ek.P), neff, maximum(dt) / sum(dt))
    println(io, "# block  sumUdt  sumdt  sumRN_solvent_dt  sumRN_solute_dt  sumdt_RN")
    for k in eachindex(ek.U.block_sum_x)
        @printf(io, "B %d %.10e %.10e %.10e %.10e %.10e\n", k, ek.U.block_sum_x[k], ek.U.block_sum_t[k],
                ek.RN_species[1].block_sum_x[k], length(ek.RN_species) > 1 ? ek.RN_species[end].block_sum_x[k] : NaN,
                ek.RN_species[1].block_sum_t[k])
    end
end
println(read(out, String))

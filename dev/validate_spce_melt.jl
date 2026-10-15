#=
Pure SPC/E (reaction field, site cut-off 10 Å, 100 kT overlap cap: the model of the solute runs) at 300 K, 1 g/cm³:
does eKMC relax the water from a quenched start? (JOURNAL 65–66, DESIGN §11.2 stage 2)

Usage: julia --project=. dev/validate_spce_melt.jl start method N seed events wall_seconds outdir
  start  = quench (random packing + steepest descent, as the solute driver) | mc (quench + 1000 + 4000 Metropolis sweeps;
           the MC production U/N and P are printed as the reference)
  method = stock (relocations only) | local (relocations + rejection-free local moves of every water, SolventLocalMoves)
Env: SPCE_SOLVENT_NU (per-candidate prefactor ν in reduced rate units; default exp(β μ_ref) with μ_ref = −29.5 kJ/mol,
the published SPC/E excess chemical potential, so that in equilibrium a slot fires about as often as one molecule relocates;
ν only sets frequencies, any value is exact), SPCE_REPORT (events between progress lines, default 1e5).
Progress lines: events, kMC time, instantaneous U/N, cumulative time-weighted U/N and P, molecules with φ > 0, counts.
=#
using MolSim, Random, Printf, Dates, StaticArrays
const MCm = MolSim.MC

start = ARGS[1]; method = ARGS[2]
N = parse(Int, ARGS[3]); seed = parse(Int, ARGS[4]); events = parse(Int, ARGS[5]); wall = parse(Float64, ARGS[6])
outdir = length(ARGS) >= 7 ? ARGS[7] : joinpath(@__DIR__, "results", "spce_melt")
start in ("quench", "mc") || error("start must be quench or mc"); method in ("stock", "local") || error("method must be stock or local")
mkpath(outdir)
m = MCm.spce_solute_model(:methanol_flex; rc_angstrom=10.0, cutoff=:site, coulomb=:rf, epsilon_rf=Inf)   # water template + RF force field
ff = first(MCm.kT_overlap_cap(m.ff, m.templates, 300.0 / m.ϵ_kelvin; U_kT=100.0))
T = 300.0 / m.ϵ_kelvin; β = 1 / T
ρ_A3 = 1000.0 * 1e3 / 18.0153 * 6.02214129e23 / 1e30
L = cbrt(N / (ρ_A3 * m.σ_angstrom^3))
kJ = m.ϵ_kelvin * 1.3806488e-23 * 6.02214129e23 / 1000
atm = m.ϵ_kelvin * 1.3806488e-23 / (m.σ_angstrom * 1e-10)^3 / 101325.0
out = joinpath(outdir, "melt_$(start)_$(method)_N$(N)_seed$(seed).txt")
io = open(out, "w")
println(io, "# SPC/E RF melt test  $(Dates.now())  host=$(gethostname())  start=$start method=$method N=$N seed=$seed events=$events wall=$wall")
sys = MCm.RigidSystem(m.templates, fill(1, N), L, β, Xoshiro(seed))
MCm.random_rigid_system!(sys, ff)
mres = MCm.minimize_rigid!(sys, ff; nsteps=500)
const_E = MCm.wolf_constant_energy(sys, ff)
Uk(Upair) = ((Upair + MCm.lrc_energy(sys, ff)) + const_E) / N * kJ
@printf(io, "# minimize_rigid!: %d steps, U/N %.2f → %.2f kJ/mol, capped pairs left %d\n", mres.steps, Uk(mres.U_start), Uk(mres.U_end), mres.n_capped)
δ = 0.1 / m.σ_angstrom * 3.0; δθ = 0.3                                        # the solute driver's step sizes (0.3 Å, 0.3 rad)
if start == "mc"
    st = MCm.RigidMCStats(); t0 = time()
    for _ in 1:1000, _ in 1:N
        MCm.mc_translate!(sys, ff, rand(sys.rng, 1:N), δ, st); MCm.mc_rotate!(sys, ff, rand(sys.rng, 1:N), δθ, st)
    end
    mc = MCm.run_rigid_mc!(deepcopy(sys), ff; sweeps=4000, δ=δ, δθ=δθ, sample_every=2, block_size=80)
    @printf(io, "MC reference: 1000 + 4000 sweeps, %.0f s  U/N=%.3f ± %.3f kJ/mol  P=%.1f ± %.1f atm  acc_t=%.3f\n", time() - t0,
            (MCm.mean(mc.U) * N + const_E) / N * kJ, MCm.stderr(mc.U) * kJ, MCm.mean(mc.P) * atm, MCm.stderr(mc.P) * atm,
            mc.stats.trans_acc / mc.stats.trans_att)
end
ek0 = MCm.init_rigid_ekmc(sys, ff)
@printf(io, "# start: U/N(inst)=%.3f kJ/mol, molecules with φ > 0: %d\n", Uk(0.5 * sum(ek0.phi)), count(>(0.0), ek0.phi))
νs = parse(Float64, get(ENV, "SPCE_SOLVENT_NU", string(exp(β * (-29.5 / kJ)))))
rep = parse(Int, get(ENV, "SPCE_REPORT", "100000"))
r = MCm.init_rigid_ekmc_run(sys, ff; events=events, sample_every=10, block_events=max(1000, events ÷ 100), species_mu=false,
                            debug_check_every=0, mc_δ=δ, mc_δθ=δθ, local_solvent=(method == "local"), solvent_nu=νs)
method == "local" && @printf(io, "# solvent L-moves: %d movers × slots %s, a = 0.3 Å, θ_max = 0.3 rad, ν = ν_D = %.4e (reduced)\n",
                             length(r.slmoves.mols), r.slmoves.kinds, νs)
println(io, "#   events      t_kMC       U/N_inst   U/N_cum(time-w)   P_cum(atm)   φ>0   relocations   L-moves   wall(s)"); flush(io)
t0 = time()
while r.ev < events && time() - t0 < wall
    MCm.advance_rigid_ekmc!(r, sys, ff; until=min(events, r.ev + rep), wall_seconds=wall - (time() - t0))
    U_i = Uk(0.5 * sum(r.ek.phi))
    U_c = (MCm.ratio_mean(r.ba_U) * N + const_E) / N * kJ
    P_c = MCm.ratio_mean(r.ba_P) * atm
    @printf(io, "%10d  %12.5e  %9.3f  %9.3f  %10.1f  %4d  %10d  %10d  %7.0f\n", r.ev, r.t_total, U_i, U_c, P_c,
            count(>(0.0), r.ek.phi), r.n_reloc, r.slmoves === nothing ? 0 : sum(r.slmoves.n_move), time() - t0)
    flush(io)
end
res = MCm.rigid_ekmc_results(r)
@printf(io, "eKMC %s from %s: events=%d  U/N=%.3f ± %.3f kJ/mol  P=%.1f ± %.1f atm  mu_water(⟨R/N⟩)=%.2f ± %.2f kJ/mol\n", method, start, r.ev,
        (MCm.ratio_mean(res.U) * N + const_E) / N * kJ, MCm.ratio_jackknife_stderr(res.U) * kJ, MCm.ratio_mean(res.P) * atm,
        MCm.ratio_jackknife_stderr(res.P) * atm, T * log(MCm.ratio_mean(res.RN)) * kJ, T * MCm.ratio_jackknife_stderr(res.RN; f=log) * kJ)
println(io, "# block means U/N (kJ/mol): ", join(round.((res.U.block_means .* N .+ const_E) ./ N .* kJ; digits=2), " "))
close(io)
println(read(out, String))

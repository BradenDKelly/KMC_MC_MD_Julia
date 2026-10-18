# Diagnostic trace (state every 3e5 events) of eKMC (flexible solvent L-moves) for the ethanol liquid seed study (dev/ethanol_liquid_seeds.jl, same
# start for the same seed), f × 3e6 events, printing every 3e6 events: running ⟨U⟩, ⟨U_intra⟩, ⟨P⟩, kMC time, relocations,
# capped (fast-path) landings, L-moves. Decides between under-sampling (drift towards MC) and bias (stays off).
# Usage: julia -t 4 --project=. dev/ethanol_liquid_diag.jl <seed>
using MolSim, Random, Printf
const MCs = MolSim.MC
seed = parse(Int, ARGS[1]); f = parse(Int, ARGS[2])
function compress_to!(sys, ff, L; steps=12, nsteps=150)
    fac = (L / sys.L)^(1 / steps)
    for _ in 1:steps
        sys.L *= fac
        for i in 1:sys.n_mol; MCs.set_molecule!(sys, i, sys.com[:, i] .* fac, sys.quat[:, i]); end
        MCs.minimize_rigid!(sys, ff; nsteps=nsteps)
    end
    sys.L = L
end
N = 64; T = 298.0
m = MCs.spce_solute_model(:ethanol_flex; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=9.0)
t = m.templates[2]; ff = m.ff; β = m.ϵ_kelvin / T; kJ = m.ϵ_kelvin * 0.0083144626
ρ = 0.785 / 46.07 * 6.02214076e23 / 1e24
L = cbrt(N / ρ) / m.σ_angstrom
sys = MCs.RigidSystem([t], fill(1, N), 2L, β, Xoshiro(100 + seed)); MCs.random_rigid_system!(sys, ff)
compress_to!(sys, ff, L)
res = MCs.build_reservoir([t], 1, ff, β, Xoshiro(200 + seed); M=4000, burnin_sweeps=500, stride_sweeps=20, δ=0.02)
δ, δθ, δs, φ = 0.05, 0.15, 0.01, 0.5
MCs.run_rigid_mc!(sys, ff; sweeps=1000, δ=δ, δθ=δθ, sample_every=100, block_size=5, reservoirs=[res], δ_site=δs, torsion_φ=φ)
events = 3_000_000; chunk = 300_000
sysE = deepcopy(sys)
r = MCs.init_rigid_ekmc_run(sysE, ff; events=events, sample_every=20, block_events=120_000, burnin_events=300_000,
        reservoirs=[res], local_solvent=true, solvent_flexible=true, solvent_nu=1e-3, mc_δ=δ, mc_δθ=δθ, solvent_site_a=δs, solvent_φ=φ)
nO = 8; nH = 9                                   # ethanol.itp: site 8 = O, site 9 = HO
sp_min(sysE) = (sp = MCs.site_positions(sysE); dmin = Inf;
    for i in 1:N, j in 1:N
        i == j && continue
        d = [MCs._min_image(sp[k, sysE.site_start[j] + nO] - sp[k, sysE.site_start[i] + nH], sysE.L) for k in 1:3]
        dmin = min(dmin, sqrt(sum(abs2, d)))
    end; dmin * m.σ_angstrom)
t0 = 0.0; rl0 = 0; fe0 = 0
for c in 1:events ÷ chunk
    MCs.advance_rigid_ekmc!(r, sysE, ff; until=c * chunk)
    U = MCs.total_pair_energy(sysE, ff) / N * kJ
    Ui = MCs.total_intra_energy(sysE, ff) / N * kJ
    φ = r.ek.phi; kmax = argmax(φ)
    @printf("DIAG seed=%d ev=%.1e  U_now %.3f  Uintra_now %.3f  max φ %.2f (mol %d)  #φ>0 %d  min HO···O %.3f Å  Δreloc %d  Δcapped %d  Δt_kMC %.4e  Δt per event %.3e\n",
            seed, c * chunk, U, Ui, φ[kmax] * kJ, kmax, count(>(0.0), φ), sp_min(sysE), r.n_reloc - rl0, r.n_fast - fe0,
            r.t_total - t0, (r.t_total - t0) / chunk)
    flush(stdout)
    global t0 = r.t_total; global rl0 = r.n_reloc; global fe0 = r.n_fast
end

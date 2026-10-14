"""
Independent Metropolis reference for the coupling-ladder μ (DESIGN §10.6 V2): the same LJ solute system and the SAME
λ-Hamiltonians as the ladder, H_k = U_solv + λ_k Σ_j min(u_bare(s,j), U_cap), sampled by plain Metropolis MC at every
rung (translations only; single-site molecules), with the bare solute energy βu_s recorded; MBAR over the rungs
(Shirts & Chodera 2008; `mbar_histograms`) gives βμ_ex = f_K − f_1 with a delete-one-block jackknife error. The λ = 0
state is sampled by uniform ghost poses of the solute in the pure solvent (Widom insertion samples).

Usage: julia --project=. dev/mbar_mc_lj.jl EPS_S N SWEEPS SEED OUTDIR   (env LAMBDA = comma list, default the ladder rungs)
"""

using MolSim, Random, Printf, Dates
const MCm = MolSim.MC
ε_s = parse(Float64, ARGS[1]); N = parse(Int, ARGS[2]); sweeps = parse(Int, ARGS[3]); seed = parse(Int, ARGS[4]); outdir = ARGS[5]
mkpath(outdir)
λs = parse.(Float64, split(get(ENV, "LAMBDA", "0,0.01,0.03,0.1,0.2,0.35,0.5,0.7,0.85,1"), ","))
ρ = 0.85; T = 0.9; β = 1 / T; L = cbrt(N / ρ); Nt = N + 1; s_idx = Nt
site = zeros(3, 1)
solv = MCm.RigidTemplate("LJ", site, [1], [0.0], [1.0]); solu = MCm.RigidTemplate("S", site, [2], [0.0], [1.0])
ff = MCm.SiteForceField([1.0, 1.0], [1.0, ε_s]; rc=2.5, coulomb=:none, mixing=:geometric)
ff = first(MCm.kT_overlap_cap(ff, [solv, solu], T; U_kT=100.0))
βu_min, βu_max, dβu = -150.0, 1050.0, 0.1      # must cover multi-pair overlaps (JOURNAL entry 40)
nb = ceil(Int, (βu_max - βu_min) / dβu)
bin(x) = clamp(floor(Int, (x - βu_min) / dβu) + 1, 1, nb)

# energy of molecule i with everyone else in H_λ (pairs with the solute scaled by λ)
function row_energy(sys, ff, i, λ)
    u = 0.0
    for j in 1:sys.n_mol
        j == i && continue
        e = MCm.pair_energy(sys, ff, i, j)
        (i == s_idx || j == s_idx) && (e *= λ)
        u += e
    end
    return u
end
solute_bare(sys, ff) = sum(MCm.pair_energy(sys, ff, s_idx, j) for j in 1:sys.n_mol if j != s_idx)
function mc_sweep!(sys, ff, λ, δ, rng)
    acc = 0
    for _ in 1:sys.n_mol
        i = rand(rng, 1:sys.n_mol)
        com_old = copy(sys.com[:, i]); q = copy(sys.quat[:, i])
        u_old = row_energy(sys, ff, i, λ)
        com_new = com_old .+ δ .* (rand(rng, 3) .- 0.5)
        MCm.set_molecule!(sys, i, com_new, q)
        u_new = row_energy(sys, ff, i, λ)
        if rand(rng) < exp(-sys.β * (u_new - u_old))
            acc += 1
        else
            MCm.set_molecule!(sys, i, com_old, q)
        end
    end
    return acc / sys.n_mol
end

out = joinpath(outdir, "mbar_mc_lj_eps$(ε_s)_N$(N)_seed$(seed).txt")
K = length(λs); nblk = 20; per_blk = max(1, sweeps ÷ nblk)
H = [[zeros(nb) for _ in 1:K] for _ in 1:nblk]                   # per block, per rung: counts
nsamp = zeros(Int, nblk, K)
ghosts = 20                                                        # ghost poses per solvent configuration at λ = 0
open(out, "w") do io
    println(io, "# MBAR Metropolis reference  $(Dates.now())  ε_s=$ε_s N=$N ρ*=$ρ T*=$T sweeps/rung=$sweeps seed=$seed λ=$(λs)")
    rng = Xoshiro(seed)
    sys = MCm.RigidSystem([solv, solu], vcat(fill(1, N), 2), L, β, Xoshiro(seed)); MCm.random_rigid_system!(sys, ff); MCm.minimize_rigid!(sys, ff; nsteps=500)
    δ = 0.12
    for (k, λ) in enumerate(λs)
        t0 = time()
        for _ in 1:(sweeps ÷ 4); mc_sweep!(sys, ff, λ, δ, rng); end             # equilibrate at this rung
        acc = 0.0; umean = 0.0
        for sw in 1:sweeps
            acc += mc_sweep!(sys, ff, λ, δ, rng)
            b = (sw - 1) ÷ per_blk + 1
            if λ == 0.0
                for _ in 1:ghosts                                                 # uniform ghost poses: the λ = 0 ensemble
                    MCm.set_molecule!(sys, s_idx, [rand(rng) * L, rand(rng) * L, rand(rng) * L], sys.quat[:, s_idx])
                    x = β * solute_bare(sys, ff); H[b][k][bin(x)] += 1; nsamp[b, k] += 1; umean += x
                end
            else
                x = β * solute_bare(sys, ff); H[b][k][bin(x)] += 1; nsamp[b, k] += 1; umean += x
            end
        end
        @printf(io, "# rung %d λ=%.3f: acc=%.3f  <βu_s>=%.3f  %.0f s\n", k, λ, acc / sweeps, umean / sum(nsamp[:, k]), time() - t0); flush(io)
    end
    function solve(skip)
        Hs = [zeros(nb) for _ in 1:K]; Ne = zeros(K)
        for b in 1:nblk
            b == skip && continue
            for k in 1:K; Hs[k] .+= H[b][k]; Ne[k] += nsamp[b, k]; end
        end
        return MCm.mbar_histograms(λs, Hs, Ne, βu_min, dβu)
    end
    full = solve(0)
    θ = [solve(b) for b in 1:nblk]; θ̄ = sum(θ) / nblk
    err = sqrt((nblk - 1) / nblk * sum((x - θ̄)^2 for x in θ))
    # Widom from the λ = 0 ghosts alone, for comparison
    w = 0.0; n0 = 0
    for b in 1:nblk; for i in 1:nb; c = βu_min + (i - 0.5) * dβu; w += H[b][1][i] * exp(-c); n0 += H[b][1][i]; end; end
    @printf(io, "MBAR Metropolis reference: mu_ex = %.4f ± %.4f (reduced)   Widom(λ=0 ghosts, binned) = %.4f   samples per rung = %s\n",
            T * full, T * err, -T * log(w / n0), string(vec(sum(nsamp; dims=1))))
end
println(read(out, String))

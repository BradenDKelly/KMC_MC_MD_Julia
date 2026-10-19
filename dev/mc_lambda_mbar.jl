# MC yardstick for the dilute excess chemical potential with ANY electrostatics (JOURNAL 124–125), v2. Metropolis MC at a set
# of coupling states λ (the solute's whole interaction energy × λ, the H_k of the eKMC ladder) through the eKMC MC kernel
# (ekmc_mc_trial!: translations, rotations, and site + torsion trials of the flexible solute, ΔU_intra unscaled), sampling the
# bare solute energy u_s (frac_u, incl. the Ewald/PME image term); μ_ex from MBAR (Shirts & Chodera, J. Chem. Phys. 129 (2008)
# 124105, Eq. 11) over the windows with reduced energies βλ_k u_s: μ_ex = kT (f(λ=1) − f(λ=0)), f = −ln Z.
# Protocol as a GROMACS decoupling run: independent windows (one per array task), each equilibrated at λ = 1 from its own
# random start and then at its λ; a fine grid near λ = 0 (u_s is bounded by the kT100 overlap cap, so linear coupling is
# integrable, but the end needs resolution — v1 with the ladder's 10 rungs was biased, JOURNAL 125).
#   window:  julia -t 4 --project=. dev/mc_lambda_mbar.jl window <solute> <rf|wolf|ewald|pme> <seed> <k (1-based)> <sweeps> <outdir>
#   combine: julia --project=. dev/mc_lambda_mbar.jl combine <outdir>
using MolSim, Random, Printf, Statistics
const MCm = MolSim.MC
const LGRID = [0.0, 1e-4, 3e-4, 1e-3, 2e-3, 4e-3, 7e-3, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
               0.6, 0.7, 0.85, 1.0]

function run_window(solute, creq, seed, k, sweeps, outdir; N=500)
    coul = creq === :pme ? :ewald : creq
    α_A = coul === :ewald ? MCm.ewald_alpha_for(10.0, 1e-5) : 0.20
    m = MCm.spce_solute_model(solute; α_per_angstrom=α_A, rc_angstrom=10.0, cutoff=:site, coulomb=coul, epsilon_rf=Inf)
    ff = first(MCm.kT_overlap_cap(m.ff, m.templates, 300.0 / m.ϵ_kelvin; U_kT=100.0))
    T = 300.0 / m.ϵ_kelvin; kJ = m.ϵ_kelvin * 1.3806488e-23 * 6.02214129e23 / 1000
    L = cbrt(N / (1000.0 * 1e3 / 18.0153 * 6.02214129e23 / 1e30 * m.σ_angstrom^3))
    s = N + 1
    sys = MCm.RigidSystem(m.templates, vcat(fill(1, N), 2), L, 1 / T, Xoshiro(1000seed + k))
    MCm.random_rigid_system!(sys, ff)
    flex = MCm.isflexible(m.solute)
    if flex
        res = MCm.build_reservoir_ekmc(m.templates, 2, ff, 1 / T, Xoshiro(1000seed + k + 7919); M=200, burnin_sweeps=200, stride_sweeps=10, δ=0.03, torsion=true)
        MCm._set_body!(sys, s, MCm.draw_conformer(res, sys.rng))
        MCm.set_molecule!(sys, s, sys.com[:, s], MCm.uniform_random_quaternion(sys.rng))
    end
    MCm.minimize_rigid!(sys, ff; nsteps=500)
    creq === :ewald && MCm.attach_ewald!(sys, ff; rtol=1e-5, precision=:double)
    creq === :pme && MCm.attach_pme!(sys, ff; spacing=1.2 / m.σ_angstrom, order=4)
    water_only = MCm.RigidSystem(m.templates, fill(1, N), L, 1 / T, Xoshiro(0))
    ΔU_tail = MCm.lrc_energy(sys, ff) - MCm.lrc_energy(water_only, ff)
    E_const = MCm.coulomb_molecule_constant(m.solute, ff)
    δ = 0.3 / m.σ_angstrom; δθ = 0.3; δs = 0.03 / m.σ_angstrom; φ = 0.3
    ntor = flex ? length(MCm.rotatable_bonds(m.solute)) : 0
    ek = MCm.init_rigid_ekmc(sys, ff)
    ek.frac = s; ek.lam[s] = 1.0
    MCm.rebuild_rigid_ekmc!(ek, sys, ff)
    function sweep!()
        for _ in 1:sys.n_mol
            MCm.ekmc_mc_trial!(ek, sys, ff, rand(sys.rng, 1:sys.n_mol), δ, δθ; rotate=false)
            MCm.ekmc_mc_trial!(ek, sys, ff, rand(sys.rng, 1:sys.n_mol), δ, δθ; rotate=true)
        end
        if flex
            for _ in 1:m.solute.n_sites; MCm.ekmc_mc_trial!(ek, sys, ff, s, δ, δθ; rotate=false, internal=MCm.LM_S, δs=δs); end
            for _ in 1:2ntor; MCm.ekmc_mc_trial!(ek, sys, ff, s, δ, δθ; rotate=false, internal=MCm.LM_D, φ=φ); end
        end
        ek.R = sum(ek.m)
    end
    t0 = time()
    for _ in 1:1000; sweep!(); end                          # equilibration of the liquid at λ = 1
    λ = LGRID[k]
    MCm.set_fractional_coupling!(ek, sys.β, λ)
    for _ in 1:max(500, sweeps ÷ 4); sweep!(); end          # equilibration at λ_k
    U = Float64[]
    for n in 1:sweeps
        sweep!(); push!(U, ek.frac_u)
        n % 1000 == 0 && MCm.rebuild_rigid_ekmc!(ek, sys, ff)   # exact resync
    end
    mkpath(outdir)
    open(joinpath(outdir, @sprintf("w%02d.txt", k)), "w") do io
        @printf(io, "# solute %s coulomb %s seed %d k %d lambda %.6g beta %.17g kJ %.17g dUtail %.17g Econst %.17g sweeps %d time %.0f\n",
                solute, creq, seed, k, λ, sys.β, kJ, ΔU_tail, E_const, sweeps, time() - t0)
        for x in U; println(io, x); end
    end
    @printf("window k=%d λ=%.4g: ⟨u_s⟩ = %.3f kJ/mol, %d samples, %.0f s\n", k, λ, mean(U) * kJ, length(U), time() - t0)
end

lse(v) = (m = maximum(v); m + log(sum(x -> exp(x - m), v)))
"MBAR free energies f_i = −ln Z_i (f_1 = 0) of states with reduced energies βλ_i u (self-consistent iteration, Eq. 11)."
function mbar(U::Vector{Vector{Float64}}, λs, β)
    xs = vcat(U...); Nk = length.(U); K = length(λs); n = length(xs)
    f = zeros(K); D = zeros(n); w = zeros(K); t = zeros(n)
    for it in 1:200000
        for j in 1:n
            @inbounds for k in 1:K; w[k] = log(Nk[k]) + f[k] - β * λs[k] * xs[j]; end
            D[j] = lse(w)
        end
        fn = zeros(K)
        for i in 1:K
            @inbounds for j in 1:n; t[j] = -β * λs[i] * xs[j] - D[j]; end
            fn[i] = -lse(t)
        end
        fn .-= fn[1]
        maximum(abs.(fn .- f)) < 1e-9 && return fn
        f = fn
    end
    error("MBAR did not converge")
end

function combine(outdir)
    files = sort(filter(f -> occursin(r"^w\d+\.txt$", f), readdir(outdir)))
    hdr = Dict{String,String}(); λs = Float64[]; U = Vector{Float64}[]
    for f in files
        lines = readlines(joinpath(outdir, f))
        toks = split(lines[1])[2:end]
        h = Dict(toks[i] => toks[i+1] for i in 1:2:length(toks)-1)
        hdr = h
        push!(λs, parse(Float64, h["lambda"])); push!(U, parse.(Float64, lines[2:end]))
    end
    o = sortperm(λs); λs = λs[o]; U = U[o]
    β = parse(Float64, hdr["beta"]); kJ = parse(Float64, hdr["kJ"])
    corr = parse(Float64, hdr["dUtail"]) + parse(Float64, hdr["Econst"])
    (λs[1] == 0.0 && λs[end] == 1.0) || error("combine: need the λ = 0 and λ = 1 windows (have $(λs))")
    f = mbar(U, λs, β)
    μ = (f[end] - f[1]) / β
    nb = min(10, minimum(length.(U)))
    μb = [(fb = mbar([x[(b - 1) * length(x) ÷ nb + 1:b * length(x) ÷ nb] for x in U], λs, β); (fb[end] - fb[1]) / β) for b in 1:nb]
    @printf("# %s: %d windows, λ = %s\n", outdir, length(λs), string(λs))
    for k in 1:length(λs)-1
        @printf("#   λ %.4g → %.4g: Δf = %.4f kT\n", λs[k], λs[k+1], f[k+1] - f[k])
    end
    @printf("RESULT MC-λ MBAR %s %s seed=%s  mu_ex = %.3f ± %.3f kJ/mol (GROMACS-comparable: + E_const + tail; %d windows; error: MBAR over %d contiguous blocks)\n",
            hdr["solute"], hdr["coulomb"], hdr["seed"], (μ + corr) * kJ, std(μb) / sqrt(nb) * kJ, length(λs), nb)
end

if ARGS[1] == "window"
    run_window(Symbol(ARGS[2]), Symbol(ARGS[3]), parse(Int, ARGS[4]), parse(Int, ARGS[5]), parse(Int, ARGS[6]), ARGS[7])
elseif ARGS[1] == "combine"
    combine(ARGS[2])
else
    error("mode must be window or combine")
end

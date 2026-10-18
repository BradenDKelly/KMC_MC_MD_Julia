# Reciprocal-space Ewald inside rigid-molecule MC and eKMC (DESIGN §13, JOURNAL 96): the RigidSystem energy equals the
# NIST-validated flat-array `ewald_energy`; removal energies equal energy differences; every incremental update (MC
# commits, eKMC rows, image energies in φ, fractional coupling, solute and solvent L-move candidates) equals a rebuild.
using Test, Random
using MolSim
const MCe = MolSim.MC

include(joinpath(@__DIR__, "nist_spce_reader.jl"))

function ewald_spce_template_angstrom()
    θ = 109.47 * π / 180
    sites = zeros(3, 3)
    sites[:, 2] .= (sin(θ / 2), cos(θ / 2), 0.0)
    sites[:, 3] .= (-sin(θ / 2), cos(θ / 2), 0.0)
    MCe.RigidTemplate("SPC/E-A", sites, [1, 0, 0], [-2 * SPCE_QH, SPCE_QH, SPCE_QH], [15.9994, 1.00794, 1.00794])
end

function nist_rigid_system(cfg, rc)
    L, M, pos, q, σ, ε, mol = read_nist_config(joinpath(NIST_DIR, "spce_sample_config_periodic$(cfg).txt"))
    ff = MCe.SiteForceField([SPCE_SIGMA], [SPCE_EPS]; rc=rc, coulomb=:ewald, α=5.6 / L, ke=KE_K_ANGSTROM,
                            use_lrc=true, cap_overlaps=false)
    sys = MCe.RigidSystem([ewald_spce_template_angstrom()], fill(1, M), L, 1.0 / 300.0, Xoshiro(1))
    for i in 1:M
        lab = pos[:, 3i-2:3i]
        for h in 2:3, d in 1:3
            lab[d, h] -= L * round((lab[d, h] - lab[d, 1]) / L)
        end
        MCe.molecule_from_sites!(sys, i, lab)
    end
    MCe.attach_ewald!(sys, ff; n2max=27, precision=:double)   # the NIST wave-vector set (|n|² < 27)
    return sys, ff, (L=L, M=M, pos=pos, q=q, σ=σ, ε=ε, mol=mol)
end

function flat_energy(d, rc; drop::Int=0)
    keep = drop == 0 ? collect(1:3d.M) : [s for s in 1:3d.M if d.mol[s] != drop]
    ep = MCe.EwaldParams(d.L, 5.6 / d.L, rc, 27, KE_K_ANGSTROM)
    lj, _ = MCe.site_lj_energy(d.pos[:, keep], d.σ[keep], d.ε[keep], d.mol[keep], d.L, rc)
    return lj + MCe.ewald_energy(d.pos[:, keep], d.q[keep], d.mol[keep], ep).total
end

@testset "Ewald in RigidSystem = NIST-validated ewald_energy (total and removal energies)" begin
    rc = 8.0                                                   # < L/2 − extent: COM image = site minimum image (as the Wolf chain test)
    for cfg in (1, 2)
        sys, ff, d = nist_rigid_system(cfg, rc)
        U = MCe.total_pair_energy(sys, ff)
        Uref = flat_energy(d, rc)
        @test abs(U - Uref) / abs(Uref) < 1e-10
        for i in (1, 7, d.M)
            u = MCe.removal_energy(sys, ff, i)
            uref = Uref - flat_energy(d, rc; drop=i)
            @test abs(u - uref) < 1e-6                         # K; |U| ≈ 5e5 K, so a 1e-12 relative difference
        end
        @test isnan(MCe.pressure(sys, ff))                     # reciprocal virial not implemented: no silent pressure
        @test MCe.total_pair_energy_virial(sys, ff)[1] ≈ U rtol = 1e-12   # the MC sampling path (regression: it lacked the Fourier part)
        @info "Ewald RigidSystem vs flat, NIST config $cfg" U Uref MCe.n_kvectors(sys.recip)
    end
end

# small SPC/E box in reduced units with Ewald parameters from rtol = 1e-5 (GROMACS default ewald-rtol)
function ewald_water(N, seed; rc_angstrom=4.5, equil=50, precision=:double)
    ρ = 0.0334                                                 # molecules / Å³
    α = MCe.ewald_alpha_for(rc_angstrom, 1e-5)
    m = MCe.spce_model(; α_per_angstrom=α, rc_angstrom=rc_angstrom, coulomb=:ewald)
    L = cbrt(N / ρ) / m.σ_angstrom
    sys = MCe.RigidSystem([m.template], fill(1, N), L, m.ϵ_kelvin / 300.0, Xoshiro(seed))
    MCe.random_rigid_system!(sys, m.ff)
    MCe.attach_ewald!(sys, m.ff; rtol=1e-5, precision=precision)
    st = MCe.RigidMCStats()
    for _ in 1:equil, _ in 1:N
        MCe.mc_translate!(sys, m.ff, rand(sys.rng, 1:N), 0.1, st); MCe.mc_rotate!(sys, m.ff, rand(sys.rng, 1:N), 0.3, st)
    end
    return m, sys, st
end

committed_drift(sys) = (S = copy(sys.recip.S); e = copy(sys.recip.eimg); MCe.recip_rebuild!(sys.recip, sys);
                        max(maximum(abs.(S .- sys.recip.S)), maximum(abs.(e .- sys.recip.eimg))))

function ek_drift(ek, sys, ff)
    φ = copy(ek.phi); P = copy(ek.pair_u); fu = ek.frac_u
    MCe.rebuild_rigid_ekmc!(ek, sys, ff)
    return max(maximum(abs.(φ .- ek.phi)), maximum(abs.(P .- ek.pair_u)), abs(fu - ek.frac_u))
end

@testset "Ewald MC commits and eKMC bookkeeping: incremental = rebuild" begin
    m, sys, st = ewald_water(48, 11)
    ff = m.ff
    @test st.trans_acc > 0 && st.rot_acc > 0
    @test committed_drift(sys) < 1e-10                         # every accepted MC move committed its S_i
    ek = MCe.init_rigid_ekmc(sys, ff)
    U0 = 0.5 * sum(ek.phi) + 0.5 * MCe._image_energy_sum(ek, sys)
    @test U0 ≈ MCe.total_pair_energy(sys, ff) rtol = 1e-11      # ½Σφ + ½Σe = Σ pairs + Σ e
    acc = MCe.ChemicalPotentialAccumulator()
    for _ in 1:300
        MCe.rigid_ekmc_step!(ek, sys, ff, acc)
    end
    for _ in 1:3
        MCe.ekmc_mc_sweep!(ek, sys, ff, 0.1, 0.3)
    end
    @test ek_drift(ek, sys, ff) < 1e-9
    @test committed_drift(sys) < 1e-10
    # fractional molecule at λ = 0.4: pairs and image energy scaled, frac_u bare
    s = 5
    ek.frac = s; ek.lam[s] = 0.4
    MCe.rebuild_rigid_ekmc!(ek, sys, ff)
    @test ek.frac_u ≈ MCe.removal_energy(sys, ff, s) rtol = 1e-11
    for _ in 1:300
        k = MCe.select_by_mobility(ek, sys.rng)
        MCe.set_molecule!(sys, k, sys.L .* rand(sys.rng, 3), MCe.uniform_random_quaternion(sys.rng))
        MCe.update_rigid_ekmc_after_move!(ek, sys, ff, k)
    end
    MCe.set_fractional_coupling!(ek, sys.β, 0.7)
    MCe.set_molecule!(sys, s, sys.L .* rand(sys.rng, 3), MCe.uniform_random_quaternion(sys.rng))
    MCe.update_rigid_ekmc_after_move!(ek, sys, ff, s)
    MCe.ekmc_mc_sweep!(ek, sys, ff, 0.1, 0.3)
    @test ek_drift(ek, sys, ff) < 1e-9
    @test ek.frac_u ≈ MCe.removal_energy(sys, ff, s) rtol = 1e-10
end

@testset "Ewald: solvent and solute L-move candidates (Fourier part by ΔS products) = rebuild" begin
    m, sys, _ = ewald_water(40, 12)
    ff = m.ff
    N = sys.n_mol; s = N
    ek = MCe.init_rigid_ekmc(sys, ff)
    ek.frac = s; ek.lam[s] = 0.6
    MCe.rebuild_rigid_ekmc!(ek, sys, ff)
    S = MCe.SolventLocalMoves(sys, ff, collect(1:N-1); kinds=[MCe.LM_T, MCe.LM_R, MCe.LM_T], a=0.1, θmax=0.3, ν=1.0,
                              νD=1.0, frac=s)
    lm = MCe.LocalMoves(sys, s, 2; kinds=[MCe.LM_T, MCe.LM_R, MCe.LM_T], a=0.1, θmax=0.3, ν=1.0, νD=1.0)
    MCe.lm_rebuild!(lm, sys, ff)
    @test any(!=(0.0), S.cr_u) && any(!=(0.0), lm.cand_rec)
    for it in 1:60
        if it % 3 == 0                                         # solvent L-move
            MCe.slm_total_rate!(S, ek, sys.β)
            i = MCe.slm_fire!(S, sys, ff, ek, rand(sys.rng) * S.W)
            MCe.lm_molecule_moved!(lm, sys, ff, i)
        elseif it % 3 == 1                                     # relocation of a solvent molecule
            k = rand(sys.rng, 1:N-1)
            MCe.set_molecule!(sys, k, sys.L .* rand(sys.rng, 3), MCe.uniform_random_quaternion(sys.rng))
            MCe.update_rigid_ekmc_after_move!(ek, sys, ff, k)
            MCe.slm_moved!(S, sys, ff, k)
            MCe.lm_molecule_moved!(lm, sys, ff, k)
        else                                                   # L-move of the fractional molecule
            MCe.lm_total_rate!(lm, sys, ff, sys.β, ek.lam[s], ek.frac_u)
            MCe.lm_fire!(lm, sys, ff, ek, rand(sys.rng) * sum(lm.rates))
            MCe.slm_moved!(S, sys, ff, s)
            MCe.lm_rebuild!(lm, sys, ff)
        end
    end
    @test MCe.slm_check!(S, sys, ff) < 1e-9
    cu = copy(lm.cand_u); MCe.lm_rebuild!(lm, sys, ff)
    @test maximum(abs.(cu .- lm.cand_u)) < 1e-9
    @test ek_drift(ek, sys, ff) < 1e-9
    # the candidate energy of an L-move slot is the removal energy of s at that pose
    j = 2
    MCe._lm_save!(lm, sys); MCe._lm_place!(lm, sys, j)
    u_place = MCe.removal_energy(sys, ff, s)
    MCe._lm_restore!(lm, sys)
    @test lm.cand_u[j] ≈ u_place rtol = 1e-10
end

@testset "SPME reciprocal energy converges to the Ewald Fourier sum (Essmann 1995)" begin
    for cfg in (1, 4)
        L, M, pos, q, σ, ε, mol = read_nist_config(joinpath(NIST_DIR, "spce_sample_config_periodic$(cfg).txt"))
        α = 5.6 / L
        Eref = MCe.ewald_recip_energy(pos, q, MCe.EwaldParams(L, α, 10.0, 400, KE_K_ANGSTROM))   # converged k-sum
        err(K, p) = abs(MCe.pme_recip_energy(pos, q, MCe.PMEParams(L, α, KE_K_ANGSTROM, K, p)) - Eref) / abs(Eref)
        e1, e2, e3, e4 = err(16, 4), err(32, 6), err(48, 6), err(64, 8)
        @info "SPME vs Ewald, NIST config $cfg" e1 e2 e3 e4
        @test e1 > e2 > e3 > e4                               # spline/mesh error falls with K and p
        @test e4 < 1e-9                                       # (observed 2e-11 and 6e-11)
    end
    @test MCe.pme_grid_size(20.0 / 1.2) == 18 && MCe.pme_grid_size(25.0) == 25 && MCe.pme_grid_size(11) == 12
end

@testset "Ewald and SPME vs GROMACS: 500 SPC/E + OPLS-AA methanol (coulomb-modifier None, ewald-rtol 1e-5)" begin
    # tolerance 2e-5 relative: GROMACS mixed precision over ~10⁶ pair terms, as test_rf_gromacs.jl
    gro = readlines(joinpath(@__DIR__, "data", "rf_reference_spce_methanol.gro"))
    nat = parse(Int, strip(gro[2])); box = parse.(Float64, split(gro[3 + nat]))
    pos_nm = zeros(3, nat)
    for k in 1:nat
        l = gro[2 + k]; pos_nm[:, k] .= (parse(Float64, l[21:31]), parse(Float64, l[32:42]), parse(Float64, l[43:53]))
    end
    ref = Dict{String,Tuple{Float64,Float64}}()
    for l in eachline(joinpath(@__DIR__, "data", "ewald_reference_energies.txt"))
        startswith(l, "#") && continue
        f = split(l); ref[f[1]] = (parse(Float64, f[4]), parse(Float64, f[5]))
    end
    rc_A = 10.0; α_A = MCe.ewald_alpha_for(rc_A, 1e-5)
    @test 10α_A ≈ 3.123413 rtol = 1e-6                        # GROMACS ewald-rtol → β (grompp, 1 nm)
    m = MCe.spce_solute_model(:methanol; coulomb=:ewald, α_per_angstrom=α_A, rc_angstrom=rc_A, cutoff=:site, cap_overlaps=true)
    ff = m.ff; σ_A = m.σ_angstrom; kJ = m.ϵ_kelvin * 0.0083144626
    L = box[1] * 10 / σ_A
    sys = MCe.RigidSystem(m.templates, vcat(2, fill(1, 500)), L, m.ϵ_kelvin / 300.0, Xoshiro(1))
    k = 0
    for i in 1:sys.n_mol
        t = sys.templates[sys.tmpl[i]]
        MCe.molecule_from_sites!(sys, i, pos_nm[:, k+1:k+t.n_sites] .* (10 / σ_A)); k += t.n_sites
    end
    MCe.attach_ewald!(sys, ff; n2max=401, precision=:double)   # |n| ≤ 20: Fourier sum converged far below 1e-5
    ffn = MCe.SiteForceField([ff.σ_mix[a, a] for a in 1:ff.n_types], [ff.ϵ_mix[a, a] for a in 1:ff.n_types]; rc=ff.rc,
                             coulomb=:none, cutoff=:site, mixing=:geometric, cap_overlaps=true)
    Uc = (MCe.total_pair_energy(sys, ff) - MCe.total_pair_energy(sys, ffn)) * kJ
    @test isapprox(Uc, sum(ref["ewald"]); rtol=2e-5)
    @test isapprox(Uc, sum(ref["pmefine"]); rtol=2e-5)
    pos, q, mol, _, _ = MCe.site_arrays(sys, ff)
    ep = MCe.EwaldParams(L, α_A * σ_A, ff.rc, 401, 1.0)
    e24 = MCe.pme_energy(pos, q, mol, ep, 24, 4)               # the GROMACS default grid for this box
    @test isapprox(e24.fourier * kJ, ref["pme"][2]; rtol=2e-5)
    @test isapprox(e24.total * kJ, sum(ref["pme"]); rtol=2e-5)
    @test isapprox((e24.real + e24.self + e24.intra) * kJ, ref["pme"][1]; rtol=2e-5)
    @info "Ewald/SPME vs GROMACS (kJ/mol)" Uc sum(ref["ewald"]) e24.fourier * kJ ref["pme"][2] e24.total * kJ sum(ref["pme"])
end

# Higham, Accuracy and Stability of Numerical Algorithms (2nd ed., 2002), Lemma 3.1/§3.1: a computed sum of n products in
# floating point with unit roundoff u differs from the exact one by at most γ_n Σ|terms|, γ_n = n u/(1 − n u); the inputs
# rounded to Float32 (S, c: three factors) add three more roundings per term. The products are real dot products of
# length 2N_k over the interleaved (Re, Im) storage (Σ_k |Re a Re b| + |Im a Im b| ≤ Σ_k |a_k||b_k|): γ_{2N_k+4}.
γ32(n) = (u = eps(Float32) / 2; n * u / (1 - n * u))

@testset "Mixed precision (GROMACS convention): Float32 products within the rounding bound; Float64 accumulation" begin
    m, sys, _ = ewald_water(40, 12; precision=:double)
    ff = m.ff; N = sys.n_mol; R = sys.recip; Nk = MCe.n_kvectors(R)
    sysm = deepcopy(sys); MCe.set_recip_backend!(sysm.recip, :cpu; precision=:mixed)
    @test eltype(sysm.recip.Sd) == ComplexF32 && eltype(sysm.recip.S) == ComplexF64
    k = 7
    rd = zeros(N); rm = zeros(N); w = zeros(N)
    MCe.eval_row!(rd, w, sys, ff, k); MCe.eval_row!(rm, w, sysm, ff, k)
    Sk = R.S[:, k]
    bound = [γ32(2Nk + 4) * sum(R.c .* abs.(Sk) .* abs.(R.S[:, j])) for j in 1:N]
    @test all(abs.(rm .- rd) .<= bound)
    @test maximum(abs.(rm .- rd)) > 0.0                      # the products really ran in single precision
    # eKMC with mixed precision: φ drift after n events is within n × 2 × (per-row bound); the rebuild is exact Float64
    ek = MCe.init_rigid_ekmc(sysm, ff)
    Smax = maximum(abs.(R.S)); Bφ = γ32(2Nk + 4) * sum(R.c) * Smax^2
    n = 200
    for _ in 1:n
        kk = MCe.select_by_mobility(ek, sysm.rng)
        MCe.set_molecule!(sysm, kk, sysm.L .* rand(sysm.rng, 3), MCe.uniform_random_quaternion(sysm.rng))
        MCe.update_rigid_ekmc_after_move!(ek, sysm, ff, kk)
    end
    φ = copy(ek.phi)
    MCe.rebuild_rigid_ekmc!(ek, sysm, ff)
    @test maximum(abs.(φ .- ek.phi)) <= 2n * N * Bφ
    sysd = deepcopy(sysm); MCe.set_recip_backend!(sysd.recip, :cpu; precision=:double)
    ekd = MCe.init_rigid_ekmc(sysd, ff)
    @test maximum(abs.(ekd.phi .- ek.phi)) < 1e-9              # the mixed-precision rebuild is the Float64 one
    # solvent L-move candidates: one ΔS product in single precision within its bound
    S = MCe.SolventLocalMoves(sysm, ff, collect(1:N-1); kinds=[MCe.LM_T, MCe.LM_R], a=0.1, θmax=0.3, ν=1.0, νD=1.0)
    @test eltype(S.crSd) == ComplexF32
    kk = 3
    MCe.set_molecule!(sysm, kk, sysm.L .* rand(sysm.rng, 3), MCe.uniform_random_quaternion(sysm.rng))
    MCe.update_rigid_ekmc_after_move!(ek, sysm, ff, kk)
    cr0 = copy(S.cr_u)
    MCe.slm_moved!(S, sysm, ff, kk)
    Δm = S.cr_u .- cr0
    dS = sysm.recip.dS
    exact = [real(sum(sysm.recip.c .* conj.(S.crS[:, c]) .* dS)) for c in 1:size(S.crS, 2)]
    bnd = [γ32(2Nk + 4) * sum(sysm.recip.c .* abs.(S.crS[:, c]) .* abs.(dS)) for c in 1:size(S.crS, 2)]
    K = length(S.kinds)
    for ii in eachindex(S.mols), j in 1:K
        S.mols[ii] == kk && continue
        c = (ii - 1) * K + j
        @test abs(Δm[j, ii] - exact[c]) <= bnd[c] + 4eps(Float64) * abs(cr0[j, ii])
    end
end

# Higham (2002): complex multiplication has relative error ≤ √2 γ_2 (Lemma 3.5); z^n by repeated multiplication ≤ n √2 γ_2.
γ64(n) = (u = eps(Float64) / 2; n * u / (1 - n * u))
"Bound on |computed − exact| of one molecule's structure factor (power tables, `recip_sf!`) or of the direct cis sum:
per site (|nx|+|ny|+|nz| ≤ 3 nmax powers + 3 products) √2 γ_2, the argument rounding γ_6 |θ| (θ ≤ 2π·3 nmax·r/L) and the
sin/cos rounding, plus the site summation γ_{n_sites}."
function sf_bound(R, sys, i)
    t = sys.templates[sys.tmpl[i]]; s0 = sys.site_start[i]
    rmax = maximum(abs(sys.com[d, i] + sys.site_off[d, s0 + a]) for d in 1:3, a in 1:t.n_sites)
    θmax = 2π * 3R.nmax * rmax / R.L
    sum(abs, t.site_q) * ((3R.nmax + 3) * sqrt(2) * γ64(2) + 2γ64(6) * θmax + 4eps() + γ64(t.n_sites))
end

@testset "Ewald kernels: grouped structure factor, rigid-translation phase, real-view products, allocations" begin
    m, sys, _ = ewald_water(40, 21; precision=:double)
    ff = m.ff; R = sys.recip; Nk = MCe.n_kvectors(R); N = sys.n_mol
    # k-vector runs reproduce the wave-vector list
    nx = Int[]; ny = Int[]; nz = Int[]
    for g in eachindex(R.gnx), t in R.gstart[g]:R.gstart[g+1]-1
        push!(nx, R.gnx[g]); push!(ny, R.gny[g]); push!(nz, R.gnz0[g] + t - R.gstart[g])
    end
    @test nx == R.nx && ny == R.ny && nz == R.nz && R.gstart[end] == Nk + 1
    # grouped recip_sf! = direct Σ_a q_a cis(2π n·r_a/L)
    t = sys.templates[sys.tmpl[5]]; s0 = sys.site_start[5]
    direct = zeros(ComplexF64, Nk)
    for a in 1:t.n_sites
        r = [sys.com[d, 5] + sys.site_off[d, s0 + a] for d in 1:3]
        direct .+= t.site_q[a] .* cis.(2π .* (R.nx .* r[1] .+ R.ny .* r[2] .+ R.nz .* r[3]) ./ R.L)
    end
    sf = MCe.recip_sf!(zeros(ComplexF64, Nk), R, sys, 5)
    @test maximum(abs.(sf .- direct)) <= 2sf_bound(R, sys, 5)
    # e^{ik·d} S_i = S of the rigidly translated molecule (set_molecule! wraps the COM: e^{ik·nL} = 1)
    d = (0.37, -0.81, 1.9) .* R.L
    b1 = sf_bound(R, sys, 5)
    tr = MCe.recip_sf_translate!(zeros(ComplexF64, Nk), R, sf, d)
    MCe.set_molecule!(sys, 5, sys.com[:, 5] .+ collect(d), sys.quat[:, 5])
    sf2 = MCe.recip_sf!(zeros(ComplexF64, Nk), R, sys, 5)
    θd = 2π * 3R.nmax * maximum(abs.(d)) / R.L
    # errors of S_i (b1) and of the direct S at the new pose, plus the phase (powers of e^{ik·d}) times |S_i|
    @test maximum(abs.(tr .- sf2)) <= b1 + sf_bound(R, sys, 5) +
                                      maximum(abs.(sf)) * ((3R.nmax + 4) * sqrt(2) * γ64(2) + 2γ64(6) * θd + 4eps())
    MCe.recip_rebuild!(R, sys)
    # real view: Re[Aᴴ v] = (real A)ᵀ (real v), each within γ_{2N_k} Σ|a||v| of the exact value
    v = R.S[:, 3] .* R.c
    p1 = real.(R.S' * v); p2 = transpose(MCe._realview(R.S)) * MCe._realview(v)
    @test all(abs.(p1 .- p2) .<= [2γ64(2Nk + 2) * sum(abs.(R.S[:, j]) .* abs.(v)) for j in 1:N])
    # per-event allocations do not grow with the number of molecules or candidates (no per-candidate heap work)
    allocs = map((40, 80)) do Nm
        mm, s, _ = ewald_water(Nm, 22; precision=:mixed); f = mm.ff
        ek = MCe.init_rigid_ekmc(s, f)
        S = MCe.SolventLocalMoves(s, f, collect(1:Nm); kinds=[MCe.LM_T, MCe.LM_R], a=0.1, θmax=0.3, ν=1.0, νD=1.0)
        step!(k) = (MCe.update_rigid_ekmc_after_move!(ek, s, f, k); MCe.slm_moved!(S, s, f, k))
        for k in 1:4; step!(k); end
        (@allocated MCe.update_rigid_ekmc_after_move!(ek, s, f, 5)) + (@allocated MCe.slm_moved!(S, s, f, 5))
    end
    @test allocs[1] == allocs[2]
end

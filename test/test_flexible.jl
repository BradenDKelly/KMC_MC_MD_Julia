"""
Flexible molecules in eKMC (DESIGN §7): reduced-unit intramolecular energies against the GROMACS reference
conformations (T7.0 in model units), the ideal-gas conformer reservoir against quadrature (T7.1), and the
cross-method test of a flexible LJ dimer solute in the rigid LJ-diatomic solvent (T7.2): Metropolis MC with
single-site and conformer-exchange moves vs stock eKMC with reservoir relocations for U, P and ⟨U_intra⟩,
and the eKMC per-species μ vs Widom insertion of ideal-gas conformers into the pure solvent. Tolerances:
exact identities at round-off; statistical comparisons within 3 combined standard errors (fixed seeds).
"""

using Test
using MolSim
using StaticArrays
using Random

const MCm = MolSim.MC
within3σ(a, ea, b, eb) = abs(a - b) <= 3 * sqrt(ea^2 + eb^2)
vmean(x) = sum(x) / length(x)
vstd(x) = sqrt(sum(abs2, x .- vmean(x)) / (length(x) - 1))

@testset "Flexible methanol: reduced-unit intramolecular energy reproduces the GROMACS conformations" begin
    m = MCm.spce_solute_model(:methanol_flex; coulomb=:none, cutoff=:site)
    t = m.solute; kJ = m.ϵ_kelvin * 0.0083144626; σ = m.σ_angstrom
    @test MCm.isflexible(t) && !MCm.isflexible(m.water)
    n = 0
    for line in eachline(joinpath(@__DIR__, "data", "opls_methanol_intra_reference.txt"))
        startswith(line, "#") && continue
        v = parse.(Float64, split(line))
        X = reshape(v[1:18], 3, 6) .* (10 / σ)                 # nm → reduced length
        pot_g, qqsr_g = v[26], v[25]
        @test isapprox(MCm.flex_intra_energy(t, m.ff, X) * kJ, pot_g - qqsr_g; rtol=1e-5, atol=1e-3)
        n += 1
    end
    @test n == 7
    # with reaction field the excluded-pair correction of the conformation is added (−ke k_rf |μ|² for a neutral molecule)
    mr = MCm.spce_solute_model(:methanol_flex; coulomb=:rf, cutoff=:site)
    X0 = mr.solute.sites_body
    μvec = X0 * mr.solute.site_q
    @test isapprox(MCm.flex_intra_energy(mr.solute, mr.ff, X0) - MCm.flex_intra_energy(t, m.ff, X0),
                   -mr.ff.rf.k_rf * sum(abs2, μvec); rtol=1e-10)
    @test_throws ErrorException MCm.flex_intra_energy(mr.solute, MCm.spce_solute_model(:methanol_flex; coulomb=:wolf).ff, X0)
end

@testset "Flexible methanol reservoir: ⟨U_intra⟩ and its terms against a 0.2 fs GROMACS vacuum run" begin
    ref = Dict{String,Tuple{Float64,Float64}}()
    for l in eachline(joinpath(@__DIR__, "data", "opls_methanol_vacuum_reference.txt"))
        startswith(l, "#") && continue
        f = split(l); ref[f[1]] = (parse(Float64, f[2]), parse(Float64, f[3]))
    end
    m = MCm.spce_solute_model(:methanol_flex; coulomb=:rf, cutoff=:site)
    t = m.solute; ff = m.ff; kJ = m.ϵ_kelvin * 0.0083144626; T = 300.0 / m.ϵ_kelvin
    res = MCm.build_reservoir(m.templates, 2, ff, 1 / T, Xoshiro(7); M=20000, burnin_sweeps=2000, stride_sweeps=20, δ=0.03)
    bt = t.bonded
    eb = Float64[]; ea = Float64[]; ed = Float64[]; e14 = Float64[]
    for X in res.confs
        b, a, d = MCm.bonded_energy(bt, X); push!(eb, b); push!(ea, a); push!(ed, d)
        push!(e14, sum(bt.fudge_qq * t.site_q[i] * t.site_q[j] / sqrt(sum(abs2, X[:, i] .- X[:, j])) for (i, j) in bt.pairs14))
    end
    se(x) = vstd(x) / sqrt(length(x) / 5) * kJ              # ≤ 5 correlated draws per 20-sweep stride
    agree(x, key) = abs(vmean(x) * kJ - ref[key][1]) <= 3 * sqrt(se(x)^2 + ref[key][2]^2)
    @test agree(res.U_intra, "Potential")
    @test agree(eb, "Bond") && agree(ea, "Angle") && agree(ed, "Ryckaert-Bell.") && agree(e14, "Coulomb-14")
    @test abs(vmean(eb) * kJ - 2.5 * 300 * 0.0083144626) < 0.1       # 5 harmonic bonds: ½kT each (+ Jacobian, < 0.1 kJ/mol)
    @test 0.2 < res.acceptance < 0.5
end

@testset "Conformer reservoir: harmonic dimer against quadrature; moves and relocations balanced" begin
    sites = zeros(3, 2); sites[1, 1] = -0.3; sites[1, 2] = 0.3
    solv = MCm.RigidTemplate("LJ2", sites, [1, 1], [0.0, 0.0], [1.0, 1.0])
    bt = MCm.BondedTerms(2; bonds=[MCm.HarmonicBond(1, 2, 0.6, 400.0)])
    dimer = MCm.RigidTemplate("flexdimer", sites, [2, 2], [0.0, 0.0], [1.0, 1.0]; bonded=bt)
    ff = MCm.SiteForceField([1.0, 1.1], [1.0, 0.8]; rc=2.5, coulomb=:none)
    N = 32; T = 2.0; β = 1 / T; L = cbrt(N / 0.3); Nt = N + 1
    res = MCm.build_reservoir([solv, dimer], 2, ff, β, Xoshiro(3); M=20000, burnin_sweeps=1000, stride_sweeps=10, δ=0.05)
    # ideal gas: ⟨½k(r−b0)²⟩ with the Cartesian (r² dr) measure, by quadrature
    rs = range(0.3, 0.9; length=200001)
    w = @. exp(-β * 0.5 * 400 * (rs - 0.6)^2) * rs^2
    Uq = sum(@. 0.5 * 400 * (rs - 0.6)^2 * w) / sum(w)
    eU = vstd(res.U_intra) / sqrt(length(res.U_intra) / 5)       # ≤ 5 correlated draws per stride
    @test abs(vmean(res.U_intra) - Uq) <= 3 * eU
    @test all(size(c) == (3, 2) for c in res.confs) && all(abs(sum(c[1, :])) < 1e-12 for c in res.confs)   # COM-centred
    @test_throws ErrorException MCm.build_reservoir([solv, dimer], 1, ff, β, Xoshiro(1); M=10)
    # Widom reference: ideal-gas dimer into the PURE solvent
    sys0 = MCm.RigidSystem([solv, dimer], fill(1, N), L, β, Xoshiro(11)); MCm.random_rigid_system!(sys0, ff)
    st = MCm.RigidMCStats()
    for _ in 1:300, _ in 1:N
        MCm.mc_translate!(sys0, ff, rand(sys0.rng, 1:N), 0.15, st); MCm.mc_rotate!(sys0, ff, rand(sys0.rng, 1:N), 0.4, st)
    end
    mc0 = MCm.run_rigid_mc!(sys0, ff; sweeps=3000, δ=0.15, δθ=0.4, sample_every=1, block_size=100,
                            widom_template=2, widom_per_sample=50, reservoirs=[nothing, res])
    w0, ew0 = MCm.mean(mc0.widom), MCm.stderr(mc0.widom)
    μ_w, eμ_w = -T * log(w0), T * ew0 / w0
    @test_throws ErrorException MCm.widom_rigid(sys0, ff, 2; ninsert=10)          # flexible ghost needs a reservoir
    # solvent + 1 flexible solute: MC (site + conformer moves) vs stock eKMC (reservoir relocations)
    sys = MCm.RigidSystem([solv, dimer], vcat(fill(1, N), 2), L, β, Xoshiro(11)); MCm.random_rigid_system!(sys, ff)
    @test size(sys.body[Nt]) == (3, 2) && size(sys.body[1]) == (3, 0) && sys.ext[Nt] ≈ 0.3
    st = MCm.RigidMCStats()
    for _ in 1:300, _ in 1:Nt
        MCm.mc_translate!(sys, ff, rand(sys.rng, 1:Nt), 0.15, st); MCm.mc_rotate!(sys, ff, rand(sys.rng, 1:Nt), 0.4, st)
        MCm.mc_site_move!(sys, ff, Nt, 0.05, st); MCm.mc_conformer_swap!(sys, ff, Nt, res, st)
    end
    @test st.site_att > 0 && st.swap_att > 0 && !MCm.mc_site_move!(sys, ff, 1, 0.05, st)    # rigid molecules: no-op
    # the stored conformation is consistent with the site offsets and the extent
    X = MCm.molecule_sites(sys, Nt); c = X * dimer.site_mass ./ sum(dimer.site_mass)
    @test maximum(abs.(X .- c .- MCm.quaternion_to_rotation_matrix(sys.quat[:, Nt]) * sys.body[Nt])) < 1e-12
    @test sys.ext[Nt] ≈ maximum(sqrt.(sum(abs2, sys.body[Nt]; dims=1)))
    mc = MCm.run_rigid_mc!(deepcopy(sys), ff; sweeps=3000, δ=0.15, δθ=0.4, sample_every=1, block_size=100,
                           reservoirs=[nothing, res], δ_site=0.05)
    events = 2_000_000
    @test_throws ErrorException MCm.run_rigid_ekmc!(deepcopy(sys), ff; events=100, block_events=50)                      # needs reservoirs
    @test_throws ErrorException MCm.run_rigid_ekmc!(deepcopy(sys), ff; events=100, block_events=50, reservoirs=[nothing, res], mc_kernel_rate=1.0)
    ek = MCm.run_rigid_ekmc!(deepcopy(sys), ff; events=events, sample_every=1, block_events=events ÷ 25, species_mu=true,
                             reservoirs=[nothing, res], debug_check_every=events ÷ 4)
    r2 = ek.RN_species[2]
    μ_ek, eμ_ek = T * log(MCm.ratio_mean(r2)), T * MCm.ratio_jackknife_stderr(r2; f=log)
    @info "flexible LJ dimer at infinite dilution, N=32+1 ρ*=0.3 T*=2" μ_w eμ_w μ_ek eμ_ek MCm.mean(mc.U) MCm.mean(ek.U) MCm.mean(mc.U_intra) MCm.mean(ek.U_intra) Uq
    @test within3σ(MCm.mean(mc.U), MCm.stderr(mc.U), MCm.mean(ek.U), MCm.stderr(ek.U))
    @test within3σ(MCm.mean(mc.P), MCm.stderr(mc.P), MCm.mean(ek.P), MCm.stderr(ek.P))
    @test within3σ(MCm.mean(mc.U_intra), MCm.stderr(mc.U_intra), MCm.mean(ek.U_intra), MCm.stderr(ek.U_intra))
    @test within3σ(μ_w, eμ_w, μ_ek, eμ_ek)
end

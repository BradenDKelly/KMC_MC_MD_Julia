# kT overlap cap for polyatomic molecules (kT_overlap_cap, DESIGN §9, user decision 2026-10-13: 100 kT).
using Test
using MolSim
using StaticArrays
using Random
const MCc = MolSim.MC

_ncap(sys, ff) = count(MCc.pair_energy(sys, ff, i, j) >= minimum(ff.ucap_mix) * 0.999
                       for i in 1:sys.n_mol-1 for j in i+1:sys.n_mol)

@testset "kT overlap cap: LJ reproduces the thesis rule u(r*) = U_cap" begin
    t = MCc.RigidTemplate("LJ", zeros(3, 1), [1], [0.0], [1.0])
    ff = MCc.SiteForceField([1.0], [1.0]; rc=2.5)
    T = 0.9
    ffc, r = MCc.kT_overlap_cap(ff, [t], T; U_kT=100.0)
    x = (2 / (1 + sqrt(1 + 4 * 100T / 4))) ^ (1 / 6)          # root of 4(x⁻¹² − x⁻⁶) = 100 kT
    @test r[1, 1] ≈ x rtol = 1e-4
    @test ffc.cap_min_u == 100T && all(ffc.ucap_mix .== 100T)
    # The kT cap makes every pair core ≥ its rim, but that does NOT remove the trap of random starts: from the
    # compressed start (U/N ≈ +23 ε) two atoms sharing one site relieve several other repulsive contacts at
    # once (a many-body gain), and the pair then needs a hole to separate (JOURNAL 27). At equilibrium entering
    # a core costs ≈ 100 kT, so only non-equilibrium starts are affected — the minimiser is the cure. Test:
    # legacy and kT cap both trap some random starts without it; with minimize_rigid! neither does.
    N = 108; L = cbrt(N / 0.85); trapped = Dict(:legacy => 0, :kT => 0, :kT_min => 0)
    for seed in 1:16, (f, tag) in ((ff, :legacy), (ffc, :kT), (ffc, :kT_min))
        sys = MCc.RigidSystem([t], fill(1, N), L, 1 / T, Xoshiro(seed)); MCc.random_rigid_system!(sys, f)
        tag === :kT_min && MCc.minimize_rigid!(sys, f; nsteps=500)
        st = MCc.RigidMCStats()
        for _ in 1:300, _ in 1:N
            MCc.mc_translate!(sys, f, rand(sys.rng, 1:N), 0.12, st)
        end
        trapped[tag] += _ncap(sys, f) > 0
    end
    @info "random LJ starts with a capped pair after 300 sweeps" trapped
    @test trapped[:legacy] > 0
    @test trapped[:kT_min] == 0
end

@testset "kT overlap cap: SPC/E + methanol (RF) — min rule, no trap, fidelity" begin
    m = MCc.spce_solute_model(:methanol; coulomb=:rf, epsilon_rf=Inf, cutoff=:site)
    T = 300.0 / m.ϵ_kelvin; U = 100T
    ffc, r = MCc.kT_overlap_cap(m.ff, m.templates, T; n_orient=5_000)
    @test all(r .> 0.0) && r[1, 1] * m.σ_angstrom > 1.0 && r[1, 1] * m.σ_angstrom < 2.374   # inside the old trigger
    ffu = MCc.uncapped(ffc)
    # every molecule pair: capped = U_cap if triggered, else min(u_bare, U_cap); never above U_cap
    sys = MCc.RigidSystem(m.templates, [1, 2, 1], 2.0, 1 / T, Xoshiro(3))
    rng = Xoshiro(4); nviol = 0; ntrig = 0; nmin = 0
    for _ in 1:20_000
        for i in 1:3
            MCc.set_molecule!(sys, i, MVector(rand(rng) * 2.0, rand(rng) * 2.0, rand(rng) * 2.0), MCc.uniform_random_quaternion(rng))
        end
        for (i, j) in ((1, 2), (1, 3), (2, 3))
            uc = MCc.pair_energy(sys, ffc, i, j); ub = MCc.pair_energy(sys, ffu, i, j)
            uc <= U || (nviol += 1)
            if uc == U && ub < U
                ntrig += 1                                       # inside a trigger radius
            elseif uc != min(ub, U)
                nviol += 1
            else
                ub >= U && (nmin += 1)
            end
        end
    end
    @test nviol == 0
    @test ntrig == 0       # inside a trigger the bare energy is ≥ U_cap for every non-collapse geometry
    @test nmin > 0         # pairs outside every trigger with u_bare ≥ U_cap are capped by the min rule
    # charge collapse: water H placed 1e-6 σ from the methanol O — bare energy → −∞, the trigger gives U_cap
    sc = MCc.RigidSystem(m.templates, [1, 2], 3.0, 1 / T, Xoshiro(5))
    MCc.set_molecule!(sc, 2, MVector(1.5, 1.5, 1.5), MCc.uniform_random_quaternion(rng))
    MCc.set_molecule!(sc, 1, MVector(1.5, 1.5, 1.5), MCc.uniform_random_quaternion(rng))
    oO = sc.site_off[:, sc.site_start[2] + 5]; oH = sc.site_off[:, sc.site_start[1] + 2]
    MCc.set_molecule!(sc, 1, MVector{3}(1.5 .+ oO .- oH .+ [1e-6, 0, 0]), sc.quat[:, 1])
    @test MCc.pair_energy(sc, ffu, 1, 2) < -1e3 * U
    @test MCc.pair_energy(sc, ffc, 1, 2) == U
    # fidelity: the GROMACS reference configuration (500 SPC/E + methanol) has capped energy = bare energy
    gro = readlines(joinpath(@__DIR__, "data", "rf_reference_spce_methanol.gro"))
    nat = parse(Int, strip(gro[2])); box = parse.(Float64, split(gro[3 + nat]))
    pos = zeros(3, nat)
    for k in 1:nat
        l = gro[2 + k]; pos[:, k] .= (parse(Float64, l[21:31]), parse(Float64, l[32:42]), parse(Float64, l[43:53]))
    end
    s = MCc.RigidSystem(m.templates, vcat(2, fill(1, 500)), box[1] * 10 / m.σ_angstrom, 1 / T, Xoshiro(1))
    k = 0
    for i in 1:s.n_mol
        n = s.templates[s.tmpl[i]].n_sites
        MCc.molecule_from_sites!(s, i, pos[:, k+1:k+n] .* (10 / m.σ_angstrom)); k += n
    end
    @test MCc.total_pair_energy(s, ffc) == MCc.total_pair_energy(s, ffu)
    @test MCc.total_pair_energy(s, ffc) == MCc.total_pair_energy(s, m.ff)
    @test _ncap(s, ffc) == 0
    # virial consistent with the energy (capped pairs carry none): pair_energy_virial agrees with pair_energy
    @test all(MCc.pair_energy_virial(s, ffc, 1, j)[1] == MCc.pair_energy(s, ffc, 1, j) for j in 2:50)
end

@testset "kT overlap cap: charge-collapse coverage check" begin
    # LJ-less +charge 2.5σ away from its LJ host: an opposite charge can reach it far outside any trigger radius
    bad = MCc.RigidTemplate("bad", [0.0 2.5; 0.0 0.0; 0.0 0.0], [1, 0], [-1.0, 1.0], [1.0, 1.0])
    ff = MCc.SiteForceField([1.0], [1.0]; rc=4.0, coulomb=:rf)
    @test_throws ErrorException MCc.kT_overlap_cap(ff, [bad], 1.0; n_orient=500)
end

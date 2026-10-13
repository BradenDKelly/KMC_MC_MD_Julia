# Checkpoint/restart of rigid-molecule eKMC (RigidEKMCRun): a run stopped part-way, serialized with its
# system (which owns the RNG), deserialized and continued must reproduce the uninterrupted run bit for bit.
using Test, Random, Serialization
using MolSim
const MCm = MolSim.MC

roundtrip(x) = (io = IOBuffer(); serialize(io, x); seekstart(io); deserialize(io))

function same_result(a, b)
    for k in (:U, :P, :rho, :V, :RN, :RNV, :U_intra)
        x, y = getfield(a, k), getfield(b, k)
        (x.block_sum_x == y.block_sum_x && x.block_sum_t == y.block_sum_t && x.sum_x == y.sum_x &&
         x.sum_t == y.sum_t && x.n == y.n) || return false
    end
    all(x.block_sum_x == y.block_sum_x && x.block_sum_t == y.block_sum_t for (x, y) in zip(a.RN_species, b.RN_species)) || return false
    return a.t_total == b.t_total && a.mu.S == b.mu.S && a.mu_npt.S_over_V == b.mu_npt.S_over_V &&
           a.trace_phi == b.trace_phi && a.trace_dt == b.trace_dt && a.vol_moves == b.vol_moves &&
           a.max_dV == b.max_dV && a.relocations == b.relocations && a.mc_events == b.mc_events &&
           a.ek.phi == b.ek.phi && a.ek.R == b.ek.R
end

@testset "eKMC checkpoint/restart reproduces the uninterrupted run" begin
    sites = zeros(3, 2); sites[1, 1] = -0.3; sites[1, 2] = 0.3
    solv = MCm.RigidTemplate("LJ2", sites, [1, 1], [0.0, 0.0], [1.0, 1.0])
    bt = MCm.BondedTerms(2; bonds=[MCm.HarmonicBond(1, 2, 0.6, 400.0)])
    dimer = MCm.RigidTemplate("flexdimer", sites, [2, 2], [0.0, 0.0], [1.0, 1.0]; bonded=bt)
    ff = MCm.SiteForceField([1.0, 1.1], [1.0, 0.8]; rc=2.5, coulomb=:none)
    N = 32; T = 2.0; β = 1 / T; L = cbrt(N / 0.3)
    res = MCm.build_reservoir([solv, dimer], 2, ff, β, Xoshiro(3); M=2000, burnin_sweeps=200, stride_sweeps=5, δ=0.05)
    flex = MCm.RigidSystem([solv, dimer], vcat(fill(1, N), 2), L, β, Xoshiro(11)); MCm.random_rigid_system!(flex, ff)
    MCm.minimize_rigid!(flex, ff; nsteps=200)
    rigid = MCm.RigidSystem([solv], fill(1, N), L, β, Xoshiro(5)); MCm.random_rigid_system!(rigid, ff)
    MCm.minimize_rigid!(rigid, ff; nsteps=200)
    events = 30_000
    cases = [
        ("NVT flexible solute, species μ, trace", flex,
         (; events=events, sample_every=3, block_events=1000, species_mu=true, trace_molecule=N + 1,
            reservoirs=[nothing, res], debug_check_every=7000)),
        ("NPT volume moves with ΔV adaptation", rigid,
         (; events=events, sample_every=2, block_events=1000, Pext=1.0, max_dV=1.0, vol_every_sweeps=5,
            adapt_dV_moves=50, adapt_dV_window=10, vol_equil_moves=20)),
        ("MC-kernel hybrid", rigid,
         (; events=events, sample_every=2, block_events=1000, mc_kernel_rate=5.0, mc_δ=0.15, mc_δθ=0.4)),
    ]
    for (name, sys0, kw) in cases
        @testset "$name" begin
            sA = deepcopy(sys0)
            a = MCm.run_rigid_ekmc!(sA, ff; kw...)
            sB = deepcopy(sys0)
            r = MCm.init_rigid_ekmc_run(sB, ff; kw...)
            for stop in (1, 4097, 12_345, 20_001)          # several restarts, incl. mid-block and on a 4096 check
                MCm.advance_rigid_ekmc!(r, sB, ff; until=stop)
                @test r.ev == stop
                sB, r = roundtrip((sB, r))
            end
            MCm.advance_rigid_ekmc!(r, sB, ff)
            @test r.ev == events
            b = MCm.rigid_ekmc_results(r)
            @test same_result(a, b)
            @test sA.com == sB.com && sA.quat == sB.quat && sA.L == sB.L && sA.body == sB.body
            @test rand(sA.rng) == rand(sB.rng)             # same RNG stream afterwards
            # a finished run does nothing more
            MCm.advance_rigid_ekmc!(r, sB, ff)
            @test r.ev == events
        end
    end
    # wall-clock stop: a zero budget stops at once, the run then completes unchanged
    sA = deepcopy(rigid); a = MCm.run_rigid_ekmc!(sA, ff; cases[3][3]...)
    sB = deepcopy(rigid); r = MCm.init_rigid_ekmc_run(sB, ff; cases[3][3]...)
    MCm.advance_rigid_ekmc!(r, sB, ff; until=5000)
    MCm.advance_rigid_ekmc!(r, sB, ff; wall_seconds=-1.0)
    @test r.ev == 8192                                     # stops at the first 4096-event check
    MCm.advance_rigid_ekmc!(r, sB, ff)
    @test same_result(a, MCm.rigid_ekmc_results(r))
end

"""
Time-weighted averager (TimeWeightedBlockAverager): `mean`/`stderr` must be the global kMC time average
Σ_k X_k / Σ_k T_k (Tan et al. 2017: ⟨A⟩ = Σ A Δt / Σ Δt) and its jackknife error, not the equal-weight mean of the
block ratios (JOURNAL 2026-10-21/22: biased by 1.2 kJ/mol in relocation-dominated liquid ethanol). Exact identities.
"""

using Test
using MolSim

const MCt = MolSim.MC

@testset "time-weighted averager: whole-run ratio" begin
    b = MCt.TimeWeightedBlockAverager(2)
    for (x, dt) in ((1.0, 1.0), (1.0, 1.0), (3.0, 1e-6), (3.0, 1e-6), (2.0, 0.5), (4.0, 0.5))
        push!(b, x, dt)
    end
    SX = 1.0 * 2 + 3.0 * 2e-6 + 2.0 * 0.5 + 4.0 * 0.5
    ST = 2.0 + 2e-6 + 1.0
    @test MCt.mean(b) ≈ SX / ST rtol = 1e-14            # time average, ≈ 1.6667
    @test MCt.mean(b) == MCt.ratio_mean(b)
    @test MCt.stderr(b) == MCt.ratio_jackknife_stderr(b)
    @test MCt.block_ratio_mean(b) ≈ (1.0 + 3.0 + 3.0) / 3 rtol = 1e-14   # the old equal-weight estimator, kept for diagnostics
    # equal block times (MC samples, dt = 1): both estimators coincide, the jackknife equals the block standard error
    e = MCt.TimeWeightedBlockAverager(3)
    for x in (1.0, 2.0, 4.0, 0.5, 0.25, 3.0, 2.0, 2.0, 5.0); push!(e, x, 1.0); end
    @test MCt.mean(e) ≈ MCt.block_ratio_mean(e) rtol = 1e-14
    @test MCt.stderr(e) ≈ MCt.block_ratio_stderr(e) rtol = 1e-12
    # no completed block: the running ratio
    p = MCt.TimeWeightedBlockAverager(10); push!(p, 2.0, 1.0); push!(p, 4.0, 3.0)
    @test MCt.mean(p) ≈ 14.0 / 4.0
end

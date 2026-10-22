"""
Correlation-corrected MC error (`statistical_inefficiency`, `stderr_corrected`): pymbar estimator (Chodera et al.,
JCTC 3 (2007) 26; Chodera, JCTC 12 (2016) 1799). Reference values from an independent Python implementation of the same
estimator (scratchpad chodera.py, JOURNAL 2026-10-22).
"""

using Test
using MolSim

const MCsi = MolSim.MC

@testset "statistical inefficiency and correlation-corrected stderr" begin
    a = [1.0, 1.2, 1.1, 1.5, 1.7, 1.6, 1.4, 1.0, 0.8, 0.9, 1.1, 1.3]     # positively correlated series
    @test MCsi.statistical_inefficiency(a) ≈ 1.6096654275092936
    @test MCsi.statistical_inefficiency(repeat([1.0, -1.0], 6)) == 1.0    # anticorrelated: clamped to 1
    @test MCsi.statistical_inefficiency(fill(2.0, 10)) == 1.0             # zero variance
    ba = MCsi.BlockAverager(1)
    for x in a; push!(ba, x); end
    @test MCsi.stderr(ba) ≈ 0.0824192289027827
    @test MCsi.stderr_corrected(ba) ≈ 0.0824192289027827 * sqrt(1.6096654275092936)
    @test isnan(MCsi.stderr_corrected(MCsi.BlockAverager(5)))
end

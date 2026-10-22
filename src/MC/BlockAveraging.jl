"""
Block averaging utility for computing means and standard errors from block means.
"""

"""
    BlockAverager

Block averager that accumulates values into blocks and computes statistics from block means.
"""
mutable struct BlockAverager
    block_size::Int
    block_means::Vector{Float64}
    current_block::Vector{Float64}
end

"""
    BlockAverager(block_size::Int)

Create a new block averager with given block size.
"""
function BlockAverager(block_size::Int)
    return BlockAverager(block_size, Float64[], Float64[])
end

"""
    push!(ba::BlockAverager, x::Float64)

Add a value to the block averager.
When a block is complete, its mean is computed and stored.
"""
function Base.push!(ba::BlockAverager, x::Float64)
    push!(ba.current_block, x)
    if length(ba.current_block) >= ba.block_size
        block_mean = sum(ba.current_block) / length(ba.current_block)
        push!(ba.block_means, block_mean)
        empty!(ba.current_block)
    end
    return ba
end

"""
    mean(ba::BlockAverager)::Float64

Compute the mean over all block means.
"""
function mean(ba::BlockAverager)::Float64
    if isempty(ba.block_means)
        if isempty(ba.current_block)
            return NaN
        else
            return sum(ba.current_block) / length(ba.current_block)
        end
    end
    n = length(ba.block_means)
    return sum(ba.block_means) / n
end

"""
    stderr(ba::BlockAverager)::Float64

Compute the standard error from block means.
Returns NaN if there are fewer than 2 blocks.
"""
function stderr(ba::BlockAverager)::Float64
    if length(ba.block_means) < 2
        return NaN
    end
    m = mean(ba)
    n = length(ba.block_means)
    sum_sq_diff = 0.0
    for x in ba.block_means
        diff = x - m
        sum_sq_diff += diff * diff
    end
    variance = sum_sq_diff / (n - 1)
    return sqrt(variance / n)
end

"""
    statistical_inefficiency(x; mintime=3)::Float64

Statistical inefficiency g of a time series, g = 1 + 2 Σ_{t≥1} (1 − t/N) C_t with C_t the normalised autocorrelation,
summed until C_t ≤ 0 for t > mintime; g ≥ 1. Same estimator as pymbar.timeseries.statisticalInefficiency
(fast=false, mintime=3): Chodera, Swope, Pitera, Seok & Dill, J. Chem. Theory Comput. 3 (2007) 26; Chodera,
J. Chem. Theory Comput. 12 (2016) 1799. A series with zero variance returns 1.
"""
function statistical_inefficiency(x::AbstractVector{<:Real}; mintime::Int=3)::Float64
    N = length(x)
    N < 2 && return 1.0
    m = sum(x) / N
    d = [xi - m for xi in x]
    σ2 = sum(abs2, d) / N
    σ2 == 0 && return 1.0
    g = 1.0
    for t in 1:N-2
        C = sum(d[i] * d[i+t] for i in 1:N-t) / (N - t) / σ2
        C <= 0 && t > mintime && break
        g += 2 * C * (1 - t / N)
    end
    return max(g, 1.0)
end

"""
    stderr_corrected(ba::BlockAverager)::Float64

Standard error of the mean corrected for correlation that survives the blocking: `stderr(ba) * sqrt(g)`, with g the
statistical inefficiency of the block-mean series (`statistical_inefficiency`). Equals `stderr(ba)` when the block means
are uncorrelated (g = 1). Returns NaN if there are fewer than 2 blocks. Used for the MC side of MC↔eKMC comparisons, whose
blocks are shorter than the MC correlation time (JOURNAL 2026-10-22: SPC/E N=48 τ ≈ 400–550 sweeps vs 100-sweep blocks).
"""
stderr_corrected(ba::BlockAverager)::Float64 =
    length(ba.block_means) < 2 ? NaN : stderr(ba) * sqrt(statistical_inefficiency(ba.block_means))

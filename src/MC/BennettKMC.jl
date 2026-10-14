"""
Bennett acceptance-ratio estimator of a species' excess chemical potential from the two ensembles that every
rigid-molecule eKMC run samples (DESIGN §10.3, JOURNAL entry 36).

Jump-chain identity: at an event the joint density of (configuration x, mover i) is ∝ π(x) m_i(x) =
e^{−βH(x∖i)}, so when molecule i is the mover the other molecules are canonical WITHOUT i and the new pose is
uniform: the landing energy u′_i is a Widom test-particle sample into the (N−1)-system (ensemble 0, "solute
decoupled"), e^{−βμ_ex} = ⟨e^{−βu′}⟩_landings. The kMC-time-weighted energies u_i of the same species are the
removal side (ensemble 1, Shing & Gubbins, Mol. Phys. 46 (1982) 1109), e^{βμ_ex} = ⟨e^{βu}⟩_time. Bennett
(J. Comput. Phys. 22 (1976) 245) combines them: for any constant M,

    ⟨ f(−β(u − ΔF) − M) ⟩_1 = e^{M} ⟨ f(β(u′ − ΔF) + M) ⟩_0 ,   f(x) = 1/(1 + e^{x}),

(from Q_1⟨w⟩_1 = Q_0⟨w e^{−βΔ}⟩_0 with w = f(−β(Δ − ΔF) − M)); ΔF = μ_ex solves it. M = ln(n_0/n_1) is the
variance-optimal choice for independent samples (Bennett Eq. 12; Shirts et al., Phys. Rev. Lett. 91 (2003)
140601); here n_1 is the Kish effective sample size of the time weights. Energies are binned (βu, width
`dβu`; the Fermi function varies on the kT scale, so 0.02 kT bins are exact to O(1e-4) kT) per block, and the
Widom sum Σe^{−βu′} is kept exactly. Errors: delete-one-block jackknife (Efron 1982) over the eKMC blocks.
"""

"""
    BennettAccumulator(species, block_events; βu_min=-150.0, βu_max=150.0, dβu=0.02)

Per-block histograms of the landing energies (counts, ensemble 0) and of the time-weighted energies (kMC time,
ensemble 1) of one species; see the file docstring. Energies outside the range are clamped into the end bins.
"""
mutable struct BennettAccumulator
    species::Int
    block_events::Int
    βu_min::Float64
    dβu::Float64
    nbins::Int
    land::Vector{Vector{Int}}          # per block: landing counts per bin
    land_wsum::Vector{Float64}         # per block: Σ e^{−βu′} (exact)
    land_n::Vector{Int}                # per block: number of landings
    time::Vector{Vector{Float64}}      # per block: Σ Δt per bin (ensemble 1)
    time_t::Vector{Float64}            # per block: Σ Δt
    time_t2::Vector{Float64}           # per block: Σ Δt² (Kish effective sample size)
    time_esum::Vector{Float64}         # per block: Σ e^{βu} Δt (exact removal estimator, consistency check)
    n_ev::Int                          # events seen (block index = (n_ev − 1) ÷ block_events + 1)
end
function BennettAccumulator(species::Int, block_events::Int; βu_min::Float64=-150.0, βu_max::Float64=150.0, dβu::Float64=0.02)
    nb = ceil(Int, (βu_max - βu_min) / dβu)
    BennettAccumulator(species, block_events, βu_min, dβu, nb, Vector{Int}[], Float64[], Int[],
                       Vector{Float64}[], Float64[], Float64[], Float64[], 0)
end
_bin(b::BennettAccumulator, βu::Float64) = clamp(floor(Int, (βu - b.βu_min) / b.dβu) + 1, 1, b.nbins)
_bincenter(b::BennettAccumulator, i::Int) = b.βu_min + (i - 0.5) * b.dβu
_block(b::BennettAccumulator) = (b.n_ev - 1) ÷ b.block_events + 1      # events 1…block_events → block 1
function _ensure_block!(b::BennettAccumulator, k::Int)
    while length(b.land) < k
        push!(b.land, zeros(Int, b.nbins)); push!(b.land_wsum, 0.0); push!(b.land_n, 0)
        push!(b.time, zeros(Float64, b.nbins)); push!(b.time_t, 0.0); push!(b.time_t2, 0.0); push!(b.time_esum, 0.0)
    end
end
"Advance the event counter (call once per event, before the pushes of that event)."
function bennett_event!(b::BennettAccumulator)
    b.n_ev += 1
    _ensure_block!(b, _block(b))
end
"Landing of a molecule of the species with energy βu′ (ensemble 0)."
function push_landing!(b::BennettAccumulator, βu′::Float64)
    k = _block(b)
    @inbounds b.land[k][_bin(b, βu′)] += 1
    b.land_wsum[k] += exp(-βu′); b.land_n[k] += 1
    return b
end
"Residence time Δt of a configuration in which a molecule of the species has energy βu (ensemble 1)."
function push_time!(b::BennettAccumulator, βu::Float64, dt::Float64)
    k = _block(b)
    @inbounds b.time[k][_bin(b, βu)] += dt
    b.time_t[k] += dt; b.time_t2[k] += dt * dt; b.time_esum[k] += exp(βu) * dt
    return b
end
nblocks(b::BennettAccumulator) = length(b.land)

_fermi(x::Float64) = x > 40.0 ? exp(-x) : (x < -40.0 ? 1.0 : 1.0 / (1.0 + exp(x)))

# Bennett equation for the summed histograms; returns βΔF (reduced). `skip` deletes one block (jackknife).
function _bennett_solve(b::BennettAccumulator, skip::Int; M::Union{Nothing,Float64}=nothing)
    nb = b.nbins
    L = zeros(Float64, nb); Tm = zeros(Float64, nb)
    n0 = 0; t1 = 0.0; t2 = 0.0
    for k in 1:nblocks(b)
        k == skip && continue
        L .+= b.land[k]; Tm .+= b.time[k]
        n0 += b.land_n[k]; t1 += b.time_t[k]; t2 += b.time_t2[k]
    end
    (n0 == 0 || t1 == 0.0) && return NaN
    n1 = t1^2 / t2                                   # Kish effective sample size of the time weights
    Mv = M === nothing ? log(n0 / n1) : M
    # g(βΔF) = e^M ⟨f(βu′ − βΔF + M)⟩_0 − ⟨f(−(βu − βΔF) − M)⟩_1, increasing in βΔF
    function g(x)
        s0 = 0.0; s1 = 0.0
        @inbounds for i in 1:nb
            c = _bincenter(b, i)
            L[i] > 0 && (s0 += L[i] * _fermi(c - x + Mv))
            Tm[i] > 0.0 && (s1 += Tm[i] * _fermi(-(c - x) - Mv))
        end
        return exp(Mv) * s0 / n0 - s1 / t1
    end
    lo, hi = b.βu_min - 50.0, b.βu_min + b.nbins * b.dβu + 50.0
    glo, ghi = g(lo), g(hi)
    (glo > 0 || ghi < 0) && return NaN
    for _ in 1:200                                   # bisection to 1e-10 kT
        mid = 0.5 * (lo + hi)
        if g(mid) < 0; lo = mid else hi = mid end
        hi - lo < 1e-10 && break
    end
    return 0.5 * (lo + hi)
end

"""
    bennett_mu(b, T; M=nothing) -> (μ_ex, err)

Excess chemical potential (reduced units, kT = T) of the species from the Bennett acceptance ratio over the
landing (insertion) and time-weighted (removal) energy histograms, with the delete-one-block jackknife error.
`M` overrides the optimal ln(n_0/n_1) (the estimator is consistent for any M).
"""
function bennett_mu(b::BennettAccumulator, T::Float64; M::Union{Nothing,Float64}=nothing)
    n = nblocks(b)
    full = T * _bennett_solve(b, 0; M=M)
    n < 2 && return (full, NaN)
    θ = [T * _bennett_solve(b, k; M=M) for k in 1:n]
    any(isnan, θ) && return (full, NaN)
    θ̄ = sum(θ) / n
    return (full, sqrt((n - 1) / n * sum((x - θ̄)^2 for x in θ)))
end

"""
    landing_widom_mu(b, T) -> (μ_ex, err)

Widom estimate from the landings alone, −kT ln⟨e^{−βu′}⟩ (exact sums, not binned), with the jackknife error.
"""
function landing_widom_mu(b::BennettAccumulator, T::Float64)
    n = nblocks(b)
    W = sum(b.land_wsum); N0 = sum(b.land_n)
    (N0 == 0 || W == 0.0) && return (NaN, NaN)
    full = -T * log(W / N0)
    n < 2 && return (full, NaN)
    θ = [-T * log((W - b.land_wsum[k]) / (N0 - b.land_n[k])) for k in 1:n]
    any(!isfinite, θ) && return (full, NaN)
    θ̄ = sum(θ) / n
    return (full, sqrt((n - 1) / n * sum((x - θ̄)^2 for x in θ)))
end

"""
    removal_mu(b, T) -> (μ_ex, err)

Time-weighted removal estimate kT ln⟨e^{βu}⟩_time from the same accumulator (exact sums), for consistency with
`RN_species` of the run.
"""
function removal_mu(b::BennettAccumulator, T::Float64)
    n = nblocks(b)
    S = sum(b.time_esum); Tt = sum(b.time_t)
    Tt == 0.0 && return (NaN, NaN)
    full = T * log(S / Tt)
    n < 2 && return (full, NaN)
    θ = [T * log((S - b.time_esum[k]) / (Tt - b.time_t[k])) for k in 1:n]
    any(!isfinite, θ) && return (full, NaN)
    θ̄ = sum(θ) / n
    return (full, sqrt((n - 1) / n * sum((x - θ̄)^2 for x in θ)))
end

"Number of landings, Kish effective sample size of the time side, and the overlap measure Σ_i min(p_0(i), p_1(i))."
function bennett_diagnostics(b::BennettAccumulator)
    n0 = sum(b.land_n); t1 = sum(b.time_t); t2 = sum(b.time_t2)
    L = zeros(Float64, b.nbins); Tm = zeros(Float64, b.nbins)
    for k in 1:nblocks(b)
        L .+= b.land[k]; Tm .+= b.time[k]
    end
    overlap = (n0 > 0 && t1 > 0) ? sum(min(L[i] / n0, Tm[i] / t1) for i in 1:b.nbins) : NaN
    return (n_landings=n0, n_eff_time=t2 > 0 ? t1^2 / t2 : NaN, overlap=overlap)
end

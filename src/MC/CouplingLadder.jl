"""
Coupling-ladder eKMC (DESIGN §10.4, JOURNAL entry 36): one "fractional" molecule s whose interactions with the
rest are scaled by λ_k, k = 1…K (λ_1 = 0, λ_K = 1), with extended state (x, k) and target density
π(x, k) ∝ exp(−β[U_rest(x) + λ_k u_s(x)] + η_k). Besides the relocations of the lemma of DESIGN §10.2 (rates
exp(β[H_k(x) − H_k(x∖i)])), rung hops are rejection-free events:

    k → k±1 (both λ > 0):  w = ν exp(½[−β(λ_j − λ_k) u_s + η_j − η_k])      (detailed balance by construction)
    1 → 2 (recoupling):    w = ν exp(½(η_2 − η_1)); the pose (and conformer) of s is drawn uniformly, so the
                           landing energy is a sample of the λ = 0 ensemble (jump-chain identity)
    2 → 1 (decoupling):    w = ν exp(½(η_1 − η_2)) exp(βλ_2 u_s)

Optional λ = 0 hub (`jump=true`, JOURNAL entry 56): direct jumps between rung 1 and every rung m ≥ 3,

    m → 1:  w = a exp(½(η_1 − η_m)) exp(βλ_m u_s)        1 → m:  w = a exp(½(η_m − η_1)), pose (and conformer) uniform

(the 1 ↔ 2 hops above are the m = 2 case; `jump_max` < K limits the hub to rungs 3…jump_max, both directions), and NO same-rung relocation of the fractional molecule at rungs ≥ 3
(c_m = 0). A uniform relocation of a large solute at an intermediate rung lands on solvent ~10⁵ times before it lands
well (ethanol in SPC/E), and every such bounce is an event; here the molecule leaves any rung for λ = 0 in one event,
and a bad landing at rung m leaves again at once (its jump rate carries exp(βλ_m u_s)), so a failed recoupling costs
two events. Detailed balance: e^{−βλ_m u_s + η_m}·w(m→1) = e^{η_1}·w(1→m) = a e^{½(η_1+η_m)}; removing a move type
keeps the rest exact, ergodicity comes through the hub (and the MC kernel's local moves if on).
At rung 1 the molecule is absent (m_s = 0, pose integrated out: weight V e^{−βU_rest + η_1}). Stock eKMC is the
ladder {0, 1} with zero sojourn at λ = 0 (DESIGN §10.4). u_s is the BARE (unscaled, capped) energy of s with the rest.
The fractional-molecule idea is CFCMC (Shi & Maginn, J. Chem. Theory Comput. 3 (2007) 1451); the weights η are
learned by Wang–Landau (Phys. Rev. Lett. 86 (2001) 2050) on the rung visits and then frozen, so the production
run is a fixed Markov process with exact time weights. Optional `wl_1t=true`: the 1/t Wang–Landau of Belardinelli &
Pereyra (Phys. Rev. E 75 (2007) 046701; J. Chem. Phys. 127 (2007) 184105) instead: f halves on flatness until
f ≤ 1/t, then f = 1/t for the rest of the run (no flatness check, no freeze), with t = ν·T_WL/K the number of
visit-equivalents per rung (T_WL = kMC time since the start of Wang–Landau). Production starts at the switch
to 1/t (forced when the `wl_events` budget is spent first) while η keeps adapting; MBAR is
η-independent, and the occupancy estimator integrates the η in force at every event (below).

Estimators of βμ_ex = −ln(Z_K / Z_1) (Z_1 = V Z_rest):
  occupancy  βμ_ex = (η_K − η_1) − ln(t_K / t_1)             (t_k = kMC time at rung k, production)
             with wl_1t: e^{−βμ_ex} = ∫ 1_K e^{−η_K(t)} dt / ∫ 1_1 e^{−η_1(t)} dt (η in force at each event; exact
             for constant η, and as η converges under the 1/t schedule)
  MBAR       over the rung histograms of βu_s (Shirts & Chodera, J. Chem. Phys. 129 (2008) 124105), reduced
             potentials u_k = βλ_k u_s; rung-1 samples are the recoupling landings (counts), rungs ≥ 2 the
             time-weighted histograms with Kish effective sample sizes.
Both with delete-one-block jackknife errors. The βu_s histograms must cover the overlap energies (several capped pairs of
100 kT each: up to ~1000 kT), because MBAR weights a λ = 0.01 state by e^{−0.01 βu}: clamping a 400 kT overlap into a
150 kT top bin biases F(0.01) − F(0) and hence μ (JOURNAL entry 40); default range −150…1050 kT in 0.1 kT bins.
"""
# Default rungs: geometric near λ = 0 so that recoupling lands softly (at λ = 0.01 a capped overlap of 100 kT costs 1 kT
# and the soft molecule relocates itself into a cavity) instead of bouncing 0 ↔ λ_2 (JOURNAL entry 38); evenly spaced above.
const LADDER_LAMBDA_DEFAULT = [0.0, 0.01, 0.03, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]

mutable struct CouplingLadder
    solute::Int
    λ::Vector{Float64}
    η::Vector{Float64}
    k::Int                         # current rung (1 = absent, K = fully coupled)
    ν::Float64
    # Wang–Landau on the rung visits (events); wl_f = 0 ⇒ frozen
    wl_f::Float64
    wl_fmin::Float64
    wl_flat::Float64
    wl_hist::Vector{Float64}       # kMC time per rung in the current WL stage
    wl_events_left::Int
    wl_stages::Int
    ev_frozen::Int                 # event at which η was frozen (0 = still adapting / never adapted)
    # production accumulators (blocks of block_events production events)
    block_events::Int
    n_ev::Int
    βu_min::Float64
    dβu::Float64
    nbins::Int
    t_rung::Vector{Vector{Float64}}            # per block: Σ Δt per rung
    t2_rung::Vector{Vector{Float64}}           # per block: Σ Δt² per rung (Kish)
    hist::Vector{Vector{Vector{Float64}}}      # per block, per rung: Σ Δt (rungs ≥ 2) or landing counts (rung 1) per βu_s bin
    n_hop::Vector{Int}                         # production hops per rung (arrivals)
    n_recouple::Int
    n_decouple::Int
    csol::Vector{Float64}                      # per-rung rate scale of the fractional molecule (exact: constant at fixed k)
    # production-time re-flattening (JOURNAL entry 45): at every block boundary η_k ← η_k − damp·ln(t_k/t̄) from the previous
    # block; the η in force during each block is stored, and the occupancy estimator combines blocks with their own η
    adapt_blocks::Bool
    adapt_damp::Float64
    η_blocks::Vector{Vector{Float64}}
    # λ = 0 hub (JOURNAL entry 56): direct jumps 1 ↔ m (m ≥ 3) with attempt frequency jump_ν; c_m = 0 for m ≥ 3
    jump::Bool
    jump_ν::Float64
    n_jump::Vector{Int}                        # production hub jumps per arrival rung
    jump_max::Int                              # hub jumps connect rung 1 with rungs 3…jump_max only (default K)
    # residence diagnostics (DESIGN §11.1 D0, production): longest uninterrupted stay per rung, and longest stay per rung
    # without any move of the fractional molecule (pose residence; closed by ladder_pose_moved! and by rung changes)
    res_k::Int
    res_t::Float64
    res_max::Vector{Float64}
    pres_t::Float64
    pres_max::Vector{Float64}
    # 1/t Wang–Landau (Belardinelli & Pereyra 2007): adaptation continues through production
    wl_1t::Bool
    wl_T::Float64                              # kMC time since the start of Wang–Landau
    wl_phase1t::Bool                           # f = K/(ν·wl_T) from here on
    occ_w::Vector{Vector{Float64}}             # per block, per rung: Σ Δt e^{−η_k} with the η in force (wl_1t occupancy)
end
function CouplingLadder(solute::Int, λ::Vector{Float64}, block_events::Int; η::Union{Nothing,Vector{Float64}}=nothing,
                        ν::Float64, wl_events::Int=0, wl_f0::Float64=1.0, wl_fmin::Float64=1e-3, wl_flat::Float64=0.8,
                        βu_min::Float64=-150.0, βu_max::Float64=1050.0, dβu::Float64=0.1,
                        csol::Union{Nothing,Vector{Float64}}=nothing, adapt_blocks::Bool=false, adapt_damp::Float64=1.0,
                        jump::Bool=false, jump_ν::Float64=ν, jump_max::Int=length(λ), wl_1t::Bool=false)
    K = length(λ)
    K >= 2 || error("CouplingLadder: at least two rungs")
    (λ[1] == 0.0 && λ[end] == 1.0 && issorted(λ) && allunique(λ)) || error("CouplingLadder: λ must increase from 0 to 1")
    ν > 0 || error("CouplingLadder: ν must be positive")
    ηv = η === nothing ? zeros(K) : copy(η)
    length(ηv) == K || error("CouplingLadder: η needs one weight per rung")
    nb = ceil(Int, (βu_max - βu_min) / dβu)
    3 <= jump_max <= K || error("CouplingLadder: jump_max must be in 3…K")
    (wl_1t && adapt_blocks) && error("CouplingLadder: wl_1t and adapt_blocks are alternative adaptation schemes")
    CouplingLadder(solute, copy(λ), ηv, K, ν, wl_events > 0 ? wl_f0 : 0.0, wl_fmin, wl_flat, zeros(K), wl_events,
                   0, wl_events > 0 ? 0 : -1, block_events, 0, βu_min, dβu, nb,
                   Vector{Float64}[], Vector{Float64}[], Vector{Vector{Float64}}[], zeros(Int, K), 0, 0,
                   _hub_csol(csol === nothing ? ones(K) : copy(csol), jump), adapt_blocks, adapt_damp, Vector{Float64}[],
                   jump, jump_ν, zeros(Int, K), jump_max, 0, 0.0, zeros(K), 0.0, zeros(K),
                   wl_1t && wl_events > 0, 0.0, false, Vector{Float64}[])
end
# with the λ = 0 hub the fractional molecule is not relocated at rungs ≥ 3 (it moves through rung 1)
function _hub_csol(c::Vector{Float64}, jump::Bool)
    jump && (c[3:end] .= 0.0)
    return c
end

"""
    solute_rate_scale(λ, ν, βμ_guess; frac=0.1) -> c_k per rung

A nearly decoupled molecule relocates at rate e^{βλu_s} ≈ 1 per unit kMC time — far more often than the whole
solvent (⟨R⟩ ≈ ν) — so with 10 rungs the soft rungs take almost all the events (dimer test: 0.25 % of the events at the
coupled rung for 11 % of the time) and each is a row evaluation. A constant factor c_k on the molecule's relocation
rate at fixed rung k is exact (the relocation flux c_k e^{−βH_k(x∖s)+η_k}/V is symmetric; hop rates do not involve
c), so c_k = min(1, frac·ν·e^{−λ_k βμ_guess}) brings its rate at the soft rungs down to ≈ frac·ν and leaves the
coupled rung unscaled (equal-rate scaling at full coupling makes bursts never terminate, JOURNAL entry 20).
"""
solute_rate_scale(λ::Vector{Float64}, ν::Float64, βμ_guess::Float64; frac::Float64=0.1) =
    [k == length(λ) ? 1.0 : min(1.0, frac * ν * exp(-λk * βμ_guess)) for (k, λk) in enumerate(λ)]
nrungs(L::CouplingLadder) = length(L.λ)
coupled(L::CouplingLadder) = L.k == nrungs(L)
adapting(L::CouplingLadder) = L.wl_f > 0.0
"Production accumulates: η frozen, or (1/t Wang–Landau) on the 1/t schedule while η keeps adapting."
ladder_production(L::CouplingLadder) = !adapting(L) || (L.wl_1t && L.wl_phase1t)
_lbin(L::CouplingLadder, βu::Float64) = clamp(floor(Int, (βu - L.βu_min) / L.dβu) + 1, 1, L.nbins)
_lblock(L::CouplingLadder) = (L.n_ev - 1) ÷ L.block_events + 1
function _ensure_lblock!(L::CouplingLadder, b::Int)
    K = nrungs(L)
    while length(L.t_rung) < b
        if L.adapt_blocks && !isempty(L.t_rung)
            # re-flatten from the block just completed: η_k ← η_k − damp·ln(t_k/t̄), unvisited rungs get +damp·ln K, step clamped to ±3
            t = L.t_rung[end]; tbar = sum(t) / K
            if tbar > 0
                for k in 1:K
                    δ = t[k] > 0 ? -L.adapt_damp * log(t[k] / tbar) : L.adapt_damp * log(K)
                    L.η[k] += clamp(δ, -3.0, 3.0)
                end
                L.η .-= L.η[end]
            end
        end
        push!(L.t_rung, zeros(K)); push!(L.t2_rung, zeros(K)); push!(L.hist, [zeros(L.nbins) for _ in 1:K]); push!(L.occ_w, zeros(K))
        push!(L.η_blocks, copy(L.η))                       # η in force during this block
    end
end
nblocks(L::CouplingLadder) = length(L.t_rung)

"Hop rates (up, down) from the current rung for bare solute energy u_s (reduced) at inverse temperature β."
function hop_rates(L::CouplingLadder, β::Float64, u_s::Float64)
    k = L.k; K = nrungs(L)
    up = 0.0; down = 0.0
    if k < K
        up = k == 1 ? L.ν * exp(0.5 * (L.η[2] - L.η[1])) :
                      L.ν * exp(min(0.5 * (-β * (L.λ[k+1] - L.λ[k]) * u_s + L.η[k+1] - L.η[k]), MOBILITY_LOG_MAX))
    end
    if k > 1
        down = k == 2 ? L.ν * exp(min(0.5 * (L.η[1] - L.η[2]) + β * L.λ[2] * u_s, MOBILITY_LOG_MAX)) :
                        L.ν * exp(min(0.5 * (-β * (L.λ[k-1] - L.λ[k]) * u_s + L.η[k-1] - L.η[k]), MOBILITY_LOG_MAX))
    end
    return up, down
end

"""
    hub_rate(L, β, u_s) -> total rate of the λ = 0 hub jumps from the current rung (0 without the hub, and at rung 2)

Rung m ≥ 3: a e^{½(η_1 − η_m) + βλ_m u_s} to rung 1. Rung 1: Σ_{m≥3} a e^{½(η_m − η_1)}; `hub_target` draws m.
"""
function hub_rate(L::CouplingLadder, β::Float64, u_s::Float64)
    L.jump || return 0.0
    k = L.k; K = nrungs(L)
    if k == 1
        w = 0.0
        for m in 3:L.jump_max
            w += L.jump_ν * exp(min(0.5 * (L.η[m] - L.η[1]), MOBILITY_LOG_MAX))
        end
        return w
    elseif 3 <= k <= L.jump_max
        return L.jump_ν * exp(min(0.5 * (L.η[1] - L.η[k]) + β * L.λ[k] * u_s, MOBILITY_LOG_MAX))
    end
    return 0.0
end
"Target rung of a hub jump from rung 1, drawn ∝ e^{½(η_m − η_1)}, 3 ≤ m ≤ jump_max."
function hub_target(L::CouplingLadder, rng)
    w = [exp(min(0.5 * (L.η[m] - L.η[1]), MOBILITY_LOG_MAX)) for m in 3:L.jump_max]
    ξ = rand(rng) * sum(w)
    for (i, x) in enumerate(w)
        ξ < x && return i + 2
        ξ -= x
    end
    return L.jump_max
end

"""
    wl_update!(L, ev, dt, k)

Wang–Landau in continuous time: the rung k occupied during the residence time dt is penalised by f·ν·dt (a relaxed
state at total rate R contributes ≈ f·ν/R; an overlap "bounce" with dt ≈ 0 contributes nothing), and the flatness
criterion is on the kMC TIME per rung — the quantity the weights must flatten for the occupancy estimator (a flat
EVENT histogram is not flat in time: overlap bounces are events of zero duration, JOURNAL entry 38). f halves when
min_k t_k ≥ wl_flat · mean_k t_k; η is frozen (and shifted to η_K = 0) when f < wl_fmin or the event budget is spent.
"""
function wl_update!(L::CouplingLadder, ev::Int, dt::Float64, k::Int)
    adapting(L) || return nothing
    L.wl_1t && return _wl_update_1t!(L, ev, dt, k)
    L.η[k] -= L.wl_f * L.ν * dt
    L.wl_hist[k] += dt
    L.wl_events_left -= 1
    if L.wl_events_left <= 0
        L.wl_f = 0.0; L.ev_frozen = ev; L.η .-= L.η[end]
        return nothing
    end
    if ev % 1000 == 0
        tt = sum(L.wl_hist)
        if tt > 0 && minimum(L.wl_hist) >= L.wl_flat * tt / nrungs(L)
            L.wl_f *= 0.5; L.wl_stages += 1; fill!(L.wl_hist, 0.0)
            if L.wl_f < L.wl_fmin
                L.wl_f = 0.0; L.ev_frozen = ev; L.η .-= L.η[end]
            end
        end
    end
    return nothing
end

"""
1/t Wang–Landau (Belardinelli & Pereyra, Phys. Rev. E 75 (2007) 046701) in the continuous-time form of `wl_update!`:
η_k −= f·ν·dt; the usual halving on time-flatness (checked every 1000 events) until a halving gives f ≤ 1/t
(t = ν·T_WL/K); from then on f = 1/t at every event, with no flatness check and no freeze. When the `wl_events`
budget is spent before the switch, f = 1/t from there. Production starts at the switch (`ev_frozen`). η is shifted to η_K = 0 after every update (hop rates and both estimators
depend on differences only), which keeps e^{−η} finite in the occupancy integral.
"""
function _wl_update_1t!(L::CouplingLadder, ev::Int, dt::Float64, k::Int)
    K = nrungs(L)
    L.η[k] -= L.wl_f * L.ν * dt
    L.wl_T += dt
    L.wl_events_left -= 1
    f1t = K / (L.ν * L.wl_T)
    if L.wl_phase1t
        L.wl_f = f1t
    else
        sw = false
        L.wl_hist[k] += dt
        if ev % 1000 == 0
            tt = sum(L.wl_hist)
            if tt > 0 && minimum(L.wl_hist) >= L.wl_flat * tt / K
                L.wl_f *= 0.5; L.wl_stages += 1; fill!(L.wl_hist, 0.0)
                sw = L.wl_f <= f1t                                         # the switch is tested after each halving
            end
        end
        sw |= L.wl_events_left <= 0                                        # budget spent before the switch
        sw && (L.wl_phase1t = true; L.wl_f = f1t; L.ev_frozen = ev)        # production starts here (η keeps adapting)
    end
    L.η .-= L.η[end]
    return nothing
end

"Production: residence time dt at the current rung with bare solute energy βu_s (rungs ≥ 2)."
function ladder_time!(L::CouplingLadder, βu_s::Float64, dt::Float64)
    L.n_ev += 1
    b = _lblock(L); _ensure_lblock!(L, b)
    k = L.k
    if k != L.res_k                                     # a new residence: close the previous one
        _close_residence!(L)
        L.res_k = k
    end
    L.res_t += dt; L.pres_t += dt
    L.t_rung[b][k] += dt; L.t2_rung[b][k] += dt * dt
    L.wl_1t && (L.occ_w[b][k] += dt * exp(-L.η[k]))
    k > 1 && (L.hist[b][k][_lbin(L, βu_s)] += dt)
    return nothing
end
function _close_residence!(L::CouplingLadder)
    if L.res_k > 0
        L.res_max[L.res_k] = max(L.res_max[L.res_k], L.res_t)
        L.pres_max[L.res_k] = max(L.pres_max[L.res_k], L.pres_t)
    end
    L.res_t = 0.0; L.pres_t = 0.0
    return nothing
end
"Production: the fractional molecule has moved without a rung change (closes the pose residence)."
function ladder_pose_moved!(L::CouplingLadder)
    L.res_k > 0 && (L.pres_max[L.res_k] = max(L.pres_max[L.res_k], L.pres_t))
    L.pres_t = 0.0
    return nothing
end
"Production: a recoupling landing with bare solute energy βu_s′ (a λ = 0 sample)."
function ladder_landing!(L::CouplingLadder, βu_s::Float64)
    L.n_ev == 0 && return nothing
    b = _lblock(L); _ensure_lblock!(L, b)
    L.hist[b][1][_lbin(L, βu_s)] += 1.0
    return nothing
end

# occupancy estimator with the η in force in each block: t_K^b e^{−η_K^b} / (t_1^b e^{−η_1^b}) = Z_K/Z_1 for every block, so
# e^{−βμ} = Σ_b t_K^b e^{−η_K^b} / Σ_b t_1^b e^{−η_1^b} (reduces to (η_K − η_1) − ln(t_K/t_1) for constant η)
function _occupancy_solve(L::CouplingLadder, skip::Int)
    K = nrungs(L)
    s1 = 0.0; sK = 0.0
    if L.wl_1t                                          # η in force at every event (∫ 1_k e^{−η_k(t)} dt)
        for b in 1:nblocks(L)
            b == skip && continue
            s1 += L.occ_w[b][1]; sK += L.occ_w[b][K]
        end
        (s1 > 0 && sK > 0) || return NaN
        return -log(sK / s1)
    end
    for b in 1:nblocks(L)
        b == skip && continue
        η = b <= length(L.η_blocks) ? L.η_blocks[b] : L.η
        s1 += L.t_rung[b][1] * exp(-η[1]); sK += L.t_rung[b][K] * exp(-η[K])
    end
    (s1 > 0 && sK > 0) || return NaN
    return -log(sK / s1)
end
function _jackknife(f, L::CouplingLadder, T::Float64)
    n = nblocks(L)
    full = T * f(L, 0)
    n < 2 && return (full, NaN)
    θ = [T * f(L, b) for b in 1:n]
    any(!isfinite, θ) && return (full, NaN)
    θ̄ = sum(θ) / n
    return (full, sqrt((n - 1) / n * sum((x - θ̄)^2 for x in θ)))
end
"μ_ex (reduced) from the rung occupancies, with the jackknife error."
ladder_mu_occupancy(L::CouplingLadder, T::Float64) = _jackknife(_occupancy_solve, L, T)

"""
    mbar_histograms(λ, H, Neff, βu_min, dβu; tol=1e-10, maxiter=200) -> f_K − f_1 (= βΔF between the last and first state)

MBAR (Shirts & Chodera, J. Chem. Phys. 129 (2008) 124105) for K states with reduced potentials u_k(x) = λ_k·βu_s(x),
given per-state histograms H[k] of βu_s (bin i centred at βu_min + (i − ½)dβu; weights proportional to the sample
weights) and effective sample sizes Neff[k]. The MBAR equations N_i = Σ_n w_n p_{n,i}, p_{n,i} = N_i e^{f_i − u_i(n)} /
Σ_k N_k e^{f_k − u_k(n)}, f_1 ≡ 0, are solved by a short self-consistent warm start followed by damped Newton–Raphson
(Jacobian ∂g_i/∂f_j = −Σ_n w_n p_{n,i}(δ_ij − p_{n,j}), Appendix C of Shirts & Chodera; the step is halved until the
residual norm decreases). Used by the coupling ladder (time-weighted kMC histograms, Kish Neff) and by the Metropolis
reference of dev/mbar_mc_lj.jl (sample counts).
"""
function mbar_histograms(λ::Vector{Float64}, H::Vector{Vector{Float64}}, Neff::Vector{Float64}, βu_min::Float64, dβu::Float64;
                         tol::Float64=1e-10, maxiter::Int=200)
    K = length(λ); nb = length(H[1])
    (length(H) == K && length(Neff) == K) || error("mbar_histograms: one histogram and one Neff per state")
    any(x -> !(x > 0), Neff) && return NaN
    cs = [βu_min + (i - 0.5) * dβu for i in 1:nb]
    idx = [(k, i) for k in 1:K for i in 1:nb if H[k][i] > 0]
    n = length(idx)
    w = [H[k][i] / sum(H[k]) * Neff[k] for (k, i) in idx]      # Σ_{n∈k} w_n = Neff[k]
    c = [cs[i] for (k, i) in idx]
    logN = log.(Neff)
    p = zeros(K); g = zeros(K); J = zeros(K, K); s = zeros(K)
    function residual!(f, wantJ)
        fill!(g, 0.0); fill!(s, 0.0); wantJ && fill!(J, 0.0)
        @inbounds for m in 1:n
            mx = -Inf
            for k in 1:K
                v = logN[k] + f[k] - λ[k] * c[m]; p[k] = v; v > mx && (mx = v)
            end
            d = 0.0
            for k in 1:K
                p[k] = exp(p[k] - mx); d += p[k]
            end
            for i in 1:K
                s[i] += w[m] * exp(-λ[i] * c[m] - mx) / d
                p[i] /= d
            end
            for i in 1:K
                g[i] += w[m] * p[i]
                wantJ && for j in 1:K
                    J[i, j] -= w[m] * p[i] * ((i == j) - p[j])
                end
            end
        end
        g .= Neff .- g
        return sqrt(sum(abs2, g))
    end
    f = zeros(K); fnew = zeros(K)
    for _ in 1:50                                               # self-consistent warm start
        residual!(f, false)
        fnew .= -log.(s); fnew .-= fnew[1]
        f .= fnew
    end
    r0 = residual!(f, true)
    for it in 1:maxiter
        r0 < tol * sum(Neff) && break
        δ = zeros(K)
        try
            δ[2:K] .= J[2:K, 2:K] \ g[2:K]
        catch
            fnew .= -log.(s); fnew .-= fnew[1]; δ .= f .- fnew          # singular Jacobian: fixed-point step
        end
        α = 1.0; r1 = Inf
        while true
            fnew .= f .- α .* δ
            r1 = residual!(fnew, false)
            (r1 < r0 || α < 1e-6) && break
            α *= 0.5
        end
        f .= fnew
        r0 = residual!(f, true)
        maximum(abs.(α .* δ)) < tol && break
    end
    return f[K] - f[1]
end

# MBAR over the ladder's rung histograms (rung 1: recoupling landing counts; rungs ≥ 2: kMC time with Kish Neff)
function _mbar_solve(L::CouplingLadder, skip::Int)
    K = nrungs(L); nb = L.nbins
    H = [zeros(nb) for _ in 1:K]; Neff = zeros(K)
    t = zeros(K); t2 = zeros(K)
    for b in 1:nblocks(L)
        b == skip && continue
        for k in 1:K
            H[k] .+= L.hist[b][k]; t[k] += L.t_rung[b][k]; t2[k] += L.t2_rung[b][k]
        end
    end
    Neff[1] = sum(H[1])
    for k in 2:K
        Neff[k] = t2[k] > 0 ? t[k]^2 / t2[k] : 0.0
    end
    return mbar_histograms(L.λ, H, Neff, L.βu_min, L.dβu)
end
"μ_ex (reduced) from MBAR over the rung energy histograms, with the jackknife error."
ladder_mu_mbar(L::CouplingLadder, T::Float64) = _jackknife(_mbar_solve, L, T)

"Time per rung (production), hops per rung, recouplings, η."
function ladder_summary(L::CouplingLadder)
    K = nrungs(L)
    t = zeros(K)
    for b in 1:nblocks(L); t .+= L.t_rung[b]; end
    land = sum(sum(L.hist[b][1]) for b in 1:nblocks(L); init=0.0)
    # D0: longest residence (rung; pose) per rung as a fraction of that rung's production time, the open one included
    rmax = copy(L.res_max); pmax = copy(L.pres_max)
    if L.res_k > 0
        rmax[L.res_k] = max(rmax[L.res_k], L.res_t); pmax[L.res_k] = max(pmax[L.res_k], L.pres_t)
    end
    res_frac = [t[k] > 0 ? rmax[k] / t[k] : NaN for k in 1:K]
    pose_frac = [t[k] > 0 ? pmax[k] / t[k] : NaN for k in 1:K]
    return (t_rung=t, frac_rung=t ./ max(sum(t), eps()), n_hop=copy(L.n_hop), n_jump=copy(L.n_jump), n_recouple=L.n_recouple, n_decouple=L.n_decouple,
            landings=land, η=copy(L.η), wl_stages=L.wl_stages, ev_frozen=L.ev_frozen, wl_f=L.wl_f,
            adapt_blocks=L.adapt_blocks, wl_1t=L.wl_1t, wl_phase1t=L.wl_phase1t, η_first=isempty(L.η_blocks) ? copy(L.η) : copy(L.η_blocks[1]),
            res_frac=res_frac, pose_frac=pose_frac)
end

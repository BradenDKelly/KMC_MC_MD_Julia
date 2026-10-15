using StaticArrays

"""
Rejection-free local moves of the fractional molecule ("L-moves", DESIGN §11.2, JOURNAL entries 58–61).

The state is augmented by a candidate set D = (d_1 … d_K) for the fractional molecule s, drawn independently of the
configuration from SYMMETRIC densities (g(d) = g(d⁻¹)); the augmented target π(x, k)·Π_j g(d_j) has the (x, k) marginal of
the coupling ladder, so every estimator is unchanged. Each slot j is an event:

    x → d_j·x  and  d_j → d_j⁻¹   (an involution: the slot then holds the move back)
    rate  ν_L exp(−½ β λ_k [u_s(d_j·x) − u_s(x)])        (u_s bare, capped; λ_k the current rung)

so π(x)g(d)·rate(x→d·x) = π(d·x)g(d⁻¹)·rate(d·x→x) (detailed balance on the augmented space, no accept/reject), plus a
refresh event of constant rate ν_D that redraws D from g (a Gibbs step on D; leaves the augmented target invariant).
Slot kinds:
  LM_T  translation of the COM by δ uniform in [−a, a]³                          inverse −δ
  LM_R  rotation about the COM, left quaternion product d ⊗ q, axis uniform,     inverse = conjugate of d
        angle uniform in [−θ_max, θ_max] (Haar measure preserved)
  LM_C  conformer swap: the slot holds a reservoir conformer B′ (drawn uniformly from the ConformerReservoir, i.e. from
        e^{−βU_intra}); firing installs B′ at the same COM and orientation and leaves the old body in the slot. With the
        stock convention (conformer from the reservoir, orientation uniform) U_intra cancels and the rate carries only
        the intermolecular change.
L-moves are active at rungs with λ_k > 0 (at λ = 0 the pose is integrated out). The candidate energies are kept as bare
pair rows cand_pair[j, i]: after a move of another molecule i only column i is re-evaluated (K pair energies); after a
move of s, an MC-kernel sweep or a refresh they are rebuilt (K rows) lazily, at the next event that needs them.
The construction is a multi-candidate form of continuous-time samplers with a flip involution and a refreshed
auxiliary variable (closest published: Jansson et al., arXiv 2504.12190 (2025); literature/NOVELTY_SEARCH_2026-10-14.md).
The functions take the RigidSystem / SiteForceField / RigidEKMCState of RigidMolecules.jl (included after this file).
"""
const LM_T = 1
const LM_R = 2
const LM_C = 3

mutable struct LocalMoves
    mol::Int                            # the fractional molecule
    kinds::Vector{Int}                  # slot kinds (LM_T, LM_R, LM_C)
    a::Float64                          # translation half-width (reduced length)
    θmax::Float64                       # rotation half-angle (rad)
    ν::Float64                          # per-candidate prefactor ν_L
    νD::Float64                         # refresh rate ν_D
    dT::Matrix{Float64}                 # 3 × K translations (LM_T slots)
    dQ::Matrix{Float64}                 # 4 × K rotation quaternions (LM_R slots)
    dB::Vector{Matrix{Float64}}         # conformers (LM_C slots; 3 × 0 otherwise)
    cand_pair::Matrix{Float64}          # K × N bare pair energies of each candidate pose with every molecule
    cand_u::Vector{Float64}             # K bare candidate energies u_s(d_j·x)
    rates::Vector{Float64}              # K rates at the current rung (filled by lm_total_rate!)
    stale::Bool                         # candidate energies must be rebuilt before use
    n_move::Vector{Int}                 # fired moves per kind (all events)
    n_move_rung::Vector{Int}            # fired moves per rung (production)
    n_refresh::Int
    n_rebuild::Int
    row_u::Vector{Float64}              # scratch rows
    row_w::Vector{Float64}
    com0::Vector{Float64}               # scratch: saved pose of s
    q0::Vector{Float64}
    B0::Matrix{Float64}
end

"""
    LocalMoves(sys, s, nrungs; kinds, a, θmax, ν, νD, reservoir=nothing)

Candidate set for molecule s with the given slot kinds (LM_C slots need the molecule's `ConformerReservoir`).
The slots are drawn at once; the candidate energies are built at the first event that needs them.
"""
function LocalMoves(sys, s::Int, nrungs::Int; kinds::Vector{Int}, a::Float64, θmax::Float64, ν::Float64, νD::Float64,
                    reservoir=nothing)
    K = length(kinds)
    K >= 1 || error("LocalMoves: at least one slot")
    all(k -> k in (LM_T, LM_R, LM_C), kinds) || error("LocalMoves: slot kinds must be LM_T, LM_R or LM_C")
    (a > 0 && θmax > 0 && ν > 0 && νD > 0) || error("LocalMoves: a, θmax, ν, νD must be positive")
    any(==(LM_C), kinds) && reservoir === nothing && error("LocalMoves: conformer slots need the molecule's reservoir")
    N = sys.n_mol
    nB = size(sys.body[s], 2)
    lm = LocalMoves(s, copy(kinds), a, θmax, ν, νD, zeros(3, K), zeros(4, K), [zeros(3, 0) for _ in 1:K],
                    zeros(K, N), zeros(K), zeros(K), true, zeros(Int, 3), zeros(Int, nrungs), 0, 0,
                    zeros(N), zeros(N), zeros(3), zeros(4), zeros(3, nB))
    lm_refresh!(lm, sys, reservoir)
    lm.n_refresh = 0
    return lm
end

"Redraw every slot from its symmetric density (the refresh event); marks the candidate energies stale."
function lm_refresh!(lm::LocalMoves, sys, reservoir)
    rng = sys.rng
    for j in eachindex(lm.kinds)
        kd = lm.kinds[j]
        if kd == LM_T
            for d in 1:3
                lm.dT[d, j] = (2rand(rng) - 1) * lm.a
            end
        elseif kd == LM_R
            lm.dQ[:, j] .= small_rotation_quaternion(rng, lm.θmax)
        else
            lm.dB[j] = copy(draw_conformer(reservoir, rng))
        end
    end
    lm.stale = true
    lm.n_refresh += 1
    return nothing
end

# save / restore the pose of s bit-exactly (set_molecule! would renormalise q)
function _lm_save!(lm::LocalMoves, sys)
    s = lm.mol
    @inbounds for d in 1:3; lm.com0[d] = sys.com[d, s]; end
    @inbounds for d in 1:4; lm.q0[d] = sys.quat[d, s]; end
    isflexible(sys.templates[sys.tmpl[s]]) && copyto!(lm.B0, sys.body[s])
    return nothing
end
function _lm_restore!(lm::LocalMoves, sys)
    s = lm.mol
    @inbounds for d in 1:3; sys.com[d, s] = lm.com0[d]; end
    @inbounds for d in 1:4; sys.quat[d, s] = lm.q0[d]; end
    isflexible(sys.templates[sys.tmpl[s]]) && _set_body!(sys, s, lm.B0)
    update_site_offsets!(sys, s)
    return nothing
end
# put s at candidate j (pose saved in com0/q0/B0)
function _lm_place!(lm::LocalMoves, sys, j::Int)
    s = lm.mol; kd = lm.kinds[j]
    if kd == LM_T
        set_molecule!(sys, s, MVector{3,Float64}(lm.com0[1] + lm.dT[1, j], lm.com0[2] + lm.dT[2, j], lm.com0[3] + lm.dT[3, j]), lm.q0)
    elseif kd == LM_R
        set_molecule!(sys, s, lm.com0, quat_mul(view(lm.dQ, :, j), lm.q0))
    else
        _set_body!(sys, s, lm.dB[j])
        @inbounds for d in 1:4; sys.quat[d, s] = lm.q0[d]; end
        update_site_offsets!(sys, s)
    end
    return nothing
end

"Rebuild every candidate row (K row evaluations)."
function lm_rebuild!(lm::LocalMoves, sys, ff)
    _lm_save!(lm, sys)
    N = sys.n_mol
    for j in eachindex(lm.kinds)
        _lm_place!(lm, sys, j)
        eval_row!(lm.row_u, lm.row_w, sys, ff, lm.mol)
        u = 0.0
        @inbounds for i in 1:N
            lm.cand_pair[j, i] = lm.row_u[i]; u += lm.row_u[i]
        end
        lm.cand_u[j] = u
    end
    _lm_restore!(lm, sys)
    lm.stale = false
    lm.n_rebuild += 1
    return nothing
end

"Molecule i ≠ s has moved: re-evaluate column i of the candidate rows (K pair energies)."
function lm_molecule_moved!(lm::LocalMoves, sys, ff, i::Int)
    lm.stale && return nothing
    _lm_save!(lm, sys)
    for j in eachindex(lm.kinds)
        _lm_place!(lm, sys, j)
        u, _ = pair_energy_virial(sys, ff, lm.mol, i)
        lm.cand_u[j] += u - lm.cand_pair[j, i]
        lm.cand_pair[j, i] = u
    end
    _lm_restore!(lm, sys)
    return nothing
end

"""
    lm_total_rate!(lm, sys, ff, β, λ, u_s) -> W

Total L-move rate Σ_j ν_L exp(−½βλ(u_j − u_s)) at coupling λ (0 at λ = 0), rebuilding stale candidates first.
"""
function lm_total_rate!(lm::LocalMoves, sys, ff, β::Float64, λ::Float64, u_s::Float64)
    λ > 0.0 || (fill!(lm.rates, 0.0); return 0.0)
    lm.stale && lm_rebuild!(lm, sys, ff)
    W = 0.0
    @inbounds for j in eachindex(lm.kinds)
        r = lm.ν * exp(min(-0.5 * β * λ * (lm.cand_u[j] - u_s), MOBILITY_LOG_MAX))
        lm.rates[j] = r; W += r
    end
    return W
end

"""
    lm_fire!(lm, sys, ff, ek, ξ) -> j

Fire the slot selected by ξ ∈ [0, Σ rates) (rates from the last `lm_total_rate!`): move s to candidate j through the
eKMC bookkeeping, store the inverse move in slot j and mark the candidates stale.
"""
function lm_fire!(lm::LocalMoves, sys, ff, ek, ξ::Float64)
    K = length(lm.kinds)
    j = K
    @inbounds for i in 1:K
        ξ < lm.rates[i] && (j = i; break)
        ξ -= lm.rates[i]
    end
    s = lm.mol
    _lm_save!(lm, sys)
    _lm_place!(lm, sys, j)
    kd = lm.kinds[j]
    if kd == LM_T
        @inbounds for d in 1:3; lm.dT[d, j] = -lm.dT[d, j]; end
    elseif kd == LM_R
        @inbounds for d in 2:4; lm.dQ[d, j] = -lm.dQ[d, j]; end
    else
        lm.dB[j] = copy(lm.B0)                      # the slot now holds the old conformer
    end
    update_rigid_ekmc_after_move!(ek, sys, ff, s)
    lm.stale = true
    lm.n_move[kd] += 1
    return j
end

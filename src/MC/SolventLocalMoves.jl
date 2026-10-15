using StaticArrays

"""
Rejection-free local moves of every solvent molecule ("solvent L-moves", DESIGN §11.2 stage 2, JOURNAL 65–66).

The construction of `LocalMoves` (LocalMoves.jl) applied to each molecule i of a set (all rigid molecules except the
fractional one): molecule i carries K candidate moves D_i = (d_i1 … d_iK) drawn independently from symmetric densities
(translation δ uniform in [−a, a]³, rotation about the COM by an angle uniform in [−θ_max, θ_max] about a uniform axis);
slot j is an event x → d_ij·x, d_ij → d_ij⁻¹ (an involution) at rate

    ν exp(−½ β [H(d_ij·x) − H(x)]),   H(d_ij·x) − H(x) = Σ_{m ≠ s} u_im(cand) + λ_s u_is(cand) − φ_i,

with s the fractional molecule of a coupling ladder (if any) and φ_i the maintained (λ-scaled) energy of i, so detailed
balance holds on the space augmented by all candidate sets with the target of the run (canonical, or the ladder's H_k);
no accept/reject. A refresh event of rate ν_D per molecule redraws one molecule's set. The candidate pair energies are
kept bare (cp[j, m, ii] for slot j of the ii-th mover and partner m): after any molecule m moves, column m is re-evaluated
for every mover whose stored values are non-zero or whose COM lies within reach (cut-off + extents + √3 a) of m's new
position — K pair energies per such mover — and m's own K rows are rebuilt. Rates are recomputed from cu, cs and φ at every
event (O(MK) exponentials). Rigid molecules only. Purpose: let the solvent relax by small moves between relocations — the
liquid's hydrogen-bond network (JOURNAL 50) and the quenched start that stock eKMC does not melt (JOURNAL 65).
"""
mutable struct SolventLocalMoves
    mols::Vector{Int}                   # molecules with local moves
    pos::Vector{Int}                    # pos[i] = index of molecule i in mols (0 = none)
    kinds::Vector{Int}                  # slot kinds (LM_T, LM_R), the same for every molecule
    a::Float64
    θmax::Float64
    ν::Float64
    νD::Float64
    frac::Int                           # fractional molecule of a ladder (0 = none)
    dT::Array{Float64,3}                # 3 × K × M
    dQ::Array{Float64,3}                # 4 × K × M
    cp::Array{Float64,3}                # K × N × M bare candidate pair energies
    cu::Matrix{Float64}                 # K × M: Σ_{m ≠ frac} cp
    cs::Matrix{Float64}                 # K × M: cp with the fractional molecule
    rates::Matrix{Float64}              # K × M
    Wm::Vector{Float64}                 # per-mover total rate
    W::Float64
    n_move::Vector{Int}                 # fired moves per kind
    n_refresh::Int
    n_rebuild::Int
    n_colupd::Int                       # column updates (mover, partner) evaluated
    row_u::Vector{Float64}
    row_w::Vector{Float64}
    com0::Vector{Float64}
    q0::Vector{Float64}
end

function SolventLocalMoves(sys, ff, mols::Vector{Int}; kinds::Vector{Int}, a::Float64, θmax::Float64, ν::Float64,
                           νD::Float64, frac::Int=0)
    N = sys.n_mol; M = length(mols); K = length(kinds)
    (K >= 1 && M >= 1) || error("SolventLocalMoves: at least one molecule and one slot")
    all(k -> k in (LM_T, LM_R), kinds) || error("SolventLocalMoves: slot kinds must be LM_T or LM_R (rigid molecules)")
    (a > 0 && θmax > 0 && ν > 0 && νD > 0) || error("SolventLocalMoves: a, θmax, ν, νD must be positive")
    for i in mols
        isflexible(sys.templates[sys.tmpl[i]]) && error("SolventLocalMoves: molecule $i is flexible (rigid solvent only)")
        i == frac && error("SolventLocalMoves: the fractional molecule has its own LocalMoves")
    end
    pos = zeros(Int, N)
    for (ii, i) in enumerate(mols); pos[i] = ii; end
    S = SolventLocalMoves(copy(mols), pos, copy(kinds), a, θmax, ν, νD, frac, zeros(3, K, M), zeros(4, K, M),
                          zeros(K, N, M), zeros(K, M), zeros(K, M), zeros(K, M), zeros(M), 0.0, zeros(Int, 3), 0, 0, 0,
                          zeros(N), zeros(N), zeros(3), zeros(4))
    for ii in 1:M
        _slm_draw!(S, sys, ii)
    end
    slm_rebuild_all!(S, sys, ff)
    return S
end

function _slm_draw!(S::SolventLocalMoves, sys, ii::Int)
    rng = sys.rng
    for j in eachindex(S.kinds)
        if S.kinds[j] == LM_T
            for d in 1:3; S.dT[d, j, ii] = (2rand(rng) - 1) * S.a; end
        else
            q = small_rotation_quaternion(rng, S.θmax)
            for d in 1:4; S.dQ[d, j, ii] = q[d]; end
        end
    end
end
function _slm_save!(S::SolventLocalMoves, sys, i::Int)
    @inbounds for d in 1:3; S.com0[d] = sys.com[d, i]; end
    @inbounds for d in 1:4; S.q0[d] = sys.quat[d, i]; end
end
function _slm_restore!(S::SolventLocalMoves, sys, i::Int)
    @inbounds for d in 1:3; sys.com[d, i] = S.com0[d]; end
    @inbounds for d in 1:4; sys.quat[d, i] = S.q0[d]; end
    update_site_offsets!(sys, i)
end
function _slm_place!(S::SolventLocalMoves, sys, i::Int, ii::Int, j::Int)
    if S.kinds[j] == LM_T
        set_molecule!(sys, i, MVector{3,Float64}(S.com0[1] + S.dT[1, j, ii], S.com0[2] + S.dT[2, j, ii], S.com0[3] + S.dT[3, j, ii]), S.q0)
    else
        set_molecule!(sys, i, S.com0, quat_mul(view(S.dQ, :, j, ii), S.q0))
    end
end

"Rebuild the K candidate rows of the ii-th mover."
function slm_rebuild_mol!(S::SolventLocalMoves, sys, ff, ii::Int)
    i = S.mols[ii]; N = sys.n_mol
    _slm_save!(S, sys, i)
    for j in eachindex(S.kinds)
        _slm_place!(S, sys, i, ii, j)
        eval_row!(S.row_u, S.row_w, sys, ff, i)
        u = 0.0
        @inbounds for m in 1:N
            S.cp[j, m, ii] = S.row_u[m]
            m == S.frac || (u += S.row_u[m])
        end
        S.cu[j, ii] = u
        S.cs[j, ii] = S.frac > 0 ? S.row_u[S.frac] : 0.0
    end
    _slm_restore!(S, sys, i)
    S.n_rebuild += 1
    return nothing
end
slm_rebuild_all!(S::SolventLocalMoves, sys, ff) = (for ii in eachindex(S.mols); slm_rebuild_mol!(S, sys, ff, ii); end)

"""
    slm_moved!(S, sys, ff, m)

Molecule m has moved (any event): rebuild its own candidate rows if it is a mover, and re-evaluate column m of every other
mover that can interact with m (stored value non-zero, or COM within reach of m's new position).
"""
function slm_moved!(S::SolventLocalMoves, sys, ff, m::Int)
    S.pos[m] > 0 && slm_rebuild_mol!(S, sys, ff, S.pos[m])
    L = sys.L; K = length(S.kinds)
    reach0 = ff.cutoff === :molecule ? ff.rc : ff.rc + sys.ext[m]
    @inbounds for ii in eachindex(S.mols)
        i = S.mols[ii]
        i == m && continue
        nonzero = false
        for j in 1:K
            S.cp[j, m, ii] != 0.0 && (nonzero = true; break)
        end
        if !nonzero
            dx = _min_image(sys.com[1, m] - sys.com[1, i], L); dy = _min_image(sys.com[2, m] - sys.com[2, i], L)
            dz = _min_image(sys.com[3, m] - sys.com[3, i], L)
            r = reach0 + (ff.cutoff === :molecule ? 0.0 : sys.ext[i]) + sqrt(3.0) * S.a
            dx * dx + dy * dy + dz * dz >= r * r && continue
        end
        _slm_save!(S, sys, i)
        for j in 1:K
            _slm_place!(S, sys, i, ii, j)
            u, _ = pair_energy_virial(sys, ff, i, m)
            if m == S.frac
                S.cs[j, ii] += u - S.cp[j, m, ii]
            else
                S.cu[j, ii] += u - S.cp[j, m, ii]
            end
            S.cp[j, m, ii] = u
        end
        _slm_restore!(S, sys, i)
        S.n_colupd += 1
    end
    return nothing
end

"Total solvent L-move rate at the current state (rates from cu, cs and the maintained φ; λ of the fractional molecule)."
function slm_total_rate!(S::SolventLocalMoves, ek, β::Float64)
    λ = S.frac > 0 ? ek.lam[S.frac] : 0.0
    W = 0.0
    @inbounds for ii in eachindex(S.mols)
        φ = ek.phi[S.mols[ii]]; w = 0.0
        for j in eachindex(S.kinds)
            r = S.ν * exp(min(-0.5 * β * (S.cu[j, ii] + λ * S.cs[j, ii] - φ), MOBILITY_LOG_MAX))
            S.rates[j, ii] = r; w += r
        end
        S.Wm[ii] = w; W += w
    end
    S.W = W
    return W
end

"Fire the solvent L-move selected by ξ ∈ [0, W); returns the moved molecule (bookkeeping of ek and S done)."
function slm_fire!(S::SolventLocalMoves, sys, ff, ek, ξ::Float64)
    M = length(S.mols); K = length(S.kinds)
    ii = M
    @inbounds for x in 1:M
        ξ < S.Wm[x] && (ii = x; break)
        ξ -= S.Wm[x]
    end
    j = K
    @inbounds for y in 1:K
        ξ < S.rates[y, ii] && (j = y; break)
        ξ -= S.rates[y, ii]
    end
    i = S.mols[ii]
    _slm_save!(S, sys, i)
    _slm_place!(S, sys, i, ii, j)
    if S.kinds[j] == LM_T
        @inbounds for d in 1:3; S.dT[d, j, ii] = -S.dT[d, j, ii]; end
    else
        @inbounds for d in 2:4; S.dQ[d, j, ii] = -S.dQ[d, j, ii]; end
    end
    update_rigid_ekmc_after_move!(ek, sys, ff, i)
    slm_moved!(S, sys, ff, i)
    S.n_move[S.kinds[j]] += 1
    return i
end

"Refresh event: redraw the candidate set of a uniformly chosen mover and rebuild its rows."
function slm_refresh!(S::SolventLocalMoves, sys, ff)
    ii = rand(sys.rng, eachindex(S.mols))
    _slm_draw!(S, sys, ii)
    slm_rebuild_mol!(S, sys, ff, ii)
    S.n_refresh += 1
    return nothing
end

"Largest |cu − rebuilt cu| and |cs − rebuilt cs| (debug check; rebuilds S)."
function slm_check!(S::SolventLocalMoves, sys, ff)
    cu = copy(S.cu); cs = copy(S.cs)
    slm_rebuild_all!(S, sys, ff)
    return max(maximum(abs.(cu .- S.cu)), maximum(abs.(cs .- S.cs)))
end

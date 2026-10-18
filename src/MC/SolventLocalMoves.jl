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
event (O(MK) exponentials). Flexible movers (DESIGN §14) also carry single-site (LM_S) and torsion (LM_D) slots — the slot
kinds of `LocalMoves` — with rate ν exp(−½β[ΔU_inter + ΔU_intra]); ΔU_intra per slot (cin) is recomputed with the mover's rows. Purpose: let the solvent relax by small moves between relocations — the
liquid's hydrogen-bond network (JOURNAL 50) and the quenched start that stock eKMC does not melt (JOURNAL 65).
With reciprocal-space Ewald (EwaldRecip.jl) cp/cu/cs hold the real-space + LJ part (local, updated as above); the Fourier
and image part of every candidate is kept separately — cr_u[j, ii] = Σ_k c_k Re[S_cand^*(Stot − S_i − S_frac)] + e(S_cand),
cr_s[j, ii] = Σ_k c_k Re[S_cand^* S_frac] — with the candidate structure factors crS (N_k × K·M). After a move of m every
candidate of every other mover changes by Σ_k c_k Re[S_cand^* ΔS_m]: one matrix–vector product Re[crSᴴ(c ∘ ΔS_m)] (real, over the interleaved storage).
"""
mutable struct SolventLocalMoves
    mols::Vector{Int}                   # molecules with local moves
    pos::Vector{Int}                    # pos[i] = index of molecule i in mols (0 = none)
    kinds::Vector{Int}                  # slot kinds (LM_T, LM_R; LM_S, LM_D for flexible movers), the same for every molecule
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
    pose::Vector{Any}                   # per-thread scratch SLMPose (saved pose of the molecule being evaluated)
    crS::Matrix{ComplexF64}             # reciprocal-space Ewald: N_k × (K·M) candidate structure factors (N_k × 0 otherwise)
    cr_u::Matrix{Float64}               # K × M Fourier + image energy of each candidate with all but i and frac
    cr_s::Matrix{Float64}               # K × M Fourier energy of each candidate with the fractional molecule
    cr_tmp::Vector{Float64}             # K·M scratch (Re[crSᴴ (c ∘ ΔS)])
    crSd::AbstractMatrix                # crS on the reciprocal-space backend (=== crS for :cpu/:double; ComplexF32 for :mixed)
    cr_tmpd::AbstractVector             # K·M real on the backend
    crSdr::AbstractMatrix               # crSd as a real 2N_k × K·M matrix (`_realview`, same memory)
    cr_stage::Vector                    # K·M host staging in the backend real type (allocation-free download)
    cr_stage_up::Matrix                 # N_k × K host staging in the backend complex type (allocation-free upload)
    # flexible movers (DESIGN §14): single-site (LM_S) and torsion (LM_D) slots
    site_a::Float64                     # LM_S half-width (reduced length)
    φmax::Float64                       # LM_D half-angle (rad)
    slot_idx::Matrix{Int}               # K × M: LM_S site, LM_D index into tors[tmpl]
    cin::Matrix{Float64}                # K × M: U_intra(candidate) − U_intra(current) (0 for T/R)
    tors::Vector{Vector{Tuple{Int,Int,Vector{Int}}}}   # rotatable bonds per template
    cand_site::Vector{Float64}          # M: largest distance of any interacting site of any candidate pose from the saved COM
    cand_com::Vector{Float64}           # M: largest COM displacement of any candidate (molecule cut-off reach)
end

"Saved pose (COM, quaternion, body) of the molecule being evaluated; one per thread (the column updates are threaded)."
mutable struct SLMPose
    com0::Vector{Float64}
    q0::Vector{Float64}
    B0::Matrix{Float64}
end
SLMPose() = SLMPose(zeros(3), zeros(4), zeros(3, 0))

function SolventLocalMoves(sys, ff, mols::Vector{Int}; kinds::Vector{Int}, a::Float64, θmax::Float64, ν::Float64,
                           νD::Float64, frac::Int=0, site_a::Float64=NaN, φmax::Float64=NaN)
    N = sys.n_mol; M = length(mols); K = length(kinds)
    (K >= 1 && M >= 1) || error("SolventLocalMoves: at least one molecule and one slot")
    all(k -> k in (LM_T, LM_R, LM_S, LM_D), kinds) || error("SolventLocalMoves: slot kinds must be LM_T, LM_R, LM_S or LM_D")
    (a > 0 && θmax > 0 && ν > 0 && νD > 0) || error("SolventLocalMoves: a, θmax, ν, νD must be positive")
    internal = any(k -> k == LM_S || k == LM_D, kinds)
    any(==(LM_S), kinds) && !(site_a > 0) && error("SolventLocalMoves: LM_S slots need site_a > 0")
    any(==(LM_D), kinds) && !(φmax > 0) && error("SolventLocalMoves: LM_D slots need φmax > 0")
    tors = [isflexible(t) ? rotatable_bonds(t) : Tuple{Int,Int,Vector{Int}}[] for t in sys.templates]
    for i in mols
        t = sys.templates[sys.tmpl[i]]
        i == frac && error("SolventLocalMoves: the fractional molecule has its own LocalMoves")
        if internal
            isflexible(t) || error("SolventLocalMoves: site/torsion slots but molecule $i ($(t.name)) is rigid")
            any(==(LM_D), kinds) && isempty(tors[sys.tmpl[i]]) && error("SolventLocalMoves: torsion slots but $(t.name) has no rotatable bond")
        end
    end
    pos = zeros(Int, N)
    for (ii, i) in enumerate(mols); pos[i] = ii; end
    S = SolventLocalMoves(copy(mols), pos, copy(kinds), a, θmax, ν, νD, frac, zeros(3, K, M), zeros(4, K, M),
                          zeros(K, N, M), zeros(K, M), zeros(K, M), zeros(K, M), zeros(M), 0.0, zeros(Int, 5), 0, 0, 0,
                          zeros(N), zeros(N), Any[SLMPose() for _ in 1:Threads.nthreads()], zeros(ComplexF64, 0, K * M), zeros(K, M), zeros(K, M),
                          zeros(K * M), zeros(ComplexF64, 0, K * M), zeros(K * M), zeros(0, K * M), Float64[],
                          zeros(ComplexF64, 0, K), site_a, φmax, zeros(Int, K, M), zeros(K, M), tors, zeros(M), zeros(M))
    for ii in 1:M
        _slm_draw!(S, sys, ii)
    end
    slm_rebuild_all!(S, sys, ff)
    return S
end

function _slm_draw!(S::SolventLocalMoves, sys, ii::Int)
    rng = sys.rng
    i = S.mols[ii]
    for j in eachindex(S.kinds)
        kd = S.kinds[j]
        if kd == LM_T
            for d in 1:3; S.dT[d, j, ii] = (2rand(rng) - 1) * S.a; end
        elseif kd == LM_R
            q = small_rotation_quaternion(rng, S.θmax)
            for d in 1:4; S.dQ[d, j, ii] = q[d]; end
        elseif kd == LM_S                                # site uniform, δ uniform in [−a_s, a_s]³ (body frame)
            S.slot_idx[j, ii] = rand(rng, 1:sys.templates[sys.tmpl[i]].n_sites)
            for d in 1:3; S.dT[d, j, ii] = (2rand(rng) - 1) * S.site_a; end
        else                                             # rotatable bond uniform, φ uniform in [−φmax, φmax]
            S.slot_idx[j, ii] = rand(rng, 1:length(S.tors[sys.tmpl[i]]))
            S.dT[1, j, ii] = (2rand(rng) - 1) * S.φmax
        end
    end
end
# save / restore the pose of i bit-exactly (flexible: also the body), in the scratch P (default: thread 1's)
function _slm_save!(S::SolventLocalMoves, sys, i::Int, P::SLMPose=S.pose[1])
    @inbounds for d in 1:3; P.com0[d] = sys.com[d, i]; end
    @inbounds for d in 1:4; P.q0[d] = sys.quat[d, i]; end
    if isflexible(sys.templates[sys.tmpl[i]])
        size(P.B0) == size(sys.body[i]) || (P.B0 = similar(sys.body[i]))
        copyto!(P.B0, sys.body[i])
    end
end
function _slm_restore!(S::SolventLocalMoves, sys, i::Int, P::SLMPose=S.pose[1])
    @inbounds for d in 1:3; sys.com[d, i] = P.com0[d]; end
    @inbounds for d in 1:4; sys.quat[d, i] = P.q0[d]; end
    isflexible(sys.templates[sys.tmpl[i]]) && _set_body!(sys, i, P.B0)
    update_site_offsets!(sys, i)
end
function _slm_place!(S::SolventLocalMoves, sys, i::Int, ii::Int, j::Int, P::SLMPose=S.pose[1])
    kd = S.kinds[j]
    if kd == LM_S || kd == LM_D
        _place_internal!(sys, i, P.B0, P.com0, P.q0, kd, S.slot_idx[j, ii], S.dT[1, j, ii], S.dT[2, j, ii], S.dT[3, j, ii],
                         kd == LM_D ? S.tors[sys.tmpl[i]][S.slot_idx[j, ii]] : nothing)
        return nothing
    end
    # T and R keep the saved body (a preceding S/D candidate changed it; set_molecule! keeps the current body)
    isflexible(sys.templates[sys.tmpl[i]]) && _set_body!(sys, i, P.B0)
    if kd == LM_T
        set_molecule!(sys, i, SVector{3,Float64}(P.com0[1] + S.dT[1, j, ii], P.com0[2] + S.dT[2, j, ii], P.com0[3] + S.dT[3, j, ii]), P.q0)
    else
        set_molecule!(sys, i, P.com0, quat_mul(view(S.dQ, :, j, ii), P.q0))
    end
end

"Rebuild the K candidate rows of the ii-th mover."
function slm_rebuild_mol!(S::SolventLocalMoves, sys, ff, ii::Int)
    i = S.mols[ii]; N = sys.n_mol
    _slm_save!(S, sys, i)
    R = _has_recip(sys, ff) ? sys.recip : nothing
    K = length(S.kinds)
    t = sys.templates[sys.tmpl[i]]
    internal = isflexible(t) && any(k -> k == LM_S || k == LM_D, S.kinds)
    uin0 = internal ? flex_intra_energy(t, ff, S.pose[1].B0) : 0.0
    if R !== nothing
        _slm_bind_recip!(S, R)
        recip_sf!(R.work, R, sys, i)                    # S_i at the saved pose: translated candidates are e^{ik·d} S_i
        corr_i = recip_corr(R, sys, i)                  # unchanged by a rigid translation
    end
    P = S.pose[1]
    rsite = 0.0; rcom = 0.0
    for j in eachindex(S.kinds)
        _slm_place!(S, sys, i, ii, j)
        # reach of the candidate poses (column updates, `_slm_column!`): exact, from the placed sites
        Dx = _min_image(sys.com[1, i] - P.com0[1], sys.L); Dy = _min_image(sys.com[2, i] - P.com0[2], sys.L)
        Dz = _min_image(sys.com[3, i] - P.com0[3], sys.L)
        rcom = max(rcom, sqrt(Dx * Dx + Dy * Dy + Dz * Dz))
        s0 = sys.site_start[i]
        @inbounds for n in 1:t.n_sites
            (t.site_type[n] > 0 || t.site_q[n] != 0.0) || continue
            x = Dx + sys.site_off[1, s0 + n]; y = Dy + sys.site_off[2, s0 + n]; z = Dz + sys.site_off[3, s0 + n]
            rsite = max(rsite, sqrt(x * x + y * y + z * z))
        end
        eval_row!(S.row_u, S.row_w, sys, ff, i; recip=false)
        u = 0.0
        @inbounds for m in 1:N
            S.cp[j, m, ii] = S.row_u[m]
            m == S.frac || (u += S.row_u[m])
        end
        S.cu[j, ii] = u
        S.cs[j, ii] = S.frac > 0 ? S.row_u[S.frac] : 0.0
        kd = S.kinds[j]
        S.cin[j, ii] = (kd == LM_S || kd == LM_D) ? flex_intra_energy(t, ff, sys.body[i]) - uin0 : 0.0
        if R !== nothing
            Sc = view(S.crS, :, (ii - 1) * K + j)
            if S.kinds[j] == LM_T
                recip_sf_translate!(Sc, R, R.work, (S.dT[1, j, ii], S.dT[2, j, ii], S.dT[3, j, ii]))
                ru = recip_half_norm(R, Sc) - corr_i
            else
                recip_sf!(Sc, R, sys, i)
                ru = recip_half_norm(R, Sc) - recip_corr(R, sys, i)
            end
            rs = 0.0
            @inbounds for kk in eachindex(R.c)
                sf = S.frac > 0 ? R.S[kk, S.frac] : zero(ComplexF64)
                a = R.Stot[kk] - R.S[kk, i] - sf
                ru += R.c[kk] * (real(Sc[kk]) * real(a) + imag(Sc[kk]) * imag(a))
                rs += R.c[kk] * (real(Sc[kk]) * real(sf) + imag(Sc[kk]) * imag(sf))
            end
            S.cr_u[j, ii] = ru; S.cr_s[j, ii] = rs
        end
    end
    if R !== nothing && S.crSd !== S.crS                # upload this mover's K candidate columns
        cols = (ii - 1) * K + 1:ii * K
        _upload!(view(S.crSd, :, cols), view(S.crS, :, cols), S.cr_stage_up)
    end
    _slm_restore!(S, sys, i)
    S.cand_site[ii] = rsite; S.cand_com[ii] = rcom
    S.n_rebuild += 1
    return nothing
end
slm_rebuild_all!(S::SolventLocalMoves, sys, ff) = (for ii in eachindex(S.mols); slm_rebuild_mol!(S, sys, ff, ii); end)

"Re-evaluate column m (K pair energies) of the ii-th mover if it can interact with m; true if it did (scratch pose P)."
function _slm_column!(S::SolventLocalMoves, sys, ff, ii::Int, m::Int, reach0::Float64, P::SLMPose)::Bool
    i = S.mols[ii]
    i == m && return false
    K = length(S.kinds); L = sys.L
    nonzero = false
    @inbounds for j in 1:K
        S.cp[j, m, ii] != 0.0 && (nonzero = true; break)
    end
    if !nonzero
        @inbounds dx = _min_image(sys.com[1, m] - sys.com[1, i], L); @inbounds dy = _min_image(sys.com[2, m] - sys.com[2, i], L)
        @inbounds dz = _min_image(sys.com[3, m] - sys.com[3, i], L)
        # no candidate pose of i can interact with m beyond: rc (+ ext_m) + the candidates' own reach (measured at rebuild)
        r = reach0 + (ff.cutoff === :molecule ? S.cand_com[ii] : S.cand_site[ii])
        dx * dx + dy * dy + dz * dz >= r * r && return false
    end
    _slm_save!(S, sys, i, P)
    @inbounds for j in 1:K
        _slm_place!(S, sys, i, ii, j, P)
        u, _ = pair_energy_virial(sys, ff, i, m)
        if m == S.frac
            S.cs[j, ii] += u - S.cp[j, m, ii]
        else
            S.cu[j, ii] += u - S.cp[j, m, ii]
        end
        S.cp[j, m, ii] = u
    end
    _slm_restore!(S, sys, i, P)
    return true
end

"""
    slm_moved!(S, sys, ff, m)

Molecule m has moved (any event): rebuild its own candidate rows if it is a mover, and re-evaluate column m of every other
mover that can interact with m (stored value non-zero, or COM within reach of m's new position).
"""
function slm_moved!(S::SolventLocalMoves, sys, ff, m::Int)
    S.pos[m] > 0 && slm_rebuild_mol!(S, sys, ff, S.pos[m])
    K = length(S.kinds)
    reach0 = ff.cutoff === :molecule ? ff.rc : ff.rc + sys.ext[m]
    M = length(S.mols)
    if _use_threads(M)                                  # columns of different movers are independent (bitwise = serial)
        nt = Threads.nthreads()
        while length(S.pose) < nt; push!(S.pose, SLMPose()); end     # (a checkpoint may come from another thread count)
        cnt = zeros(Int, nt)
        Threads.@threads :static for c in 1:nt              # one chunk of movers per task, scratch indexed by chunk
            P = S.pose[c]
            for ii in (c - 1) * M ÷ nt + 1:c * M ÷ nt
                _slm_column!(S, sys, ff, ii, m, reach0, P) && (cnt[c] += 1)
            end
        end
        S.n_colupd += sum(cnt)
    else
        for ii in 1:M
            _slm_column!(S, sys, ff, ii, m, reach0, S.pose[1]) && (S.n_colupd += 1)
        end
    end
    R = _has_recip(sys, ff) ? sys.recip : nothing
    if R !== nothing                                    # Fourier part of every candidate: crSᴴ (c ∘ ΔS_m)
        R.last == m || error("slm_moved!: reciprocal space last committed molecule $(R.last), not $m")
        R.vd === R.work ? copyto!(R.work, R.dS) : _upload!(R.vd, R.dS, R.stage)
        R.cwd .= R.cd .* R.vd
        mul!(S.cr_tmpd, transpose(S.crSdr), R.cwdr)      # real product over the interleaved storage (`_realview`)
        to_host!(S.cr_tmp, S.cr_tmpd, S.cr_stage)
        own = S.pos[m]
        target = m == S.frac ? S.cr_s : S.cr_u
        @inbounds for ii in eachindex(S.mols)
            ii == own && continue                       # m's own candidates were rebuilt above
            for j in 1:K
                target[j, ii] += S.cr_tmp[(ii - 1) * K + j]
            end
        end
    end
    return nothing
end

"Candidate structure-factor storage for R (sizes and backend follow R; contents rebuilt by the callers)."
function _slm_bind_recip!(S::SolventLocalMoves, R)
    KM = length(S.kinds) * length(S.mols)
    if size(S.crS, 1) != n_kvectors(R)
        S.crS = zeros(ComplexF64, n_kvectors(R), KM)
        S.crSd = S.crS; S.cr_tmpd = S.cr_tmp; S.crSdr = _realview(S.crS)
    end
    if R.Sd === R.S                                     # :cpu, :double: the host arrays themselves
        S.crSd === S.crS || (S.crSd = S.crS; S.cr_tmpd = S.cr_tmp; S.crSdr = _realview(S.crS))
        size(S.crSdr, 1) == 2 * size(S.crS, 1) || (S.crSdr = _realview(S.crS))
    elseif S.crSd === S.crS || size(S.crSd) != size(S.crS) || eltype(S.crSd) != eltype(R.Sd) || typeof(S.crSd) != typeof(R.Sd)
        C = eltype(R.Sd); F = real(C)
        S.crSd = backend_array(R.backend, convert(Matrix{C}, S.crS))
        S.cr_tmpd = backend_array(R.backend, zeros(F, length(S.cr_tmp)))
        S.crSdr = _realview(S.crSd)
        S.cr_stage = zeros(F, length(S.cr_tmp)); S.cr_stage_up = zeros(C, size(S.crS, 1), length(S.kinds))
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
            x = S.cu[j, ii] + S.cr_u[j, ii] + λ * (S.cs[j, ii] + S.cr_s[j, ii]) - φ + S.cin[j, ii]   # + ΔU_intra (S/D)
            r = S.ν * exp(min(-0.5 * β * x, MOBILITY_LOG_MAX))
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
    kd = S.kinds[j]
    if kd == LM_T || kd == LM_S                         # inverse: −δ (same site for LM_S)
        @inbounds for d in 1:3; S.dT[d, j, ii] = -S.dT[d, j, ii]; end
    elseif kd == LM_R                                   # inverse: conjugate quaternion
        @inbounds for d in 2:4; S.dQ[d, j, ii] = -S.dQ[d, j, ii]; end
    else                                                # LM_D inverse: −φ (same bond)
        S.dT[1, j, ii] = -S.dT[1, j, ii]
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
    cu = S.cu .+ S.cr_u; cs = S.cs .+ S.cr_s; ci = copy(S.cin)
    slm_rebuild_all!(S, sys, ff)
    return max(maximum(abs.(cu .- (S.cu .+ S.cr_u))), maximum(abs.(cs .- (S.cs .+ S.cr_s))), maximum(abs.(ci .- S.cin)))
end

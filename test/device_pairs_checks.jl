# Shared checks of the device pair kernel (DevicePairs.jl, DESIGN §15) against pair_energy_virial; included by
# test_device_pairs.jl (KernelAbstractions CPU backend) and test_device_pairs_cuda.jl (CUDA).
#
# Bounds (Higham, Accuracy and Stability of Numerical Algorithms, 2002, §3.1–3.4; γ_n = nu/(1 − nu)):
#  * Both codes sum the same site-pair terms (LJ, LJ shift, Coulomb; virial separately) in different orders, and evaluate
#    each term from r² with at most K_u = 24 (energy) / K_w = 32 (virial) roundings (counted: LJ 20 incl. the stored σ², 4ϵ;
#    RF 10; Wolf/Ewald ≤ 16 incl. erfc/erfcf ≤ 4 ulp, CUDA math API; the virial adds R·d, /r², ·fr: 7). Each result is
#    within γ_{n+K} Σ M of the exact value, M = the term evaluated with absolute values, n = the additions of that code
#    (CPU: 3 per site pair in sequence; kernel: 3 per site pair per work-item + 5 reduction levels).
#  * :double — r² is computed by the same Float64 expression in both codes (identical), so |Δ| ≤ (γ⁽⁶⁴⁾_{n_cpu+K} +
#    γ⁽⁶⁴⁾_{n_dev+K}) Σ M.
#  * :mixed — kernel arithmetic in Float32 (γ⁽³²⁾), plus the Float32 coordinates: the relative site vector d = R + o_j − o_i
#    from Float32 R, o_j, o_i with two additions, |δd_a| ≤ γ⁽³²⁾_3 (|R_a| + |o_j,a| + |o_i,a|), r² with γ⁽³²⁾_3; propagated
#    to first order through u(r²) (every pair term depends on d only through r²): |δu| ≤ |fr| / (2r²) · δr², fr = −r du/dr.
#    Site pairs whose cut-off or cap test can flip within δr² are ambiguous; tasks containing one are counted and skipped.
#    The :mixed virial is reported (max relative deviation), not asserted: its coordinate propagation needs d(fr)/dr.
using Test, Random
using MolSim
const MCd = MolSim.MC

γd(n, u) = n * u / (1 - n * u)
const U64 = eps(Float64) / 2
const U32 = Float64(eps(Float32)) / 2

# per-site-pair magnitudes (energy M, fr magnitude) from the Float64 state, exactly the terms of pair_energy_virial
function _site_pair_mag(ff, tyn, qn, tym, qm, r2)
    Mu = 0.0; Mf = 0.0
    if tyn > 0 && tym > 0
        s6 = (ff.σ_mix[tyn, tym]^2 / r2)^3; ϵ = ff.ϵ_mix[tyn, tym]
        Mu += 4ϵ * (s6 * s6 + s6) + (ff.shift_lj ? abs(ff.u_rc_mix[tyn, tym]) : 0.0)
        Mf += 24ϵ * (2s6 * s6 + s6)
    end
    if qn != 0.0 && qm != 0.0 && ff.coulomb !== :none
        r = sqrt(r2)
        if ff.coulomb === :rf
            A = abs(ff.rf.ke * qn * qm / ff.rf.ε_r)
            Mu += A * (1 / r + ff.rf.k_rf * r2 + ff.rf.c_rf); Mf += A * (1 / r + 2 * ff.rf.k_rf * r2)
        else
            wp = ff.wolf; A = abs(wp.ke * qn * qm); e = MolSim.MC.erfc(wp.α * r)
            Mf += A * (e / r + (2wp.α / sqrt(π)) * exp(-wp.α^2 * r2))
            if ff.coulomb === :wolf
                erc = MolSim.MC.erfc(wp.α * wp.rc)
                sl = erc / wp.rc^2 + (2wp.α / sqrt(π)) * exp(-wp.α^2 * wp.rc^2) / wp.rc
                Mu += A * (e / r + erc / wp.rc + (wp.model === :dsf ? sl * abs(r - wp.rc) : 0.0))
            else
                Mu += A * e / r
            end
        end
    end
    return Mu, Mf
end

"Bounds (bu, bw, ambiguous) for pair (i at its current pose in sys, j) at the given precision (see the header)."
function pair_bounds(sys, ff, i, j, precision)
    ti = sys.templates[sys.tmpl[i]]; tj = sys.templates[sys.tmpl[j]]
    L = sys.L; si = sys.site_start[i]; sj = sys.site_start[j]
    R = [MCd._min_image(sys.com[d, j] - sys.com[d, i], L) for d in 1:3]
    molcut = ff.cutoff === :molecule
    molcut && sum(abs2, R) >= ff.rc2 && return 0.0, 0.0, false
    ΣMu = 0.0; ΣMw = 0.0; prop = 0.0; amb = false; np = 0
    for a in 1:ti.n_sites, b in 1:tj.n_sites
        dv = R .+ sys.site_off[:, sj + b] .- sys.site_off[:, si + a]; r2 = sum(abs2, dv)
        δr2 = 0.0
        if precision === :mixed
            δd = [γd(3, U32) * (abs(R[k]) + abs(sys.site_off[k, sj + b]) + abs(sys.site_off[k, si + a])) for k in 1:3]
            δr2 = sum(2 .* abs.(dv) .* δd) + γd(3, U32) * r2
            (!molcut && abs(r2 - ff.rc2) <= δr2 + U32 * ff.rc2) && (amb = true)
            tyn = ti.site_type[a]; tym = tj.site_type[b]
            (ff.cap_overlaps && tyn > 0 && tym > 0 && abs(r2 - ff.rcap2_mix[tyn, tym]) <= δr2 + U32 * ff.rcap2_mix[tyn, tym]) && (amb = true)
        end
        (molcut || r2 < ff.rc2) || continue
        np += 1
        Mu, Mf = _site_pair_mag(ff, ti.site_type[a], ti.site_q[a], tj.site_type[b], tj.site_q[b], r2)
        ΣMu += Mu; ΣMw += sum(abs.(R .* dv)) / r2 * Mf
        prop += Mf / (2r2) * δr2
    end
    ndev = 3 * cld(ti.n_sites * tj.n_sites, 32) + 5; ncpu = 3np
    if precision === :double
        bu = (γd(ncpu + 24, U64) + γd(ndev + 24, U64)) * ΣMu
        bw = (γd(ncpu + 32, U64) + γd(ndev + 32, U64)) * ΣMw
    else
        bu = γd(ncpu + 24, U64) * ΣMu + (γd(ndev + 24, U32) + γd(5, U64)) * ΣMu + prop
        bw = NaN
    end
    return bu, bw, amb
end

# save / restore molecule i's pose bit-exactly
_pose(sys, i) = (copy(sys.com[:, i]), copy(sys.quat[:, i]), copy(sys.body[i]), sys.ext[i],
                 copy(sys.site_off[:, sys.site_start[i] .+ (1:sys.templates[sys.tmpl[i]].n_sites)]))
function _restore!(sys, i, P)
    sys.com[:, i] .= P[1]; sys.quat[:, i] .= P[2]; sys.body[i] = copy(P[3]); sys.ext[i] = P[4]
    sys.site_off[:, sys.site_start[i] .+ (1:size(P[5], 2))] .= P[5]
end

"""
Poses: committed poses, random translation+rotation candidates, flexible body changes (set_conformation!), and one
forced overlap; compare dp_rows! and dp_tasks! with pair_energy_virial; returns a summary NamedTuple.
"""
function check_device_pairs(sys, ff, backend, precision; sel, seed=1, label="")
    N = sys.n_mol
    D = MCd.DevicePairs(sys, ff; backend=backend, precision=precision, max_poses=4 * length(sel) + 1)
    rng = Xoshiro(seed)
    refs = Vector{Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},Vector{Bool}}}()
    slot = 0
    function stage_and_ref!(i)
        slot += 1
        MCd.dp_stage_pose!(D, slot, sys, i)
        u = zeros(N); w = zeros(N); bu = zeros(N); bw = zeros(N); amb = falses(N)
        for j in 1:N
            j == i && continue
            u[j], w[j] = MCd.pair_energy_virial(sys, ff, i, j)
            bu[j], bw[j], amb[j] = pair_bounds(sys, ff, i, j, precision)
        end
        push!(refs, (u, w, bu, bw, amb))
    end
    for i in sel
        P = _pose(sys, i)
        stage_and_ref!(i)                                                   # committed pose
        MCd.set_molecule!(sys, i, sys.com[:, i] .+ 0.6 .* (2 .* rand(rng, 3) .- 1), MCd.uniform_random_quaternion(rng))
        stage_and_ref!(i)                                                   # rigid-body candidate
        if MCd.isflexible(sys.templates[sys.tmpl[i]])
            B = MCd.body_sites(sys, i) .+ 0.08 .* (2 .* rand(rng, 3, size(sys.body[i], 2)) .- 1)
            c = copy(sys.com[:, i]); MCd.set_conformation!(sys, i, B); MCd.set_molecule!(sys, i, c, MCd.uniform_random_quaternion(rng))
            stage_and_ref!(i)                                               # internal (conformation) candidate
        end
        _restore!(sys, i, P)
    end
    i, j = sel[1], sel[2]                                                    # forced overlap with molecule j
    P = _pose(sys, i)
    MCd.set_molecule!(sys, i, sys.com[:, j] .+ [0.05, 0.0, 0.0], sys.quat[:, j]); stage_and_ref!(i)
    _restore!(sys, i, P)
    np = slot
    MCd.dp_upload_poses!(D, np)
    U, W = MCd.dp_rows!(D, np)
    nok = 0; nbad = 0; namb = 0; ncap = 0; worst = 0.0; wdev = 0.0
    for p in 1:np, jj in 1:N
        u, w, bu, bw, amb = refs[p]
        x = U[(p - 1) * N + jj]; y = W[(p - 1) * N + jj]
        if amb[jj]; namb += 1; continue; end
        if u[jj] == ff.cap_min_u || any(==(u[jj]), ff.ucap_mix) && w[jj] == 0.0 && u[jj] != 0.0
            ncap += 1
            (x == u[jj] && y == 0.0) ? (nok += 1) : (nbad += 1)
            continue
        end
        ok = abs(x - u[jj]) <= bu[jj] && (precision === :mixed || abs(y - w[jj]) <= bw[jj])
        ok ? (nok += 1) : (nbad += 1)
        bu[jj] > 0 && (worst = max(worst, abs(x - u[jj]) / bu[jj]))
        precision === :mixed && abs(w[jj]) > 0 && (wdev = max(wdev, abs(y - w[jj]) / abs(w[jj])))
    end
    # explicit tasks reproduce the row entries bit-for-bit (same kernel, same reduction)
    ps = [rand(rng, 1:np) for _ in 1:200]; js = [rand(rng, 1:N) for _ in 1:200]
    Ur = copy(U); Wr = copy(W)
    Ut, Wt = MCd.dp_tasks!(D, ps, js)
    same = all(Ut[t] == Ur[(ps[t] - 1) * N + js[t]] && Wt[t] == Wr[(ps[t] - 1) * N + js[t]] for t in eachindex(ps))
    # a committed move of another molecule: dp_sync_molecule! keeps the device state current
    k = sel[end]; P = _pose(sys, k)
    MCd.set_molecule!(sys, k, sys.com[:, k] .+ [0.3, -0.2, 0.1], MCd.uniform_random_quaternion(rng)); MCd.dp_sync_molecule!(D, sys, k)
    MCd.dp_stage_pose!(D, 1, sys, sel[1]); MCd.dp_upload_poses!(D, 1)
    U1, _ = MCd.dp_rows!(D, 1)
    u1, _ = MCd.pair_energy_virial(sys, ff, sel[1], k); b1, _, a1 = pair_bounds(sys, ff, sel[1], k, precision)
    sync_ok = a1 || abs(U1[k] - u1) <= b1
    _restore!(sys, k, P); MCd.dp_sync_molecule!(D, sys, k)
    return (; label, ntask=np * N, nok, nbad, namb, ncap, worst, wdev, same, sync_ok)
end

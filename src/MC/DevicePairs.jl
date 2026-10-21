# ============================================================================
# Device pair kernel (DESIGN §15): batches of (pose, molecule) pair energies and virials
# ============================================================================
# One task = (pose p, molecule j). A pose is a molecule i at its committed pose or at a candidate pose (COM, site offsets,
# extent); the task result is exactly the term of `pair_energy_virial(sys, ff, i, j)` with i at pose p: minimum image on
# the COM, then the offsets; site or molecule cut-off; LJ (+ shift); RF, Wolf/DSF or Ewald real space; the overlap cap is
# the FIRST overlapping site pair in the loop order of the lower-index molecule (`_canonical_ucap`; minimum linear index
# n·n_j + m for i < j, m·n_i + n for i > j, over the work-group), capped pairs carry no virial; then min(u, cap_min_u). A work-group of DP_GROUP work-items per task strides over
# the n_i·n_j site pairs and reduces in local memory in a fixed order (same result on every call). KernelAbstractions: the
# same kernel runs on the CPU (test oracle against pair_energy_virial) and on CUDA (MolSimCUDAExt: `device_backend`).
#
# Precision: :double — Float64 throughout; :mixed (GROMACS convention, DESIGN §15.3) — the COM minimum image in Float64,
# then Float32 relative coordinates, site-pair terms and per-work-item partial sums, Float64 work-group reduction.

using KernelAbstractions
const KA = KernelAbstractions

"KernelAbstractions backend for a device symbol (:cpu here; :cuda from MolSimCUDAExt)."
device_backend(::Val{:cpu}) = KA.CPU()
device_backend(::Val{B}) where {B} = error("device backend :$B is not loaded (for :cuda: `using CUDA`)")
device_backend(b::Symbol) = device_backend(Val(b))

const DP_GROUP = 32

# isbits scalars of the force field for the kernel
struct DPParams{T}
    L::Float64
    rc::Float64
    rc2::Float64        # molecule cut-off test on the Float64 COM distance
    rc2T::T             # site cut-off test on the site distance
    molcut::Bool
    coul::Int32         # 0 none, 1 RF, 2 Wolf/DSF, 3 Ewald real space
    ke_rf::T            # ke/ε_r
    k_rf::T
    c_rf::T
    wke::T              # Wolf/Ewald ke
    α::T
    wshift::T           # erfc(α rc)/rc
    dsf::Bool
    dsf_slope::T
    rcT::T
    cap::Bool
    cap_min_u::Float64
    shift::Bool
    nt::Int32
end

mutable struct DevicePairs{T}
    backend::Symbol
    kab::Any
    precision::Symbol
    N::Int
    maxs::Int
    prm::DPParams{T}
    # committed state
    com::Any; off::Any; sty::Any; sq::Any; mstart::Any; mn::Any; ext::Any
    # force-field tables (n_types², column-major (a, b) ↦ (a − 1)·n_t + b)
    σ2::Any; e4::Any; e24::Any; urc::Any; rcap2::Any; ucap::Any
    # pose bank (P poses, maxs site slots each) and its host staging
    P::Int
    pcom::Any; poff::Any; pmol::Any; pext::Any
    hcom::Matrix{Float64}; hoff::Matrix{T}; hmol::Vector{Int32}; hext::Vector{Float64}
    # explicit task lists and outputs
    tp::Any; tj::Any; U::Any; W::Any
    htp::Vector{Int32}; htj::Vector{Int32}; hU::Vector{Float64}; hW::Vector{Float64}
    # staging for one molecule's offsets
    hstage::Matrix{T}
end

_dp_array(b::Symbol, A::AbstractArray) = b === :cpu ? copy(A) : backend_array(b, A)

"""
    DevicePairs(sys, ff; backend=:cpu, precision=:mixed, max_poses=64, max_tasks=max_poses·N)

Device copy of the committed pair-energy inputs of `sys` (offsets, COMs, site types and charges, extents) and the
force-field tables, with a bank of `max_poses` poses and room for `max_tasks` explicit tasks (DESIGN §15.3).
"""
function DevicePairs(sys::RigidSystem, ff::SiteForceField; backend::Symbol=:cpu, precision::Symbol=:mixed,
                     max_poses::Int=64, max_tasks::Int=max_poses * sys.n_mol)
    precision in (:double, :mixed) || error("DevicePairs: precision must be :double or :mixed")
    ff.coulomb in (:none, :rf, :wolf, :ewald) || error("DevicePairs: coulomb $(ff.coulomb) not supported")
    T = precision === :double ? Float64 : Float32
    N = sys.n_mol; Ns = sys.n_sites_total
    maxs = maximum(sys.templates[sys.tmpl[i]].n_sites for i in 1:N)
    sty = zeros(Int32, Ns); sq = zeros(T, Ns); mstart = zeros(Int32, N); mn = zeros(Int32, N)
    for i in 1:N
        t = sys.templates[sys.tmpl[i]]; s0 = sys.site_start[i]
        mstart[i] = s0; mn[i] = t.n_sites
        for n in 1:t.n_sites
            sty[s0 + n] = t.site_type[n]; sq[s0 + n] = t.site_q[n]
        end
    end
    nt = ff.n_types
    flat(M) = [M[a, b] for a in 1:nt for b in 1:nt]        # index (a − 1)·nt + b
    σ2 = T.(flat(ff.σ_mix .^ 2)); e4 = T.(4 .* flat(ff.ϵ_mix)); e24 = T.(24 .* flat(ff.ϵ_mix)); urc = T.(flat(ff.u_rc_mix))
    rcap2 = T.(flat(ff.rcap2_mix)); ucap = Float64.(flat(ff.ucap_mix))
    rf = ff.rf; wp = ff.wolf
    coul = ff.coulomb === :none ? Int32(0) : ff.coulomb === :rf ? Int32(1) : ff.coulomb === :wolf ? Int32(2) : Int32(3)
    erfc_rc = erfc(wp.α * wp.rc)
    slope = erfc_rc / (wp.rc^2) + (2.0 * wp.α / SQRT_PI) * exp(-wp.α^2 * wp.rc^2) / wp.rc
    prm = DPParams{T}(sys.L, ff.rc, ff.rc2, T(ff.rc2), ff.cutoff === :molecule, coul, T(rf.ke / rf.ε_r), T(rf.k_rf), T(rf.c_rf),
                      T(wp.ke), T(wp.α), T(erfc_rc / wp.rc), wp.model === :dsf, T(slope), T(wp.rc),
                      ff.cap_overlaps, ff.cap_min_u, ff.shift_lj, Int32(nt))
    b = backend; d(A) = _dp_array(b, A)
    D = DevicePairs{T}(b, device_backend(b), precision, N, maxs, prm,
                       d(copy(sys.com)), d(T.(sys.site_off)), d(sty), d(sq), d(mstart), d(mn), d(copy(sys.ext)),
                       d(σ2), d(e4), d(e24), d(urc), d(rcap2), d(ucap),
                       max_poses, d(zeros(3, max_poses)), d(zeros(T, 3, max_poses * maxs)), d(zeros(Int32, max_poses)),
                       d(zeros(max_poses)),
                       zeros(3, max_poses), zeros(T, 3, max_poses * maxs), zeros(Int32, max_poses), zeros(max_poses),
                       d(zeros(Int32, max_tasks)), d(zeros(Int32, max_tasks)), d(zeros(max_tasks)), d(zeros(max_tasks)),
                       zeros(Int32, max_tasks), zeros(Int32, max_tasks), zeros(max_tasks), zeros(max_tasks),
                       zeros(T, 3, maxs))
    return D
end

"Re-upload the whole committed state (after a rebuild, a volume change or a box rescale)."
function dp_sync!(D::DevicePairs{T}, sys::RigidSystem) where {T}
    sys.L == D.prm.L || (D.prm = _dp_with_L(D.prm, sys.L))
    copyto!(D.com, sys.com); copyto!(D.off, T.(sys.site_off)); copyto!(D.ext, sys.ext)
    return D
end
_dp_with_L(p::DPParams{T}, L) where {T} = DPParams{T}(L, (getfield(p, f) for f in fieldnames(DPParams)[2:end])...)

"Upload molecule i's committed pose (COM, offsets, extent) after a committed move."
function dp_sync_molecule!(D::DevicePairs{T}, sys::RigidSystem, i::Int) where {T}
    n = sys.templates[sys.tmpl[i]].n_sites; s0 = sys.site_start[i]
    st = D.hstage
    @inbounds for a in 1:n, d in 1:3
        st[d, a] = T(sys.site_off[d, s0 + a])
    end
    copyto!(D.off, 3 * s0 + 1, st, 1, 3 * n)
    copyto!(D.com, 3 * (i - 1) + 1, sys.com, 3 * (i - 1) + 1, 3)
    copyto!(D.ext, i, sys.ext, i, 1)
    return D
end

"Stage molecule i's CURRENT pose in sys (committed or a candidate placed by the caller) as pose `slot` of the bank."
function dp_stage_pose!(D::DevicePairs{T}, slot::Int, sys::RigidSystem, i::Int) where {T}
    1 <= slot <= D.P || error("dp_stage_pose!: slot $slot outside the bank (P = $(D.P))")
    n = sys.templates[sys.tmpl[i]].n_sites; s0 = sys.site_start[i]; o = (slot - 1) * D.maxs
    @inbounds begin
        for d in 1:3
            D.hcom[d, slot] = sys.com[d, i]
        end
        for a in 1:n, d in 1:3
            D.hoff[d, o + a] = T(sys.site_off[d, s0 + a])
        end
        D.hmol[slot] = i; D.hext[slot] = sys.ext[i]
    end
    return D
end
"Upload poses 1…np of the host staging to the device bank."
function dp_upload_poses!(D::DevicePairs, np::Int)
    np == 0 && return D
    copyto!(D.pcom, 1, D.hcom, 1, 3 * np); copyto!(D.poff, 1, D.hoff, 1, 3 * np * D.maxs)
    copyto!(D.pmol, 1, D.hmol, 1, np); copyto!(D.pext, 1, D.hext, 1, np)
    return D
end

@kernel function _dp_kernel!(U, W, @Const(tp), @Const(tj), implicit::Int32, Nm::Int32,
                             @Const(com), @Const(off), @Const(sty), @Const(sq), @Const(mstart), @Const(mn), @Const(ext),
                             @Const(pcom), @Const(poff), @Const(pmol), @Const(pext), maxs::Int32,
                             @Const(σ2), @Const(e4), @Const(e24), @Const(urc), @Const(rcap2), @Const(ucap), prm)
    g = @index(Group, Linear)
    l = @index(Local, Linear)
    su = @localmem Float64 (DP_GROUP,)
    sw = @localmem Float64 (DP_GROUP,)
    smi = @localmem Int32 (DP_GROUP,)
    au, aw, mi = _dp_partial(g, l, tp, tj, implicit, Nm, com, off, sty, sq, mstart, mn, ext, pcom, poff, pmol, pext, maxs,
                             σ2, e4, e24, urc, rcap2, prm)
    @inbounds su[l] = au; @inbounds sw[l] = aw; @inbounds smi[l] = mi
    @synchronize
    _dp_reduce_step!(su, sw, smi, @index(Local, Linear), 16)
    @synchronize
    _dp_reduce_step!(su, sw, smi, @index(Local, Linear), 8)
    @synchronize
    _dp_reduce_step!(su, sw, smi, @index(Local, Linear), 4)
    @synchronize
    _dp_reduce_step!(su, sw, smi, @index(Local, Linear), 2)
    @synchronize
    _dp_reduce_step!(su, sw, smi, @index(Local, Linear), 1)
    @synchronize
    if @index(Local, Linear) == 1
        gg = @index(Group, Linear)
        u, w = @inbounds _dp_final(gg, su[1], sw[1], smi[1], tp, tj, implicit, Nm, sty, mstart, mn, pmol, ucap, prm)
        @inbounds U[gg] = u; @inbounds W[gg] = w
    end
end
# one step of the fixed-order tree reduction (work-items 1…s add work-item l + s)
@inline function _dp_reduce_step!(su, sw, smi, l, s)
    @inbounds if l <= s
        su[l] += su[l + s]; sw[l] += sw[l + s]; smi[l] = min(smi[l], smi[l + s])
    end
    return nothing
end

@inline function _dp_task(g, tp, tj, implicit, Nm)
    if implicit != 0
        return (g - 1) ÷ Nm + 1, (g - 1) % Nm + 1          # rows: task g = (pose p, molecule j), j fastest
    else
        return Int(@inbounds tp[g]), Int(@inbounds tj[g])
    end
end
@inline _dp_mi(d::Float64, L::Float64) = d - L * round(d / L)

@inline function _dp_partial(g, l, tp, tj, implicit, Nm, com, off, sty, sq, mstart, mn, ext, pcom, poff, pmol, pext, maxs,
                             σ2, e4, e24, urc, rcap2, prm::DPParams{T}) where {T}
    au = zero(T); aw = zero(T); mi = typemax(Int32)
    p, j = _dp_task(g, tp, tj, implicit, Nm)
    @inbounds i = Int(pmol[p])
    if j != i
        @inbounds begin
            Rx = _dp_mi(com[1, j] - pcom[1, p], prm.L)
            Ry = _dp_mi(com[2, j] - pcom[2, p], prm.L)
            Rz = _dp_mi(com[3, j] - pcom[3, p], prm.L)
        end
        R2 = Rx * Rx + Ry * Ry + Rz * Rz
        rmax = prm.rc + @inbounds(ext[j]) + @inbounds(pext[p])
        inr = prm.molcut ? R2 < prm.rc2 : R2 < rmax * rmax
        if inr
            si0 = Int(@inbounds mstart[i]); sj0 = Int(@inbounds mstart[j])
            ni = Int(@inbounds mn[i]); nj = Int(@inbounds mn[j])
            po = (p - 1) * Int(maxs)
            Rxt = T(Rx); Ryt = T(Ry); Rzt = T(Rz)
            # site cut-off, large molecules in a small box: wrap each site separation (as pair_energy_virial)
            e = @inbounds(ext[j]) + @inbounds(pext[p]); hL = 0.5 * prm.L
            wrap = !prm.molcut && prm.rc < hL && (abs(Rx) + e > hL || abs(Ry) + e > hL || abs(Rz) + e > hL)
            LT = T(prm.L)
            nt = Int(prm.nt)
            q = l - 1
            while q < ni * nj
                n = q ÷ nj; m = q - n * nj
                @inbounds begin
                    dx = Rxt + off[1, sj0 + m + 1] - poff[1, po + n + 1]
                    dy = Ryt + off[2, sj0 + m + 1] - poff[2, po + n + 1]
                    dz = Rzt + off[3, sj0 + m + 1] - poff[3, po + n + 1]
                end
                Rwx = Rxt; Rwy = Ryt; Rwz = Rzt
                if wrap
                    kx = LT * round(dx / LT); ky = LT * round(dy / LT); kz = LT * round(dz / LT)
                    dx -= kx; dy -= ky; dz -= kz; Rwx -= kx; Rwy -= ky; Rwz -= kz
                end
                r2 = dx * dx + dy * dy + dz * dz
                if prm.molcut || r2 < prm.rc2T
                    tyn = Int(@inbounds sty[si0 + n + 1]); tym = Int(@inbounds sty[sj0 + m + 1])
                    fr = zero(T)
                    if tyn > 0 && tym > 0
                        k = (tyn - 1) * nt + tym
                        if prm.cap && r2 < @inbounds(rcap2[k])
                            mi = min(mi, Int32(i < j ? q : m * ni + n))   # lower-index molecule's order (_canonical_ucap)
                        end
                        s6 = (@inbounds(σ2[k]) / r2)^3
                        au += @inbounds(e4[k]) * (s6 * s6 - s6)
                        prm.shift && (au -= @inbounds(urc[k]))
                        fr += @inbounds(e24[k]) * (2 * s6 * s6 - s6)
                    end
                    qn = @inbounds sq[si0 + n + 1]; qm = @inbounds sq[sj0 + m + 1]
                    if prm.coul != 0 && qn != 0 && qm != 0
                        r = sqrt(r2); qq = qn * qm
                        if prm.coul == 1
                            au += prm.ke_rf * qq * (1 / r + prm.k_rf * r2 - prm.c_rf)
                            fr += prm.ke_rf * qq * (1 / r - 2 * prm.k_rf * r2)
                        else
                            e = erfc(prm.α * r)
                            if prm.coul == 2
                                uw = e / r - prm.wshift
                                prm.dsf && (uw += prm.dsf_slope * (r - prm.rcT))
                                au += prm.wke * qq * uw
                            else
                                au += prm.wke * qq * e / r
                            end
                            fr += prm.wke * qq * (e / r + (2 * prm.α / T(SQRT_PI)) * exp(-prm.α * prm.α * r2))
                        end
                    end
                    aw += (Rwx * dx + Rwy * dy + Rwz * dz) / r2 * fr
                end
                q += DP_GROUP
            end
        end
    end
    return Float64(au), Float64(aw), mi
end

@inline function _dp_final(g, u, w, mi, tp, tj, implicit, Nm, sty, mstart, mn, pmol, ucap, prm)
    p, j = _dp_task(g, tp, tj, implicit, Nm)
    @inbounds i = Int(pmol[p])
    j == i && return 0.0, 0.0
    if mi != typemax(Int32)                        # first overlapping site pair in the lower-index molecule's order
        if i < j
            nj = Int(@inbounds mn[j]); n = Int(mi) ÷ nj; m = Int(mi) - n * nj
        else
            ni = Int(@inbounds mn[i]); m = Int(mi) ÷ ni; n = Int(mi) - m * ni
        end
        tyn = Int(@inbounds sty[Int(@inbounds mstart[i]) + n + 1]); tym = Int(@inbounds sty[Int(@inbounds mstart[j]) + m + 1])
        return @inbounds(ucap[(tyn - 1) * Int(prm.nt) + tym]), 0.0
    end
    prm.cap && u > prm.cap_min_u && return prm.cap_min_u, 0.0
    return u, w
end

function _dp_launch!(D::DevicePairs, ntask::Int, implicit::Bool)
    ntask == 0 && return D
    k = _dp_kernel!(D.kab, DP_GROUP)
    k(D.U, D.W, D.tp, D.tj, Int32(implicit), Int32(D.N), D.com, D.off, D.sty, D.sq, D.mstart, D.mn, D.ext,
      D.pcom, D.poff, D.pmol, D.pext, Int32(D.maxs), D.σ2, D.e4, D.e24, D.urc, D.rcap2, D.ucap, D.prm;
      ndrange=DP_GROUP * ntask)
    KA.synchronize(D.kab)
    copyto!(D.hU, 1, D.U, 1, ntask); copyto!(D.hW, 1, D.W, 1, ntask)
    return D
end

"""
    dp_rows!(D, np) -> (U, W)

Rows of the first `np` staged-and-uploaded poses against every committed molecule: U[(p − 1)·N + j], W[...] = the pair
energy and virial of pose p with molecule j (0 for j = the pose's own molecule). Views into host buffers (valid until the
next call).
"""
function dp_rows!(D::DevicePairs, np::Int)
    np * D.N <= length(D.hU) || error("dp_rows!: $np poses × N exceed max_tasks")
    _dp_launch!(D, np * D.N, true)
    return view(D.hU, 1:np * D.N), view(D.hW, 1:np * D.N)
end

"""
    dp_tasks!(D, ps, js) -> (U, W)

Explicit tasks (pose ps[t], molecule js[t]), t = 1…length(ps), on the uploaded pose bank.
"""
function dp_tasks!(D::DevicePairs, ps::AbstractVector{<:Integer}, js::AbstractVector{<:Integer})
    nt = length(ps); nt == length(js) || error("dp_tasks!: ps and js differ in length")
    nt <= length(D.htp) || error("dp_tasks!: $nt tasks exceed max_tasks")
    @inbounds for t in 1:nt
        D.htp[t] = ps[t]; D.htj[t] = js[t]
    end
    copyto!(D.tp, 1, D.htp, 1, nt); copyto!(D.tj, 1, D.htj, 1, nt)
    _dp_launch!(D, nt, false)
    return view(D.hU, 1:nt), view(D.hW, 1:nt)
end

# --- eKMC hook-up (DESIGN §15 G2) ---

"""
    attach_device_pairs!(sys, ff; backend=:cpu, precision=:double, max_poses=64)

Route the real-space + LJ rows of `ff` (`eval_row!`: eKMC events, candidate rebuilds, MC trials) through the device pair
kernel: the device keeps the committed poses (synced in `molecule_committed!` and at eKMC rebuilds) and a row is one launch
of N tasks for the moved molecule's current pose. Other force fields (e.g. an LJ-only copy) stay on the host. `precision =
:double` reproduces `pair_energy_virial` to rounding (the debug rebuild, on the host, is the check); `:mixed` follows the
GROMACS convention (DESIGN §15.3). NVT only (a volume change rescales every pose: `run_rigid_ekmc!` refuses Pext with it).
"""
function attach_device_pairs!(sys::RigidSystem, ff::SiteForceField; backend::Symbol=:cpu, precision::Symbol=:double,
                              max_poses::Int=64)
    D = DevicePairs(sys, ff; backend=backend, precision=precision, max_poses=max_poses)
    dp_sync!(D, sys)
    sys.dev = (D=D, ff=ff)
    return D
end
detach_device_pairs!(sys::RigidSystem) = (sys.dev = nothing; nothing)

"The DevicePairs of sys if it was attached for this force field, else nothing."
@inline _dev_for(sys, ff) = (d = sys.dev; d !== nothing && d.ff === ff ? d.D : nothing)

"row_u[j], row_w[j] of molecule k at its current pose with every committed j (row[k] = 0)."
function _dp_row!(row_u::Vector{Float64}, row_w::Vector{Float64}, D::DevicePairs, sys::RigidSystem, k::Int)
    sys.L == D.prm.L || dp_sync!(D, sys)
    dp_stage_pose!(D, 1, sys, k)
    dp_upload_poses!(D, 1)
    U, W = dp_rows!(D, 1)
    @inbounds for j in 1:sys.n_mol
        row_u[j] = U[j]; row_w[j] = W[j]
    end
    return nothing
end

"Upload staged poses a…b of the bank."
function dp_upload_pose_range!(D::DevicePairs, a::Int, b::Int)
    b < a && return D
    n = b - a + 1
    copyto!(D.pcom, 3 * (a - 1) + 1, D.hcom, 3 * (a - 1) + 1, 3 * n)
    copyto!(D.poff, 3 * (a - 1) * D.maxs + 1, D.hoff, 3 * (a - 1) * D.maxs + 1, 3 * n * D.maxs)
    copyto!(D.pmol, a, D.hmol, a, n); copyto!(D.pext, a, D.hext, a, n)
    return D
end

"Explicit tasks already written to D.htp / D.htj (1…nt)."
function _dp_tasks_staged!(D::DevicePairs, nt::Int)
    nt <= length(D.htp) || error("device tasks: $nt tasks exceed max_tasks $(length(D.htp))")
    copyto!(D.tp, 1, D.htp, 1, nt); copyto!(D.tj, 1, D.htj, 1, nt)
    _dp_launch!(D, nt, false)
    return view(D.hU, 1:nt)
end

"""
    slm_attach_device!(S, sys, ff)

Keep the K·M candidate poses of the solvent L-moves in the device pose bank (slots 2 … K·M + 1; slot 1 is `eval_row!`'s) so
that a mover's rebuild is one launch of K·N tasks and the column updates after an event are one launch over every
(candidate, moved molecule) pair within reach (the host rule of `_slm_column!`). Needs `attach_device_pairs!` with
max_poses ≥ K·M + 1.
"""
function slm_attach_device!(S::SolventLocalMoves, sys::RigidSystem, ff::SiteForceField)
    D = _dev_for(sys, ff)
    D === nothing && error("slm_attach_device!: attach_device_pairs!(sys, ff) first")
    need = length(S.kinds) * length(S.mols) + 1
    D.P >= need || error("slm_attach_device!: the pose bank has $(D.P) slots, need $need (attach_device_pairs!(…; max_poses=$need))")
    length(D.htp) >= length(S.kinds) * max(sys.n_mol, length(S.mols)) ||
        error("slm_attach_device!: max_tasks too small")
    S.dbase = 2
    slm_rebuild_all!(S, sys, ff)
    return S
end

function _slm_device_rows!(S::SolventLocalMoves, D::DevicePairs, ii::Int, N::Int)
    K = length(S.kinds)
    a = S.dbase + (ii - 1) * K
    dp_upload_pose_range!(D, a, a + K - 1)
    nt = 0
    @inbounds for j in 1:K
        S.act[j, ii] || continue
        for m in 1:N
            nt += 1; D.htp[nt] = a + j - 1; D.htj[nt] = m
        end
    end
    U = _dp_tasks_staged!(D, nt)
    t = 0
    @inbounds for j in 1:K
        S.act[j, ii] || continue
        u = 0.0
        for m in 1:N
            t += 1
            S.cp[j, m, ii] = U[t]
            m == S.frac || (u += U[t])
        end
        S.cu[j, ii] = u
        S.cs[j, ii] = S.frac > 0 ? S.cp[j, S.frac, ii] : 0.0
    end
    return nothing
end

function _slm_device_columns!(S::SolventLocalMoves, D::DevicePairs, sys, ff, m::Int, reach0::Float64)::Int
    K = length(S.kinds); L = sys.L
    nt = 0; nmov = 0
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
            r = reach0 + (ff.cutoff === :molecule ? S.cand_com[ii] : S.cand_site[ii])
            dx * dx + dy * dy + dz * dz >= r * r && continue
        end
        nmov += 1
        for j in 1:K
            S.act[j, ii] || continue
            nt += 1; D.htp[nt] = S.dbase + (ii - 1) * K + j - 1; D.htj[nt] = m
        end
    end
    nt == 0 && return 0
    U = _dp_tasks_staged!(D, nt)
    @inbounds for t in 1:nt
        p = Int(D.htp[t]) - S.dbase; ii = p ÷ K + 1; j = p - (ii - 1) * K + 1
        u = U[t]
        if m == S.frac
            S.cs[j, ii] += u - S.cp[j, m, ii]
        else
            S.cu[j, ii] += u - S.cp[j, m, ii]
        end
        S.cp[j, m, ii] = u
    end
    return nmov
end

# Large molecules in a small box (site cut-off, L/2 < rc + ext_i + ext_j): every site pair interacts through its own nearest
# image, as in GROMACS (one image per site pair since rc < L/2). Before the fix pair_energy used the image of the nearest COM
# for all site pairs of a molecule pair and missed site pairs that are within rc only through another image (100 LigParGen
# ibuprofen at 935 kg/m³, L = 33.2 Å, rc 10 Å: −2.59 kJ/mol against GROMACS 2024.3 Coulomb (SR)). Reference here: an
# independent brute-force sum over site positions with per-site minimum images; bound as in test_cells.jl (Higham §3.1).
using Test, Random
using MolSim
const MCb = MolSim.MC

γb(n) = (u = eps(Float64) / 2; n * u / (1 - n * u))
box_wraps(R, e, L) = any(abs.(R) .+ e .> L / 2)          # some site pair may need another image than the COM's

"""
Brute-force (u, w) of molecules i, j: EVERY site separation wrapped to its nearest image (no shortcut through the COM
image), built with the codes' arithmetic (R + o_j − o_i, then − L·round(·/L)) so r² is the same floating-point number; the
virial with the COM separation of each site pair's image.
"""
function brute_pair(sys, ff, i, j; device=false)
    ti = sys.templates[sys.tmpl[i]]; tj = sys.templates[sys.tmpl[j]]
    L = sys.L
    R = [MCb._min_image(sys.com[k, j] - sys.com[k, i], L) for k in 1:3]
    u = 0.0; w = 0.0; su = 0.0; sw = 0.0; n = 0
    for a in 1:ti.n_sites, b in 1:tj.n_sites
        pa = sys.site_start[i] + a; pb = sys.site_start[j] + b
        dc = R .+ sys.site_off[:, pb] .- sys.site_off[:, pa]
        sh = L .* round.(dc ./ L)
        d = dc .- sh
        r2 = sum(abs2, d); r2 < ff.rc2 || continue
        Rimg = R .- sh                                                  # COM separation of that image
        uu, fr = MCb._site_pair_uf(ff, ti.site_type[a], ti.site_q[a], tj.site_type[b], tj.site_q[b], r2)
        wv = sum(Rimg .* d) / r2 * fr
        # rounding scales with each term's expression evaluated in absolute values (Higham §3): LJ 4ϵ(s¹² + s⁶),
        # RF |ke qq/ε_r|(1/r + k_rf r² + c_rf), which near r_c is much larger than the term itself
        Mu = 0.0; Mf = 0.0; tyn = ti.site_type[a]; tym = tj.site_type[b]
        if tyn > 0 && tym > 0
            s6 = (ff.σ_mix[tyn, tym]^2 / r2)^3; Mu += 4ff.ϵ_mix[tyn, tym] * (s6^2 + s6); Mf += 24ff.ϵ_mix[tyn, tym] * (2s6^2 + s6)
        end
        if ti.site_q[a] != 0 && tj.site_q[b] != 0
            r = sqrt(r2); A = abs(ff.rf.ke * ti.site_q[a] * tj.site_q[b] / ff.rf.ε_r)
            Mu += A * (1 / r + ff.rf.k_rf * r2 + ff.rf.c_rf); Mf += A * (1 / r + 2ff.rf.k_rf * r2)
        end
        u += uu; w += wv; su += Mu; sw += sum(abs.(Rimg .* d)) / r2 * Mf; n += 3
    end
    # second code with the same terms in another order: ≤ 8 roundings per term from r² (both CPU paths); the device kernel
    # (device = true) instead ≤ 24 per term and its own order: 3 additions per site pair per work-item + 5 reduction levels
    # (device_pairs_checks.jl)
    nd = 3 * cld(ti.n_sites * tj.n_sites, 32) + 5
    device && return u, w, (γb(n + 8) + γb(nd + 24)) * su, (γb(n + 8) + γb(nd + 32)) * sw
    return u, w, 2γb(n + 8) * su, 2γb(n + 8) * sw
end

@testset "Site cut-off in a small box: per-site nearest images (pair energies, virials, site cells, device kernel)" begin
    mi = MCb.spce_solute_model(:ibuprofen_lpg; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=9.0,
                               cap_overlaps=false)
    ff = mi.ff; t = mi.templates[2]
    L = 22.0 / mi.σ_angstrom                                           # L/2 = 11 Å < rc + 2 ext ≈ 19 Å; rc = 9 Å < L/2
    sys = MCb.RigidSystem([t], fill(1, 8), L, mi.ϵ_kelvin / 400.0, Xoshiro(4))
    MCb.random_rigid_system!(sys, ff)
    N = sys.n_mol
    @test ff.rc < L / 2
    nwrap = 0; ok = true; nmiss = 0
    for i in 1:N-1, j in i+1:N
        R = [MCb._min_image(sys.com[k, j] - sys.com[k, i], L) for k in 1:3]
        box_wraps(R, sys.ext[i] + sys.ext[j], L) && (nwrap += 1)
        u, w = MCb.pair_energy_virial(sys, ff, i, j)
        ub, wb, bu, bw = brute_pair(sys, ff, i, j)
        (abs(u - ub) <= bu && abs(w - wb) <= bw && abs(MCb.pair_energy(sys, ff, i, j) - ub) <= bu) || (ok = false)
        # site pairs the COM image alone would miss (what the fix adds)
        ti = sys.templates[sys.tmpl[i]]
        for a in 1:ti.n_sites, b in 1:ti.n_sites
            dc = R .+ sys.site_off[:, sys.site_start[j] + b] .- sys.site_off[:, sys.site_start[i] + a]
            ds = MCb._min_image.(dc, L)
            (ds != dc && sum(abs2, ds) < ff.rc2) && (nmiss += 1)
        end
    end
    @test nwrap > 0 && nmiss > 0                 # the configuration exercises the per-site images
    @test ok
    # rows of the site cell list = all-pairs rows (within the reordering bound) on the same configuration
    c = MCb.attach_cells!(sys, ff)
    u0 = zeros(N); w0 = zeros(N); u1 = zeros(N); w1 = zeros(N)
    okc = true
    for k in 1:N
        sys.cells = nothing; MCb.eval_row!(u0, w0, sys, ff, k)
        sys.cells = c; MCb.eval_row!(u1, w1, sys, ff, k)
        for j in 1:N
            j == k && continue
            _, _, bu, bw = brute_pair(sys, ff, k, j)
            (abs(u1[j] - u0[j]) <= 2bu && abs(w1[j] - w0[j]) <= 2bw) || (okc = false)
        end
    end
    sys.cells = nothing
    @test okc
    # device kernel (KernelAbstractions CPU backend, :double): committed rows = brute force
    D = MCb.DevicePairs(sys, ff; backend=:cpu, precision=:double, max_poses=N)
    for k in 1:N; MCb.dp_stage_pose!(D, k, sys, k); end
    MCb.dp_upload_poses!(D, N)
    U, W = MCb.dp_rows!(D, N)
    okd = true
    for k in 1:N, j in 1:N
        j == k && continue
        ub, wb, bu, bw = brute_pair(sys, ff, k, j; device=true)
        (abs(U[(k - 1) * N + j] - ub) <= bu && abs(W[(k - 1) * N + j] - wb) <= bw) || (okd = false)
    end
    @test okd
end

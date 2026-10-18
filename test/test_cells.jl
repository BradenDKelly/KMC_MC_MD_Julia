# Cell lists for the real-space rows (MolCells.jl): molecular cells (molecule cut-off) and site cells (site cut-off) give the
# all-pairs rows; the incremental membership stays current through MC moves and eKMC events (including flexible molecules).
using Test, Random
using MolSim
const MCc = MolSim.MC

γc(n) = (u = eps(Float64) / 2; n * u / (1 - n * u))      # Higham (2002) Eq. 3.8

"Bound on |site-cell row − all-pairs row| for pair (k, j): both sums of the same site-pair terms (LJ, Coulomb, virial added
separately in `pair_energy_virial`) in different orders, each within γ_n Σ|terms| of the exact sum (Higham §3.1)."
function cells_row_bound(sys, ff, k, j)
    tk = sys.templates[sys.tmpl[k]]; tj = sys.templates[sys.tmpl[j]]
    L = sys.L; sk = sys.site_start[k]; sj = sys.site_start[j]
    R = [MCc._min_image(sys.com[d, j] - sys.com[d, k], L) for d in 1:3]
    su = 0.0; sw = 0.0; n = 0
    for a in 1:tk.n_sites, b in 1:tj.n_sites
        dv = R .+ sys.site_off[:, sj + b] .- sys.site_off[:, sk + a]; r2 = sum(abs2, dv)
        r2 < ff.rc2 || continue
        ulj, flj = MCc._site_pair_uf(ff, tk.site_type[a], 0.0, tj.site_type[b], 0.0, r2)
        uc, fc = MCc._site_pair_uf(ff, 0, tk.site_q[a], 0, tj.site_q[b], r2)
        su += abs(ulj) + abs(uc); sw += abs(sum(R .* dv) / r2) * (abs(flj) + abs(fc)); n += 3
    end
    return 2γc(n + 1) * su, 2γc(n + 1) * sw
end

function water_box(N, seed; coul=:rf, cutoff=:site, rc=10.0, equil=2)
    α = MCc.ewald_alpha_for(rc, 1e-5)
    m = MCc.spce_model(; α_per_angstrom=α, rc_angstrom=rc, coulomb=coul, epsilon_rf=Inf, cutoff=cutoff)
    L = cbrt(N / 0.0334) / m.σ_angstrom
    sys = MCc.RigidSystem([m.template], fill(1, N), L, m.ϵ_kelvin / 300.0, Xoshiro(seed))
    MCc.random_rigid_system!(sys, m.ff)
    st = MCc.RigidMCStats()
    for _ in 1:equil * N; MCc.mc_translate!(sys, m.ff, rand(sys.rng, 1:N), 0.1, st); end
    return m, sys
end

@testset "Cell lists: rows = all-pairs rows (molecular cells exactly; site cells within the reordering bound)" begin
    for cutoff in (:molecule, :site)
        m, sys = water_box(1500, 3; cutoff=cutoff)              # L ≈ 35.5 Å, rc = 10 Å: the cells really prune
        ff = m.ff; N = sys.n_mol
        c = MCc.attach_cells!(sys, ff)
        @test c isa (cutoff === :molecule ? MCc.MolCells : MCc.SiteCells)
        u0 = zeros(N); w0 = zeros(N); u1 = zeros(N); w1 = zeros(N)
        # an overlapping pair exercises the cap rule (first overlapping site pair, then the cap clamp)
        MCc.set_molecule!(sys, 2, sys.com[:, 1] .+ [0.05, 0.0, 0.0], sys.quat[:, 1]); MCc.cells_rebuild!(c, sys)
        for k in (1, 2, 7, 500, 1499)
            sys.cells = nothing; MCc.eval_row!(u0, w0, sys, ff, k)
            sys.cells = c; MCc.eval_row!(u1, w1, sys, ff, k)
            if cutoff === :molecule
                @test u1 == u0 && w1 == w0
            else
                ok = true
                @test u1[k] == 0.0 && w1[k] == 0.0
                for j in 1:N
                    j == k && continue
                    bu, bw = cells_row_bound(sys, ff, k, j)
                    (abs(u1[j] - u0[j]) <= bu && abs(w1[j] - w0[j]) <= bw) || (ok = false)
                end
                @test ok
            end
        end
        @test c.n_full == 0                                      # every query used the cells
        sys.cells = nothing; MCc.eval_row!(u0, w0, sys, ff, 1); sys.cells = c; MCc.eval_row!(u1, w1, sys, ff, 1)
        @test u0[2] == ff.ucap_mix[1, 1] && u1[2] == u0[2] && w1[2] == 0.0       # the capped pair
    end
end

@testset "Cell lists: membership current after MC moves and eKMC events (rigid water, flexible solute ladder)" begin
    m, sys = water_box(400, 5; cutoff=:site)
    ff = m.ff
    c = MCc.attach_cells!(sys, ff)
    st = MCc.RigidMCStats()
    for _ in 1:2000
        i = rand(sys.rng, 1:sys.n_mol)
        MCc.mc_translate!(sys, ff, i, 0.3, st); MCc.mc_rotate!(sys, ff, i, 0.5, st)
    end
    @test MCc.cells_check(c, sys) == 0 && st.trans_acc > 0
    # eKMC with solvent L-moves: identical trajectory with and without cells (debug rebuilds check the list)
    s1 = deepcopy(sys); s2 = deepcopy(sys); s1.cells = nothing
    r1 = MCc.run_rigid_ekmc!(s1, ff; events=3000, sample_every=10, block_events=1000, local_solvent=true, solvent_nu=1e-4)
    r2 = @test_logs min_level=Base.CoreLogging.Warn MCc.run_rigid_ekmc!(s2, ff; events=3000, sample_every=10,
            block_events=1000, local_solvent=true, solvent_nu=1e-4, neighbor_cells=true, debug_check_every=500)
    @test isapprox(s1.com, s2.com; rtol=0, atol=1e-9) && isapprox(r1.ek.phi, r2.ek.phi; rtol=1e-12)
    # flexible ethanol as the ladder solute with L-moves incl. conformers (its extent changes): list stays current
    ms = MCc.spce_solute_model(:ethanol_flex; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=9.0)
    N = 200; L = cbrt(N / 0.0334) / ms.σ_angstrom
    sy = MCc.RigidSystem(ms.templates, vcat(fill(1, N - 1), 2), L, ms.ϵ_kelvin / 300.0, Xoshiro(9))
    MCc.random_rigid_system!(sy, ms.ff); MCc.minimize_rigid!(sy, ms.ff; nsteps=200)
    res = MCc.build_reservoir(ms.templates, 2, ms.ff, sy.β, Xoshiro(4); M=500, burnin_sweeps=100, stride_sweeps=5, δ=0.03)
    ev = 4000
    @test_logs min_level=Base.CoreLogging.Warn MCc.run_rigid_ekmc!(sy, ms.ff; events=ev, sample_every=10, block_events=ev ÷ 4,
            reservoirs=[nothing, res], ladder_solute=N, ladder_nu=1e-3, ladder_wl_events=ev ÷ 2, ladder_local=true,
            neighbor_cells=true, debug_check_every=250)
    @test MCc.cells_check(sy.cells, sy) == 0
end

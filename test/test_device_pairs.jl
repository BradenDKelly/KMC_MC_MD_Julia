# Device pair kernel (DevicePairs.jl, DESIGN §15) on the KernelAbstractions CPU backend: rows and explicit tasks of committed,
# candidate (rigid-body and conformational) and overlapping poses = pair_energy_virial within the derived bounds
# (device_pairs_checks.jl); caps exact; :double and :mixed.
include(joinpath(@__DIR__, "device_pairs_checks.jl"))

function dp_water(N, seed; coul=:rf, cutoff=:site, rc=9.0, shift_lj=false)
    α = MCd.ewald_alpha_for(rc, 1e-5)
    m = MCd.spce_model(; α_per_angstrom=α, rc_angstrom=rc, coulomb=coul, epsilon_rf=Inf, cutoff=cutoff, shift_lj=shift_lj)
    L = cbrt(N / 0.0334) / m.σ_angstrom
    sys = MCd.RigidSystem([m.template], fill(1, N), L, m.ϵ_kelvin / 300.0, Xoshiro(seed))
    MCd.random_rigid_system!(sys, m.ff)
    return m.ff, sys
end
function dp_ibuprofen_water(nw, ni, seed)
    mi = MCd.spce_solute_model(:ibuprofen_lpg; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=9.0)
    ρ = 0.0334 * 0.6                                                    # dilute enough for random placement of the solutes
    L = cbrt((nw + 15ni) / ρ) / mi.σ_angstrom
    sys = MCd.RigidSystem([mi.templates[1], mi.templates[2]], vcat(fill(1, nw), fill(2, ni)), L, mi.ϵ_kelvin / 300.0, Xoshiro(seed))
    MCd.random_rigid_system!(sys, mi.ff)
    return mi.ff, sys
end

@testset "Device pair kernel (CPU backend) = pair_energy_virial: rows, tasks, caps, sync ($prec)" for prec in (:double, :mixed)
    cases = Any[]
    ff, sys = dp_water(120, 1; coul=:rf);                   push!(cases, (ff, sys, [1, 2, 60], "SPC/E RF site cut-off"))
    ff, sys = dp_water(120, 2; coul=:wolf, cutoff=:molecule); push!(cases, (ff, sys, [3, 4, 90], "SPC/E Wolf molecule cut-off"))
    ff, sys = dp_water(120, 3; coul=:ewald);                push!(cases, (ff, sys, [5, 6, 7], "SPC/E Ewald real space"))
    ff, sys = dp_water(120, 4; coul=:none, shift_lj=true);  push!(cases, (ff, sys, [8, 9, 10], "LJ only, shifted"))
    ff, sys = dp_ibuprofen_water(150, 4, 5);                push!(cases, (ff, sys, [151, 152, 1, 154], "ibuprofen (LigParGen) + SPC/E, RF"))
    for (ff, sys, sel, label) in cases
        r = check_device_pairs(sys, ff, :cpu, prec; sel=sel, seed=7, label=label)
        @info "device pairs $prec: $label" r.ntask r.nok r.nbad r.namb r.ncap r.worst r.wdev
        @test r.nbad == 0
        @test r.nok > 0.9 * r.ntask                    # the comparison is not vacuous (ambiguous tasks are few)
        @test r.ncap >= 1                              # the forced overlap exercised the cap
        @test r.same && r.sync_ok
    end
end

"""eKMC with device rows on `backend` (:double): every incremental quantity = the host rebuild (DESIGN §15 G2)."""
function dp_ekmc_check(backend::Symbol)
    # RF water with solvent L-moves, Ewald real space on the device + reciprocal space on the host, flexible ethanol liquid
    # (site/torsion slots, Wolf): every row the eKMC loop and the candidate rebuilds evaluate comes from the device kernel;
    # the debug rebuild (rebuild_rigid_ekmc!, slm_check!) recomputes everything with pair_energy_virial on the host.
    function drift(ek, sys, ff)
        φ = copy(ek.phi); P = copy(ek.pair_u)
        MCd.rebuild_rigid_ekmc!(ek, sys, ff)
        return max(maximum(abs.(φ .- ek.phi)), maximum(abs.(P .- ek.pair_u)))
    end
    for coul in (:rf, :ewald, :pme)                            # :pme = Ewald real space on the device + SPME mesh on the host
        ff, sys = dp_water(60, 7; coul=coul === :pme ? :ewald : coul, rc=6.0)
        coul === :ewald && MCd.attach_ewald!(sys, ff; rtol=1e-5, precision=:double)
        coul === :pme && MCd.attach_pme!(sys, ff; K=16, order=4)
        MCd.attach_device_pairs!(sys, ff; backend=backend, precision=:double)
        ek = MCd.init_rigid_ekmc(sys, ff)
        S = MCd.SolventLocalMoves(sys, ff, collect(1:sys.n_mol); kinds=[MCd.LM_T, MCd.LM_R], a=0.05, θmax=0.2, ν=1.0, νD=1.0)
        # candidate poses in the device bank: rebuilds and column updates as single launches
        MCd.attach_device_pairs!(sys, ff; backend=backend, precision=:double, max_poses=length(S.kinds) * length(S.mols) + 1)
        MCd.slm_attach_device!(S, sys, ff)
        @test S.dbase == 2
        acc = MCd.ChemicalPotentialAccumulator()
        for it in 1:150
            if it % 2 == 0
                MCd.slm_total_rate!(S, ek, sys.β); MCd.slm_fire!(S, sys, ff, ek, rand(sys.rng) * S.W)
            else
                k = MCd.select_by_mobility(ek, sys.rng)
                MCd.set_molecule!(sys, k, sys.L .* rand(sys.rng, 3), MCd.uniform_random_quaternion(sys.rng))
                MCd.update_rigid_ekmc_after_move!(ek, sys, ff, k); MCd.slm_moved!(S, sys, ff, k)
            end
        end
        @test MCd.slm_check!(S, sys, ff) < 1e-9
        @test drift(ek, sys, ff) < 1e-9
        MCd.ekmc_mc_sweep!(ek, sys, ff, 0.1, 0.3)              # MC trials: rows through eval_row! (device), commits synced
        @test drift(ek, sys, ff) < 1e-9
    end
    m = MCd.spce_solute_model(:ethanol_flex; coulomb=:wolf, cutoff=:site, rc_angstrom=7.0)
    t = m.templates[2]; ff = m.ff; N = 24
    sys = MCd.RigidSystem([t], fill(1, N), cbrt(N / (0.4 / 46.07 * 6.02214076e23 / 1e24)) / m.σ_angstrom,
                          m.ϵ_kelvin / 300.0, Xoshiro(4))
    MCd.random_rigid_system!(sys, ff); MCd.minimize_rigid!(sys, ff; nsteps=200)
    MCd.attach_device_pairs!(sys, ff; backend=backend, precision=:double)
    ek = MCd.init_rigid_ekmc(sys, ff)
    S = MCd.SolventLocalMoves(sys, ff, collect(1:N); kinds=[MCd.LM_S, MCd.LM_T, MCd.LM_D, MCd.LM_R], a=0.05, θmax=0.2,
                              ν=1.0, νD=1.0, site_a=0.02, φmax=0.6)
    MCd.attach_device_pairs!(sys, ff; backend=backend, precision=:double, max_poses=length(S.kinds) * N + 1)
    MCd.slm_attach_device!(S, sys, ff)
    for _ in 1:40
        MCd.slm_total_rate!(S, ek, sys.β); MCd.slm_fire!(S, sys, ff, ek, rand(sys.rng) * S.W)
    end
    @test all(S.n_move[[MCd.LM_S, MCd.LM_D]] .> 0)
    @test MCd.slm_check!(S, sys, ff) < 1e-9
    @test drift(ek, sys, ff) < 1e-9
    MCd.detach_device_pairs!(sys)
    @test sys.dev === nothing
end

@testset "Device pair rows inside eKMC (attach_device_pairs!, CPU backend, :double): incremental = host rebuild (G2)" begin
    dp_ekmc_check(:cpu)
end

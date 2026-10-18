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

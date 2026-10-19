# Device pair kernel (DevicePairs.jl, DESIGN §15) on CUDA: the checks of test_device_pairs.jl with backend = :cuda (rows,
# explicit tasks, caps, sync = pair_energy_virial within the derived bounds; :double and :mixed). Run with CUDA:
#   julia --project=/shared/kmc_polyatomic/envs/molsim_gpu test/test_device_pairs_cuda.jl
using CUDA
include(joinpath(@__DIR__, "test_device_pairs.jl"))               # CPU backend first (same systems), then CUDA below

@testset "Device pair kernel (CUDA) = pair_energy_virial: rows, tasks, caps, sync ($prec)" for prec in (:double, :mixed)
    CUDA.functional() || error("CUDA not functional on $(gethostname())")
    cases = Any[]
    ff, sys = dp_water(120, 1; coul=:rf);                   push!(cases, (ff, sys, [1, 2, 60], "SPC/E RF site cut-off"))
    ff, sys = dp_water(120, 2; coul=:wolf, cutoff=:molecule); push!(cases, (ff, sys, [3, 4, 90], "SPC/E Wolf molecule cut-off"))
    ff, sys = dp_water(120, 3; coul=:ewald);                push!(cases, (ff, sys, [5, 6, 7], "SPC/E Ewald real space"))
    ff, sys = dp_water(120, 4; coul=:none, shift_lj=true);  push!(cases, (ff, sys, [8, 9, 10], "LJ only, shifted"))
    ff, sys = dp_ibuprofen_water(150, 4, 5);                push!(cases, (ff, sys, [151, 152, 1, 154], "ibuprofen (LigParGen) + SPC/E, RF"))
    for (ff, sys, sel, label) in cases
        r = check_device_pairs(sys, ff, :cuda, prec; sel=sel, seed=7, label=label)
        @info "device pairs CUDA $prec: $label" r.ntask r.nok r.nbad r.namb r.ncap r.worst r.wdev
        @test r.nbad == 0
        @test r.nok > 0.9 * r.ntask
        @test r.ncap >= 1
        @test r.same && r.sync_ok
    end
end

@testset "Device pair rows inside eKMC on CUDA (:double): incremental = host rebuild (G2)" begin
    CUDA.functional() || error("CUDA not functional on $(gethostname())")
    dp_ekmc_check(:cuda)
end

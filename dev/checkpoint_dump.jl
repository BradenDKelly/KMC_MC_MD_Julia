"""
Checkpoint migration, step 1 (run with the OLD code, e.g. --project=/shared/kmc_polyatomic/KMC_old): read a chain
checkpoint (sys, run::RigidEKMCRun, ff, ctx) and write a version-independent dump in which RigidEKMCRun and its
RigidEKMCState are replaced by Dict{Symbol,Any} of their fields (all other types are unchanged between versions).
Usage: julia --project=<old> dev/checkpoint_dump.jl in.jls out.dump
"""

using MolSim, Serialization
const MCm = MolSim.MC
todict(x) = Dict{Symbol,Any}(f => getfield(x, f) for f in fieldnames(typeof(x)))
sys, r, ff, ctx = deserialize(ARGS[1])
d = todict(r)
d[:ek] = todict(r.ek)
open(io -> serialize(io, (sys, d, ff, ctx, string(typeof(r)))), ARGS[2], "w")
println("dumped ", ARGS[1], " → ", ARGS[2], "  (", length(d), " run fields, ", length(d[:ek]), " state fields, ev=", d[:ev], ")")

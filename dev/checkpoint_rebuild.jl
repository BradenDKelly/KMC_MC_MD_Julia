"""
Checkpoint migration, step 2 (run with the NEW code): rebuild RigidEKMCRun/RigidEKMCState from the dump of
dev/checkpoint_dump.jl, filling the fields added since (no Bennett accumulator, no coupling ladder, no fractional
molecule), and write a checkpoint the current code can continue. The continuation is identical to the one the old
code would have produced (test/test_checkpoint_migration.jl).
Usage: julia --project=<new> dev/checkpoint_rebuild.jl in.dump out.jls
"""

using MolSim, Serialization
const MCm = MolSim.MC
sys, d, ff, ctx, oldtype = deserialize(ARGS[1])
N = sys.n_mol
e = d[:ek]
state_defaults = Dict{Symbol,Any}(:lam => ones(N), :frac => 0, :bare_u => zeros(N), :bare_w => zeros(N), :frac_u => 0.0, :frac_w => 0.0)
ek = MCm.RigidEKMCState([haskey(e, f) ? e[f] : state_defaults[f] for f in fieldnames(MCm.RigidEKMCState)]...)
run_defaults = Dict{Symbol,Any}(:bar_species => 0, :bar => nothing, :bar_members => Int[], :ladder => nothing)
vals = Any[]
for f in fieldnames(MCm.RigidEKMCRun)
    f == :ek && (push!(vals, ek); continue)
    haskey(d, f) ? push!(vals, d[f]) : (haskey(run_defaults, f) ? push!(vals, run_defaults[f]) : error("no default for new field $f"))
end
r = MCm.RigidEKMCRun(vals...)
open(io -> serialize(io, (sys, r, ff, ctx)), ARGS[2], "w")
println("rebuilt ", ARGS[1], " → ", ARGS[2], "  ev=", r.ev, " of ", r.events)

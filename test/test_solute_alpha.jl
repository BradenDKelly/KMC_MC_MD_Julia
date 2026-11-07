# Ewald/PME splitting parameter of the solute drivers (JOURNAL 2026-11-06): the chain scripts exported the Wolf default
# SPCE_ALPHA=0.20 for every Coulomb method, so dilute Ewald/PME runs used α = 0.2 Å⁻¹ at r_c = 10 Å instead of the
# GROMACS ewald-rtol value (1/β = 0.320163 nm in the gromacs_fe logs, i.e. β = 0.31234 Å⁻¹).
using Test
using MolSim
const MCa = MolSim.MC

@testset "solute α: Ewald/PME from ewald-rtol, Wolf default" begin
    α = MCa.solute_alpha(:ewald, nothing)
    @test isapprox(α, 1 / 3.20163; rtol=1e-5)                         # GROMACS log: Gaussian width 1/β = 0.320163 nm
    @test MCa.solute_alpha(:ewald, α) == α
    @test_throws ErrorException MCa.solute_alpha(:ewald, 0.20)        # the old chain-script default is refused
    @test MCa.solute_alpha(:ewald, 0.20; override=true) == 0.20
    @test MCa.solute_alpha(:wolf, nothing) == 0.20
    @test MCa.solute_alpha(:wolf, 0.25) == 0.25
    @test MCa.solute_alpha(:rf, nothing) == 0.20
end

@testset "solute chain scripts do not force α for Ewald/PME" begin
    for f in ("spce_solute_chain.sbatch", "spce_solute_validation.sbatch")
        lines = filter(l -> startswith(l, "export SPCE_"), readlines(joinpath(@__DIR__, "..", "scripts", "slurm", f)))
        script = "unset SPCE_ALPHA; export SPCE_COULOMB=pme; " * join(lines, "; ") * "; echo \${SPCE_ALPHA-unset}"
        @test strip(read(`bash -c $script`, String)) == "unset"
    end
end

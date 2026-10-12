#!/bin/bash
#SBATCH --job-name=rigid_slow
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/shared/kmc_polyatomic/slurm_logs/rigid_slow_%j.out
#SBATCH --error=/shared/kmc_polyatomic/slurm_logs/rigid_slow_%j.err
# Slow-gated rigid-molecule tests (SPC/E NVT and NPT, MC vs eKMC) — CPU only, all paths on /shared.
export JULIA_DEPOT_PATH=/shared/kmc_polyatomic/julia_depot
export MOLSIM_SLOW_TESTS=1
JULIA=/shared/julia_depot/juliaup/julia-1.12.5+0.x64.linux.gnu/bin/julia
PROJECT=/shared/kmc_polyatomic/KMC_MC_MD_Julia
cd "$PROJECT" || exit 1
echo "host=$(hostname) start=$(date)"
"$JULIA" -t "${SLURM_CPUS_PER_TASK:-1}" --project="$PROJECT" -e 'using Test, MolSim; include("test/test_rigid_molecules.jl")'
echo "end=$(date) exit=$?"

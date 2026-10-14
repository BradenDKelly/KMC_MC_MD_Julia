#!/bin/bash
#SBATCH --job-name=mbar_mc_lj
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/kmc_polyatomic/slurm_logs/mbar_mc_lj_%A_%a.out
#SBATCH --error=/shared/kmc_polyatomic/slurm_logs/mbar_mc_lj_%A_%a.err
#SBATCH --array=1-4
# Metropolis + MBAR reference for the LJ ladder validation (dev/mbar_mc_lj.jl): EPS_S, N, SWEEPS per rung
export JULIA_DEPOT_PATH=/shared/kmc_polyatomic/julia_depot
JULIA=/shared/julia_depot/juliaup/julia-1.12.5+0.x64.linux.gnu/bin/julia
PROJECT=/shared/kmc_polyatomic/KMC_MC_MD_Julia
OUT=/shared/kmc_polyatomic/KMC_MC_MD_Julia/dev/results/bennett_lj
mkdir -p "$OUT"; cd "$PROJECT" || exit 1
echo "host=$(hostname) eps=${EPS_S:-9} N=${N:-108} sweeps=${SWEEPS:-20000} seed=$SLURM_ARRAY_TASK_ID start=$(date)"
"$JULIA" --project="$PROJECT" dev/mbar_mc_lj.jl "${EPS_S:-9}" "${N:-108}" "${SWEEPS:-20000}" "$SLURM_ARRAY_TASK_ID" "$OUT"
echo "end=$(date)"

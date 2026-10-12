#!/bin/bash
#SBATCH --job-name=spce_solute
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/shared/kmc_polyatomic/slurm_logs/spce_solute_%A_%a.out
#SBATCH --error=/shared/kmc_polyatomic/slurm_logs/spce_solute_%A_%a.err
#SBATCH --array=1-3
#
# One TraPPE-UA solute (SOLUTE=methane|methanol) at infinite dilution in SPC/E water: MC Widom /
# removal estimators vs hybrid eKMC per-species μ with jackknife errors (dev/validate_spce_solute.jl).
# CPU-only; every path on /shared. Submit:  SOLUTE=methanol sbatch scripts/slurm/spce_solute_validation.sbatch

export JULIA_DEPOT_PATH=/shared/kmc_polyatomic/julia_depot
JULIA=/shared/julia_depot/juliaup/julia-1.12.5+0.x64.linux.gnu/bin/julia
PROJECT=/shared/kmc_polyatomic/KMC_MC_MD_Julia
OUT=/shared/kmc_polyatomic/KMC_MC_MD_Julia/dev/results/spce_solute

SOLUTE=${SOLUTE:-methane}
N=${N:-500}
MC_SWEEPS=${MC_SWEEPS:-8000}        # 2000 equilibration + 8000 production sweeps
EKMC_EVENTS=${EKMC_EVENTS:-100000}  # hybrid events (≈ sweeps)
SEED=${SLURM_ARRAY_TASK_ID:-1}
export SPCE_ALPHA=${SPCE_ALPHA:-0.20} SPCE_CUTOFF=${SPCE_CUTOFF:-molecule} SPCE_LAMBDA=${SPCE_LAMBDA:-10}
mkdir -p "$OUT"

echo "host=$(hostname) solute=$SOLUTE N=$N mc_sweeps=$MC_SWEEPS ekmc_events=$EKMC_EVENTS seed=$SEED alpha=$SPCE_ALPHA cutoff=$SPCE_CUTOFF lambda=$SPCE_LAMBDA start=$(date)"
cd "$PROJECT" || exit 1
"$JULIA" -t "${SLURM_CPUS_PER_TASK:-1}" --project="$PROJECT" dev/validate_spce_solute.jl "$SOLUTE" "$N" "$MC_SWEEPS" "$EKMC_EVENTS" "$SEED" "$OUT"
echo "end=$(date)"

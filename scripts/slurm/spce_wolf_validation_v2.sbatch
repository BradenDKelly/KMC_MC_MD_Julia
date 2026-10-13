#!/bin/bash
#SBATCH --job-name=spce_wolf2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/kmc_polyatomic/slurm_logs/spce_wolf_%A_%a.out
#SBATCH --error=/shared/kmc_polyatomic/slurm_logs/spce_wolf_%A_%a.err
#SBATCH --array=1-3
#
# SPC/E liquid water, Wolf electrostatics: Metropolis MC vs eKMC vs NIST reference.
# CPU-only job (no --gres); every path is on /shared because /home is not exported to workers.
# Submit from the head node:  sbatch scripts/slurm/spce_wolf_validation.sbatch
# Arguments to the Julia script: N  mc_sweeps  ekmc_events  seed  outdir

export JULIA_DEPOT_PATH=/shared/kmc_polyatomic/julia_depot
JULIA=/shared/julia_depot/juliaup/julia-1.12.5+0.x64.linux.gnu/bin/julia
PROJECT=/shared/kmc_polyatomic/KMC_MC_MD_Julia
OUT=/shared/kmc_polyatomic/KMC_MC_MD_Julia/dev/results/spce_wolf

N=${N:-500}
MC_SWEEPS=${MC_SWEEPS:-16000}   # v2: 4000 equilibration + 16000 production sweeps (8 threads)
EKMC_EVENTS=${EKMC_EVENTS:-100000000}
SEED=${SLURM_ARRAY_TASK_ID:-1}

echo "host=$(hostname) N=$N mc_sweeps=$MC_SWEEPS ekmc_events=$EKMC_EVENTS seed=$SEED start=$(date)"
export SPCE_COULOMB=${SPCE_COULOMB:-wolf} SPCE_EPS_RF=${SPCE_EPS_RF:-0}
cd "$PROJECT" || exit 1
"$JULIA" -t "${SLURM_CPUS_PER_TASK:-1}" --project="$PROJECT" dev/validate_spce_wolf.jl "$N" "$MC_SWEEPS" "$EKMC_EVENTS" "$SEED" "$OUT"
echo "end=$(date)"

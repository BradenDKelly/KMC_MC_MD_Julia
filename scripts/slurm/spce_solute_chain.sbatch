#!/bin/bash
#SBATCH --job-name=spce_chain
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/kmc_polyatomic/slurm_logs/spce_chain_%A_%a.out
#SBATCH --error=/shared/kmc_polyatomic/slurm_logs/spce_chain_%A_%a.err
#
# One 4 h piece of a long solute eKMC run (dev/validate_spce_solute.jl in checkpoint mode). The first piece
# builds the reservoir, runs MC and starts eKMC; every piece runs eKMC for SPCE_WALL_SECONDS, writes the
# checkpoint and exits; the piece that reaches EKMC_EVENTS writes the report and renames the checkpoint to
# .final; later pieces exit at once. 4 h pieces backfill on the shared queue (JOURNAL entry 30). Chain with
#   scripts/slurm/submit_solute_chain.sh <pieces>      (same env variables as spce_solute_validation.sbatch)

export JULIA_DEPOT_PATH=/shared/kmc_polyatomic/julia_depot
JULIA=/shared/julia_depot/juliaup/julia-1.12.5+0.x64.linux.gnu/bin/julia
PROJECT=/shared/kmc_polyatomic/KMC_MC_MD_Julia
OUT=/shared/kmc_polyatomic/KMC_MC_MD_Julia/dev/results/spce_solute
CKDIR=/shared/kmc_polyatomic/checkpoints

SOLUTE=${SOLUTE:-methanol_flex}
N=${N:-500}
MC_SWEEPS=${MC_SWEEPS:-4000}
EKMC_EVENTS=${EKMC_EVENTS:-1000000000}
SEED=${SLURM_ARRAY_TASK_ID:-1}
export SPCE_ALPHA=${SPCE_ALPHA:-0.20} SPCE_CUTOFF=${SPCE_CUTOFF:-site} SPCE_LAMBDA=${SPCE_LAMBDA:-0}
export SPCE_COULOMB=${SPCE_COULOMB:-rf} SPCE_EPS_RF=${SPCE_EPS_RF:-0} SPCE_TRACE=${SPCE_TRACE:-0}
# eKMC budget per piece: 4 h minus 30 min for Julia start-up, the first piece's reservoir + MC (~6 min) and the checkpoint
export SPCE_WALL_SECONDS=${SPCE_WALL_SECONDS:-12600}
mkdir -p "$OUT" "$CKDIR"
export SPCE_CHECKPOINT=$CKDIR/${SOLUTE}_N${N}_seed${SEED}${SPCE_TAG}.jls

echo "host=$(hostname) solute=$SOLUTE N=$N mc_sweeps=$MC_SWEEPS ekmc_events=$EKMC_EVENTS seed=$SEED coulomb=$SPCE_COULOMB cutoff=$SPCE_CUTOFF lambda=$SPCE_LAMBDA trace=$SPCE_TRACE wall=$SPCE_WALL_SECONDS ckpt=$SPCE_CHECKPOINT start=$(date)"
cd "$PROJECT" || exit 1
"$JULIA" -t "${SLURM_CPUS_PER_TASK:-1}" --project="$PROJECT" dev/validate_spce_solute.jl "$SOLUTE" "$N" "$MC_SWEEPS" "$EKMC_EVENTS" "$SEED" "$OUT"
echo "end=$(date)"

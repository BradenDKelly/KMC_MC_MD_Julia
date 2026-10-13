#!/bin/bash
# Submit a chain of 4 h pieces of spce_solute_chain.sbatch: piece k of seed s starts when piece k−1 of seed s
# has finished successfully (Slurm --dependency=aftercorr). Usage (env as in spce_solute_chain.sbatch):
#   SPCE_TAG=_ev1e9 scripts/slurm/submit_solute_chain.sh <pieces> [array, default 1-6]
set -e
PIECES=${1:?number of pieces}
ARRAY=${2:-1-6}
SCRIPT=/shared/kmc_polyatomic/KMC_MC_MD_Julia/scripts/slurm/spce_solute_chain.sbatch
prev=$(sbatch --parsable --array="$ARRAY" --export=ALL "$SCRIPT")
echo "$prev"
for ((k = 2; k <= PIECES; k++)); do
    prev=$(sbatch --parsable --array="$ARRAY" --dependency=aftercorr:"$prev" --export=ALL "$SCRIPT")
    echo "$prev"
done

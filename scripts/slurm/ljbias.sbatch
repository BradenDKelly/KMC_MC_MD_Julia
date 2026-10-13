#!/bin/bash
#SBATCH --job-name=ljbias
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=02:00:00
#SBATCH --output=/shared/kmc_polyatomic/slurm_logs/ljbias_%A_%a.out
#SBATCH --error=/shared/kmc_polyatomic/slurm_logs/ljbias_%A_%a.err
#SBATCH --array=1-48
#
# Finite-run bias of the eKMC time average on single-site LJ (dev/ekmc_time_average_bias.jl), 16 seeds per
# condition: 1–16 dense liquid ρ*=0.85 T*=0.9; 17–32 same + one solute ε=9 (ε_us=3); 33–48 control ρ*=0.70 T*=1.2.
# CPU-only; every path on /shared.
export JULIA_DEPOT_PATH=/shared/kmc_polyatomic/julia_depot
JULIA=/shared/julia_depot/juliaup/julia-1.12.5+0.x64.linux.gnu/bin/julia
PROJECT=/shared/kmc_polyatomic/KMC_MC_MD_Julia
OUT=/shared/kmc_polyatomic/KMC_MC_MD_Julia/dev/results/ljbias
EVENTS=${EVENTS:-50000000}; MC_SWEEPS=${MC_SWEEPS:-20000}; N=${N:-108}
i=${SLURM_ARRAY_TASK_ID}; c=$(( (i - 1) / 16 )); seed=$(( (i - 1) % 16 + 1 ))
case $c in
  0) RHO=0.85; T=0.9; ES=0 ;;
  1) RHO=0.85; T=0.9; ES=9 ;;
  2) RHO=0.70; T=1.2; ES=0 ;;
esac
echo "host=$(hostname) rho=$RHO T=$T eps_s=$ES seed=$seed events=$EVENTS start=$(date)"
cd "$PROJECT" || exit 1
"$JULIA" --project="$PROJECT" dev/ekmc_time_average_bias.jl $RHO $T $N $EVENTS $seed $ES $MC_SWEEPS "$OUT" | grep -v "^B "
echo "end=$(date)"

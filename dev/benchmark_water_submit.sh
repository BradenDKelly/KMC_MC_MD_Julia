#!/bin/bash
# Submit the B1 water benchmark (JOURNAL 68): one prep job per seed (quench + 2000 MC sweeps → shared start), then
# mc / stock / local for each seed, 1 thread, WALL seconds each. Usage: dev/benchmark_water_submit.sh [seeds=3] [wall=10800]
set -e
SEEDS=${1:-3}; WALL=${2:-10800}
TREE=/shared/kmc_polyatomic/KMC_dev5
OUT=$TREE/dev/results/bench_water
JL=/shared/julia_depot/juliaup/julia-1.12.5+0.x64.linux.gnu/bin/julia
E="export JULIA_DEPOT_PATH=/shared/kmc_polyatomic/julia_depot; cd $TREE"
LOG=/shared/kmc_polyatomic/slurm_logs/v2_jobids.txt
mkdir -p "$OUT"
for s in $(seq 1 "$SEEDS"); do
    p=$(sbatch --parsable --job-name=bench_prep$s --partition=gpu --cpus-per-task=1 --mem=4G --time=04:00:00 \
        --output=/shared/kmc_polyatomic/slurm_logs/bench_%j.out --wrap "$E; $JL -t 1 --project=. dev/benchmark_water.jl prep $s $OUT")
    echo "$p BENCH prep seed $s $(date '+%F %H:%M')" >> $LOG
    for m in mc stock local; do
        j=$(sbatch --parsable --job-name=bench_${m}$s --partition=gpu --cpus-per-task=1 --mem=4G --time=04:00:00 --dependency=afterok:$p \
            --output=/shared/kmc_polyatomic/slurm_logs/bench_%j.out --wrap "$E; $JL -t 1 --project=. dev/benchmark_water.jl $m $s $WALL $OUT")
        echo "$j BENCH $m seed $s wall $WALL afterok:$p $(date '+%F %H:%M')" >> $LOG
        echo "seed $s $m $j (after prep $p)"
    done
done

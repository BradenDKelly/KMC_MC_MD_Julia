#!/usr/bin/env python3
"""
Cost-per-accuracy table of the B1 water benchmark (dev/benchmark_water.jl; JOURNAL 68).
For each method and observable (U/N kJ/mol, P atm, mu_water kJ/mol): mean over seeds, the error of that mean from the block
errors (sqrt(sum σ_s²)/n) and from the seed spread (sd/sqrt(n)), σ = the larger of the two; total CPU C (s, 1 thread); the
cost to reach a target error σ_t, C·(σ/σ_t)² (error ∝ 1/√time; Flyvbjerg & Petersen 1989), as in dev/cost_per_accuracy.py;
and the difference from the Metropolis mean in units of the combined σ (a method whose average is off is flagged; its cost
is meaningless). Usage: python3 dev/benchmark_water_table.py <dir> [σ_U=0.05] [σ_P=20] [σ_mu=0.5]
"""
import glob, math, re, statistics as st, sys
d = sys.argv[1]
tgt = {'U': float(sys.argv[2]) if len(sys.argv) > 2 else 0.05, 'P': float(sys.argv[3]) if len(sys.argv) > 3 else 20.0,
       'mu_w': float(sys.argv[4]) if len(sys.argv) > 4 else 0.5}
pat = re.compile(r'RESULT method=(\w+) seed=(\d+) cpu_s=(\d+) moves=(\d+).*?U=([-\d.]+) ± ([-\d.]+)\s+P=([-\d.]+) ± ([-\d.]+)\s+mu_w=([-\d.]+) ± ([-\d.]+)')
runs = {}
for f in sorted(glob.glob(d + '/bench_*_seed*.txt')):
    for line in open(f):
        m = pat.search(line)
        if m:
            g = m.groups()
            runs.setdefault(g[0], []).append(dict(seed=int(g[1]), cpu=float(g[2]), moves=int(g[3]),
                U=(float(g[4]), float(g[5])), P=(float(g[6]), float(g[7])), mu_w=(float(g[8]), float(g[9]))))
def summary(rs, ob):
    v = [r[ob][0] for r in rs]; e = [r[ob][1] for r in rs]; n = len(rs)
    mean = sum(v) / n
    s_blk = math.sqrt(sum(x * x for x in e)) / n
    s_seed = st.stdev(v) / math.sqrt(n) if n > 1 else float('nan')
    s = max(s_blk, s_seed) if n > 1 else s_blk
    return mean, s_blk, s_seed, s
ref = {ob: summary(runs['mc'], ob) for ob in tgt} if 'mc' in runs else {}
print(f"{'method':7} {'obs':5} {'seeds':>5} {'mean':>10} {'σ_block':>8} {'σ_seed':>8} {'CPU h':>7} {'CPU h to σ_t':>13} {'Δ vs MC / σ':>12}")
for meth in ('mc', 'stock', 'local'):
    if meth not in runs: continue
    rs = runs[meth]; C = sum(r['cpu'] for r in rs) / 3600
    for ob in ('U', 'P', 'mu_w'):
        mean, sb, ss, s = summary(rs, ob)
        cost = C * (s / tgt[ob]) ** 2
        dz = ''
        if ref and meth != 'mc':
            rm, _, _, rsg = ref[ob]; dz = f"{(mean - rm) / math.sqrt(s * s + rsg * rsg):+.1f}"
        print(f"{meth:7} {ob:5} {len(rs):5d} {mean:10.3f} {sb:8.3f} {ss:8.3f} {C:7.2f} {cost:13.2f} {dz:>12}")
print(f"targets σ_t: U {tgt['U']} kJ/mol, P {tgt['P']} atm, mu_w {tgt['mu_w']} kJ/mol; |Δ/σ| > 3 means the method's average disagrees with Metropolis")

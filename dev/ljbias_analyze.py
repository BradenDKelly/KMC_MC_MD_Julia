#!/usr/bin/env python3
"""Analyse dev/results/ljbias/*.txt (dev/ekmc_time_average_bias.jl): per condition, the eKMC global time
average U/N from the first K of 100 blocks (K = 1, 3, 10, 30, 100) across seeds vs the pooled MC reference.
Reports mean bias, its standard error (seed spread / sqrt(n)), seed SD, the median per-seed jackknife error
(what a single run would claim), and Kish n_eff of the event residence times."""
import glob, re, sys, math, statistics as st
d = sys.argv[1] if len(sys.argv) > 1 else 'dev/results/ljbias'
runs = {}
for f in glob.glob(f'{d}/ljbias_*.txt'):
    s = open(f).read()
    key = re.search(r'ljbias_(r[\d.]+_T[\d.]+_N\d+_es[\d.]+)_seed', f).group(1)
    mc = re.search(r'^MC .*U/N=(\S+) ± (\S+)', s, re.M)
    ek = re.search(r'^EKMC .*kish_neff=(\S+) max_dt_share=(\S+)', s, re.M)
    B = [tuple(map(float, l.split()[2:4])) for l in s.splitlines() if l.startswith('B ')]
    if not (mc and ek and len(B) == 100): continue
    ev = int(re.search(r'events=(\d+)', s).group(1))
    runs.setdefault(key, []).append(dict(ev=ev, mc=float(mc.group(1)), emc=float(mc.group(2)), B=B,
                                         neff=float(ek.group(1)), maxs=float(ek.group(2))))
def jk(B):  # global ratio + delete-one-block jackknife error
    X = sum(b[0] for b in B); T = sum(b[1] for b in B); n = len(B)
    if n < 2: return X / T, float('nan')
    th = [(X - b[0]) / (T - b[1]) for b in B]; m = sum(th) / n
    return X / T, math.sqrt((n - 1) / n * sum((t - m) ** 2 for t in th))
for key, rs in sorted(runs.items()):
    mcs = [r['mc'] for r in rs]; ref = st.mean(mcs); eref = st.stdev(mcs) / math.sqrt(len(mcs))
    print(f"\n== {key}: {len(rs)} seeds  MC U/N = {ref:.4f} ± {eref:.4f} (seed spread)  "
          f"Kish n_eff median {st.median(r['neff'] for r in rs):.0f}  max single-event time share median {st.median(r['maxs'] for r in rs):.3f}")
    print(f"{'events':>10} {'mean bias':>10} {'± se':>7} {'seed SD':>8} {'median jk err':>13} {'frac seeds above MC':>20}")
    for K in (1, 3, 10, 30, 100):
        est = [jk(r['B'][:K]) for r in rs]
        v = [e[0] for e in est]; errs = [e[1] for e in est if not math.isnan(e[1])]
        b = st.mean(v) - ref; sd = st.stdev(v)
        print(f"{K * rs[0]['ev'] // 100:>10.0e} {b:>10.4f} {sd / math.sqrt(len(v)):>7.4f} {sd:>8.4f} "
              f"{(st.median(errs) if errs else float('nan')):>13.4f} {sum(x > ref for x in v) / len(v):>20.2f}")

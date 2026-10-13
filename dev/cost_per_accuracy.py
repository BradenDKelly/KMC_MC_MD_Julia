#!/usr/bin/env python3
"""
Cost of each method per unit of accuracy, from the logged runs (JOURNAL 2026-10-12 entry 18).

For every (method, system) that has finished: value, standard error σ, CPU cost C = cores × wall time
(including equilibration / EM phases), and the cost extrapolated to a target error σ_t:
    C(σ_t) = C × (σ/σ_t)²        (error ∝ 1/√time; Flyvbjerg & Petersen 1989 for the block errors)
Sources: dev/results/spce_solute/*.txt and spce_wolf/*.txt (MC and eKMC lines carry time= and ± errors;
4 cores per task), gromacs_fe/<solute>/lambda_*/prod.log (wall time per window; 4 cores) and bar.sh output.
Caveats printed with the table: the eKMC jackknife can underestimate heavy-tailed errors (compare with the
seed spread, which is also reported), BAR's error comes from gmx bar's 5-block estimate, Widom's error is a
plain block error and its *bias* is not an error bar.
Usage: python3 dev/cost_per_accuracy.py [sigma_target_kJ=0.5]
"""
import glob, os, re, subprocess, sys, statistics as st
G = '/shared/kmc_polyatomic/gromacs_fe'
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
CORES = 4
sig_t = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
rows = []

def add(method, system, model, value, err, cpu_s, note=''):
    rows.append((method, system, model, value, err, cpu_s, note))

# ---------------- eKMC / MC (MolSim) ----------------
for f in sorted(glob.glob(R + '/spce_solute/*.txt') + glob.glob(R + '/spce_wolf/*ev1e5*.txt') + glob.glob(R + '/spce_wolf/*_pure*.txt')):
    name = os.path.basename(f)
    m = re.match(r'spce_(\w+?)_N(\d+)_seed(\d+)_(\w+?)_(a[\d.]+|rfinf|rf[\d.]+)_cap(\d+)', name)
    if not m: continue
    sol, N, seed, cutoff, elec, cap = m.groups()
    system = ('water' if sol == 'wolf' else sol + ' in water')
    model = ('Wolf' if elec.startswith('a') else 'RF') + f'/{cutoff}'
    txt = open(f).read()
    mc = re.search(r'^MC +sweeps=(\d+) time=(\d+)s.*?(?:mu_sol_widom|mu_widom)=([-\d.]+) ± ([-\d.]+)', txt, re.M)
    eq = re.search(r'MC equilibration \d+ sweeps: ([\d.]+) s', txt)
    t_eq = float(eq.group(1)) if eq else 0.0
    if mc:
        add('MC Widom insertion', system, model, float(mc.group(3)), float(mc.group(4)), CORES * (float(mc.group(2)) + t_eq), f'seed {seed}')
    ek = re.search(r'^eKMC events=(\d+) time=(\d+)s.*?(?:mu_sol=|mu_jk=)([-\d.]+) ± ([-\d.]+)', txt, re.M)
    if ek:
        t_mc = float(mc.group(2)) if mc else 0.0
        extra = re.search(r'mu_sol \+ E_const = ([-\d.]+) ± ([-\d.]+)', txt)
        val = float(extra.group(1)) if extra else float(ek.group(3))
        lam = re.search(r'λ=([\d.e+-]+)', txt)
        stock = lam is not None and float(lam.group(1)) == 0.0
        kind = 'eKMC stock (λ=0)' if stock else 'eKMC hybrid (λ=%.1e)' % (float(lam.group(1)) if lam else float('nan'))
        add(kind, system, model, val, float(ek.group(4)), CORES * (float(ek.group(2)) + t_eq + t_mc), f'seed {seed}, {int(ek.group(1)):.0e} events; cost incl. MC equilibration')

# ---------------- GROMACS BAR ----------------
for d in sorted(glob.glob(G + '/*/lambda_00')):
    S = os.path.basename(os.path.dirname(d))
    logs = glob.glob(G + f'/{S}/lambda_*/prod.log')
    if len(logs) < 21: continue
    wall = 0.0
    for L in (glob.glob(G + f'/{S}/lambda_*/prod.log') + glob.glob(G + f'/{S}/lambda_*/eq.log') + glob.glob(G + f'/{S}/lambda_*/em.log')):
        t = re.search(r'^\s+Time:\s+[\d.]+\s+([\d.]+)\s+', open(L).read(), re.M)   # "Time: core-s wall-s %"
        if t: wall += float(t.group(1))
    try:
        out = subprocess.run(['bash', G + '/bar.sh', S], capture_output=True, text=True, timeout=600).stdout
        tot = re.search(r'total\s+0\s+-\s+20,\s+DG\s+([-\d.]+)\s+\+/-\s+([\d.]+)', out)
    except Exception:
        tot = None
    if tot:
        elec = 'RF' if S.endswith('_rf') else 'PME'
        add('GROMACS BAR decoupling (21 λ)', S.replace('_rf', '') + ' in water', elec + '/site', -float(tot.group(1)), float(tot.group(2)), CORES * wall, '1 ns/window + EM + 100 ps eq')

print(f'# cost per accuracy, target σ = {sig_t} kJ/mol; CPU cost in core-hours (4 cores per task)')
print(f'{"method":32s} {"system":20s} {"model":12s} {"value":>9s} {"σ":>6s} {"core-h":>8s} {"core-h@σ_t":>11s}  note')
for method, system, model, v, e, c, note in rows:
    ch = c / 3600
    print(f'{method:32s} {system:20s} {model:12s} {v:9.2f} {e:6.2f} {ch:8.2f} {ch * (e / sig_t) ** 2:11.2f}  {note}')
# seed spreads for the eKMC and Widom rows (the honest error when > jackknife)
print('\n# seed spread (std of the per-seed values) where ≥ 2 seeds exist:')
from collections import defaultdict
grp = defaultdict(list)
for method, system, model, v, e, c, note in rows:
    if 'seed' in note: grp[(method.split(' (')[0], system, model)].append((v, e, c))
for k, L in grp.items():
    if len(L) >= 2:
        vals = [x[0] for x in L]; errs = [x[1] for x in L]; cost = sum(x[2] for x in L) / 3600
        sd = st.pstdev(vals) * (len(vals) / (len(vals) - 1)) ** 0.5
        sem = sd / len(vals) ** 0.5
        print(f'{k[0]:32s} {k[1]:20s} {k[2]:12s} mean {st.mean(vals):8.2f}  std over seeds {sd:5.2f}  SEM {sem:5.2f}  mean jackknife {st.mean(errs):5.2f}  total core-h {cost:6.2f}  core-h@σ_t (from SEM) {cost * (sem / sig_t) ** 2:7.2f}')

# Caffeine, OPLS-AA / 1.14*CM1A-LBCC, from LigParGen

- Requested by the user 2026-10-18 ("do butanol and caffeine as well (solvated in water)"): hydration free energy by eKMC vs
  GROMACS, as for the C1–C3 alcohols.
- Generated 2026-10-18 with the LigParGen web service (jorgensenresearch.com/ligpargen, API `/api/lpg-run`; LigParGen 2.1,
  BOSS 5.0, RDKit 2023.03.3), the request used for ibuprofen.
- Input: SMILES `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` (caffeine, 1,3,7-trimethylxanthine), charge model 1.14*CM1A-LBCC (site default
  for neutral molecules), net charge 0, 0 optimisation rounds (site default).
- Files kept: `LIG.gmx.itp`, `LIG.gmx.gro`, `LIG.log` — unmodified. Raw response in /shared/kmc_polyatomic/ligpargen/caffeine_raw.
- Cite: L. S. Dodda, I. Cabeza de Vaca, J. Tirado-Rives, W. L. Jorgensen, Nucleic Acids Res. 45 (2017) W331–W336;
  L. S. Dodda, J. Z. Vilseck, J. Tirado-Rives, W. L. Jorgensen, J. Phys. Chem. B 121 (2017) 3864 (CM1A-LBCC).

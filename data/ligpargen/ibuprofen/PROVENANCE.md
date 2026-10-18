# Ibuprofen, OPLS-AA / 1.14*CM1A-LBCC, from LigParGen

- Requested by the user 2026-10-17 ("use ibuprofen with OPLS-AA from LigParGen") as the drug-sized test molecule (DESIGN §14 V4).
- Generated 2026-10-17 with the LigParGen web service (jorgensenresearch.com/ligpargen, API `/api/lpg-run`;
  LigParGen 2.1, BOSS 5.0, RDKit 2023.03.3).
- Input: SMILES `CC(C)Cc1ccc(cc1)C(C)C(=O)O` (racemic ibuprofen, no stereo), charge model 1.14*CM1A-LBCC (the site's default
  for neutral molecules), net charge 0, 0 optimisation rounds (site default).
- Files kept: `LIG.gmx.itp` (GROMACS topology: [atomtypes] with σ/ϵ, [atoms], bonds, angles, RB propers (type 3),
  periodic impropers (type 4), [pairs]; defaults comb-rule 3, fudgeLJ = fudgeQQ = 0.5), `LIG.gmx.gro` (LigParGen geometry, nm),
  `LIG.log` (LigParGen log). Unmodified.
- Cite: L. S. Dodda, I. Cabeza de Vaca, J. Tirado-Rives, W. L. Jorgensen, Nucleic Acids Res. 45 (2017) W331–W336;
  L. S. Dodda, J. Z. Vilseck, J. Tirado-Rives, W. L. Jorgensen, J. Phys. Chem. B 121 (2017) 3864 (CM1A-LBCC).

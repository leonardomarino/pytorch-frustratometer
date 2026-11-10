# tests/length_test — LOCAL REGENERATION, not the upstream fixture

`201_1A0F_A.{pdb,cif}` were missing from the upstream commit (56cd8adf): `*.pdb` is gitignored and the
`.cif` was never committed. They were rebuilt locally on 2026-11-10 with `build_fixture.py` from RCSB
1A0F (E. coli glutathione S-transferase) chain A, ATOM records only.

**Deliberate modification:** residue 100 (TYR) is deleted. RCSB 1A0F chain A has no `label_seq_id` gap
(1-201 contiguous), but `tests/test_pdb_cif_consistency.py` is about a CIF with "a genuine label_seq_id gap
(chain A: jump of 2, i.e. one missing residue)". The deletion gives exactly that (99 -> 101) in both files.
The result is 200 residues / 1600 atoms. The upstream file's true origin is unknown, so treat the
`test_pdb_cif_*` results as exercising the described gap case, not as parity with the upstream fixture.

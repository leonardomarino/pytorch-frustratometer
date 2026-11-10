"""Derive tests/length_test/201_1A0F_A.{pdb,cif} from RCSB 1A0F chain A, ATOM records only,
with residue 100 (TYR) deleted so label_seq_id jumps 99 -> 101 (the gap the test docstring describes)."""
import sys, shlex
src_dir, out_dir, DROP = sys.argv[1], sys.argv[2], 100
# --- CIF: keep the atom_site loop, chain A, ATOM only, drop residue DROP; label_seq_id values left untouched
lines = open(f"{src_dir}/1A0F.cif").read().splitlines()
cols = [l.strip() for l in lines if l.startswith("_atom_site.")]
first = next(i for i, l in enumerate(lines) if l.startswith("_atom_site."))
start = first + len(cols)
ix = {c.split(".")[1]: i for i, c in enumerate(cols)}
kept = []
for l in lines[start:]:
    if l.startswith(("ATOM", "HETATM")):
        f = shlex.split(l)
        assert len(f) == len(cols), l
        if f[ix["group_PDB"]] == "ATOM" and f[ix["auth_asym_id"]] == "A" and int(f[ix["label_seq_id"]]) != DROP:
            kept.append(l)
    elif l.startswith(("#", "loop_", "_")):
        break
with open(f"{out_dir}/201_1A0F_A.cif", "w") as o:
    o.write("data_201_1A0F_A\n# Derived from RCSB 1A0F chain A (ATOM only); residue 100 (TYR) deleted on purpose.\n#\nloop_\n")
    o.write("\n".join(cols) + "\n" + "\n".join(kept) + "\n#\n")
# --- PDB: chain A ATOM records, residue DROP removed
pdb = [l for l in open(f"{src_dir}/1A0F.pdb") if l.startswith("ATOM") and l[21] == "A" and int(l[22:26]) != DROP]
with open(f"{out_dir}/201_1A0F_A.pdb", "w") as o:
    o.write("REMARK   Derived from RCSB 1A0F chain A (ATOM only); residue 100 (TYR) deleted on purpose.\n")
    o.writelines(pdb); o.write("TER\nEND\n")
print("cif atoms", len(kept), "pdb atoms", len(pdb))

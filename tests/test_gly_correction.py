"""
test_gly_correction.py  --  Diagnose GLY frustration correction

Checks four things:
  1. Whether _compute_gly_alt_densities actually produces different
     densities at GLY positions (rho_vcb != rho_native, rho_ca != rho_native).
  2. Whether _apply_gly_correction changes the energies at GLY positions.
  3. Whether the final FrstIndex at GLY positions matches TrueFrstIndex
     in a reference file (if provided).
  4. Whether mutation-scanning FrstIndex is correct at GLY positions,
     compared against a FrustratometeR MutationsData reference.
     Can load a pre-computed file (--mutations) or run from scratch (--pdb + --mut-ref).

Usage:
  # Just check the code is working (no reference needed):
  python tests/test_gly_correction.py --pdb path/to/protein.pdb

  # Full comparison against FrustratometeR reference output:
  python tests/test_gly_correction.py \\
      --pdb path/to/protein.pdb \\
      --ref path/to/protein.pdb_singleresidue \\
      --mode singleresidue

  # Mutation scanning: run code from PDB and compare to reference:
  python tests/test_gly_correction.py \\
      --pdb  path/to/protein.pdb \\
      --mut-ref path/to/protein.pdb.done/MutationsData

  # Mutation scanning: load pre-computed file and compare:
  python tests/test_gly_correction.py \\
      --mutations path/to/protein.pdb_frustratometer3.singleresidue_mutations \\
      --mut-ref   path/to/protein.pdb.done/MutationsData
"""
import argparse
import sys
import os

# Command-line diagnostic, not a pytest module: its test_* helpers take (model, batch, ...) arguments
# supplied by main(), not fixtures, so pytest must not collect them.
__test__ = False

import torch
import numpy as np
import pandas as pd

# Allow running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytorch_frustratometer as pf
from pytorch_frustratometer import FrustratometerDataset, collate_frustratometer

GLY_IDX = 5  # index of GLY in our alphabetical AA ordering

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_batch(pdb_path, model=None):
    from pytorch_frustratometer.coefficients import Coefficients
    cfg = Coefficients()
    dataset = FrustratometerDataset([pdb_path], cfg)
    batch = collate_frustratometer([dataset[0]])
    if model is not None and "residue_density" not in batch:
        with torch.no_grad():
            model.compute_densities(batch)
    return batch


def get_model(mode="singleresidue", save_mutations=False,
              burial_rmin=None, burial_rmax=None, burial_min_sep=None,
              seq_dist=None, no_electrostatics=False):
    from pytorch_frustratometer.coefficients import Coefficients
    cfg = Coefficients()
    if save_mutations:
        cfg.save_mutations = True
    if burial_rmin is not None:
        cfg.burial_rmin = burial_rmin
    if burial_rmax is not None:
        cfg.burial_rmax = burial_rmax
    if burial_min_sep is not None:
        cfg.burial_min_sep = burial_min_sep
    if seq_dist is not None:
        cfg.min_seq_distance = seq_dist
    if no_electrostatics:
        cfg.use_electrostatics = False
    from pytorch_frustratometer import Frustratometer
    return Frustratometer(mode, coefficients=cfg)


def read_ref(ref_path):
    return pd.read_csv(ref_path, sep=r"\s+", comment="#")


# ---------------------------------------------------------------------------
# Test 1: densities are actually different at GLY positions
# ---------------------------------------------------------------------------

def test_densities(model, batch):
    print("\n=== Test 1: GLY alternative densities ===")

    rho_native = batch["residue_density"][0]  # [L]
    aa         = batch["aa_type"][0]           # [L]
    gly_mask   = (aa == GLY_IDX)

    if gly_mask.sum() == 0:
        print("  WARNING: no GLY residues found in this protein.")
        return None, None

    rho_vcb, rho_ca, _vcb_hybrid = model._compute_gly_alt_densities(batch)

    if rho_vcb is None or rho_ca is None:
        print("  FAIL: _compute_gly_alt_densities returned None!")
        print("        The correction will be silently skipped.")
        return None, None

    rho_vcb_0 = rho_vcb[0]
    rho_ca_0  = rho_ca[0]

    gly_positions = torch.where(gly_mask)[0].tolist()
    print(f"  GLY positions (0-indexed): {gly_positions}")
    print(f"  {'Pos':>5}  {'rho_native':>12}  {'rho_vcb':>12}  {'delta_vcb':>12}")
    for pos in gly_positions:
        nat = rho_native[pos].item()
        vcb = rho_vcb_0[pos].item()
        print(f"  {pos:>5}  {nat:>12.4f}  {vcb:>12.4f}  {vcb - nat:>+12.4f}")

    non_gly_positions = torch.where(~gly_mask)[0].tolist()[:5]  # show first 5
    print(f"\n  Sample non-GLY positions (first 5):")
    print(f"  {'Pos':>5}  {'rho_native':>12}  {'rho_ca':>12}  {'delta_ca':>12}")
    for pos in non_gly_positions:
        nat = rho_native[pos].item()
        ca  = rho_ca_0[pos].item()
        print(f"  {pos:>5}  {nat:>12.4f}  {ca:>12.4f}  {ca - nat:>+12.4f}")

    vcb_changed = not torch.allclose(rho_vcb_0[gly_mask], rho_native[gly_mask])
    ca_changed  = not torch.allclose(rho_ca_0[~gly_mask], rho_native[~gly_mask])

    if vcb_changed:
        print("\n  PASS: rho_vcb differs from rho_native at GLY positions.")
    else:
        print("\n  FAIL: rho_vcb == rho_native at GLY positions — correction has no effect!")

    if ca_changed:
        print("  PASS: rho_ca differs from rho_native at non-GLY positions.")
    else:
        print("  FAIL: rho_ca == rho_native at non-GLY positions — correction has no effect!")

    return rho_vcb, rho_ca


# ---------------------------------------------------------------------------
# Test 2: energies change after GLY correction
# ---------------------------------------------------------------------------

def test_energy_change(model, batch, rho_vcb, rho_ca):
    """
    Show FI at GLY positions with and without the GLY correction.

    FrustratometeR uses fixed native density for all decoys, which
    corresponds to the 'without correction' column below.
    """
    print("\n=== Test 2: energy change after _apply_gly_correction ===")
    if rho_vcb is None:
        print("  SKIP: densities are None.")
        return

    aa    = batch["aa_type"][0]

    with torch.no_grad():
        native_energy = model.compute_singleresidue_native_ixn(batch, batch["aa_type"])
        stats_without, _, _ = model.compute_singleresidue_stats_fast(
            batch, native_energy, rho_vcb=None, rho_ca=None
        )
        stats_with, _, _ = model.compute_singleresidue_stats_fast(
            batch, native_energy, rho_vcb=rho_vcb, rho_ca=rho_ca
        )

    gly_mask = (aa == GLY_IDX)
    gly_pos  = torch.where(gly_mask)[0]

    # stats: [B, L, 2]  where [..., 0]=mean, [..., 1]=std
    fi_without = model.compute_frustration_index(native_energy[0], stats_without[0])
    fi_with    = model.compute_frustration_index(native_energy[0], stats_with[0])

    print(f"  {'Pos':>5}  {'FI_without':>12}  {'FI_with':>10}  {'delta':>10}")
    print(f"  {'':5}  {'(native density)':>12}  {'(corrected)':>10}")
    for pos in gly_pos.tolist():
        fw_no = fi_without[pos].item()
        fw    = fi_with[pos].item()
        print(f"  {pos:>5}  {fw_no:>12.4f}  {fw:>10.4f}  {fw - fw_no:>+10.4f}")

    if torch.allclose(fi_without[gly_mask], fi_with[gly_mask], atol=1e-4):
        print("\n  NOTE: FI values unchanged at GLY positions — correction has no effect.")
    else:
        print("\n  Correction changes FI at GLY positions.")
        print("  FI_without uses native density (matches FrustratometeR's decoy approach).")
        print("  FI_with uses virtual Cβ for non-GLY decoys (more accurate physics, "
              "but differs from FrustratometeR).")


# ---------------------------------------------------------------------------
# Test 3: compare to FrustratometeR reference
# ---------------------------------------------------------------------------

def test_vs_reference(model, batch, ref_path, mode):
    """
    Compare our FrstIndex against a FrustratometeR reference file.

    The reference file uses 'FrstIndex' (not 'TrueFrstIndex').  We also
    show what the FI would be WITHOUT the GLY correction (rho_vcb=None)
    so you can see whether the correction is helping or hurting.
    """
    print(f"\n=== Test 3: comparison to FrustratometeR reference ({mode}) ===")

    ref = read_ref(ref_path)
    print(f"  Reference columns: {list(ref.columns)}")

    # Accept either column name
    ref_fi_col = None
    for cname in ("FrstIndex", "TrueFrstIndex"):
        if cname in ref.columns:
            ref_fi_col = cname
            break
    if ref_fi_col is None:
        print("  SKIP: no FrstIndex or TrueFrstIndex column in reference.")
        return

    aa_seq = batch["aa_type"][0]  # [L]

    if mode != "singleresidue":
        print("  (Test 3 single-residue comparison only; skipping pair modes)")
        return

    ref_fi   = pd.to_numeric(ref[ref_fi_col], errors="coerce")
    res_nums = ref["Res"].values
    aa_col   = ref["AA"].values if "AA" in ref.columns else None

    # --- run model twice: with and without GLY correction ---
    with torch.no_grad():
        native_energy = model.compute_singleresidue_native_ixn(batch, batch["aa_type"])

        rho_vcb, rho_ca, vcb_hybrid = model._compute_gly_alt_densities(batch)

        stats_with, _, _    = model.compute_singleresidue_stats_fast(batch, native_energy,
                                                                     rho_vcb=rho_vcb, rho_ca=rho_ca,
                                                                     vcb=vcb_hybrid)
        stats_without, _, _ = model.compute_singleresidue_stats_fast(batch, native_energy,
                                                                     rho_vcb=None, rho_ca=None)

    fi_with    = model.compute_frustration_index(native_energy[0], stats_with[0]).cpu().numpy()
    fi_without = model.compute_frustration_index(native_energy[0], stats_without[0]).cpu().numpy()

    # Build comparison table aligned to reference rows
    gly_errs_with, gly_errs_without, nongly_errs_with, nongly_errs_without = [], [], [], []
    n_shown = 0
    print(f"\n  Positions with |error_with_correction| > 0.1 OR |error_without| > 0.1:")
    print(f"  {'Res':>5}  {'AA':>3}  {'GLY?':>5}  {'FI_with':>10}  {'FI_without':>12}  {'FI_ref':>9}  "
          f"{'err_with':>10}  {'err_no':>8}")

    for idx in range(len(ref)):
        fi_r = ref_fi.iloc[idx]
        if pd.isna(fi_r):
            continue
        res = res_nums[idx]
        aa_str = aa_col[idx] if aa_col is not None else "?"
        is_gly = (aa_str == "G")

        # match to our position by row index (assumes same ordering)
        if idx >= len(fi_with):
            break
        fw = fi_with[idx]
        fw_no = fi_without[idx]
        err_w  = abs(fw - fi_r)
        err_no = abs(fw_no - fi_r)

        if is_gly:
            gly_errs_with.append(err_w)
            gly_errs_without.append(err_no)
        else:
            nongly_errs_with.append(err_w)
            nongly_errs_without.append(err_no)

        if err_w > 0.1 or err_no > 0.1:
            print(f"  {res:>5}  {aa_str:>3}  {'GLY' if is_gly else '':>5}  "
                  f"{fw:>10.4f}  {fw_no:>12.4f}  {fi_r:>9.4f}  "
                  f"{err_w:>10.4f}  {err_no:>8.4f}")
            n_shown += 1

    if n_shown == 0:
        print("  PASS: no positions with error > 0.1 for either variant.")

    def _stats(errs):
        if not errs:
            return "n=0"
        return f"mean={np.mean(errs):.4f}  max={np.max(errs):.4f}  n={len(errs)}"

    print(f"\n  Summary:")
    print(f"  {'':25}  {'with correction':>16}  {'without correction':>18}")
    print(f"  {'GLY positions':25}  {_stats(gly_errs_with):>16}  {_stats(gly_errs_without):>18}")
    print(f"  {'non-GLY positions':25}  {_stats(nongly_errs_with):>16}  {_stats(nongly_errs_without):>18}")

    if gly_errs_with and gly_errs_without:
        if np.mean(gly_errs_without) < np.mean(gly_errs_with):
            print("\n  DIAGNOSIS: GLY errors are SMALLER without the correction.")
            print("  The correction diverges from FrustratometeR (which uses native density for all decoys).")
            print("  Consider removing _apply_gly_correction from compute_singleresidue_stats_fast.")
        else:
            print("\n  GLY errors are smaller WITH the correction — correction is helping.")


# ---------------------------------------------------------------------------
# Test 5: energy breakdown diagnostic — pinpoint the discrepancy
# ---------------------------------------------------------------------------

def test_energy_breakdown(model, batch, pos_1indexed, ref_fi_by_mut=None, ref_fi_by_mut2=None):
    """
    For a specific residue position (1-indexed), print the per-AA energy
    breakdown (burial, water, elec, total) and computed FI.

    ref_fi_by_mut: optional dict {aa_1letter: fi_value} from FrustratometeR.
    """
    print(f"\n=== Energy breakdown at position {pos_1indexed} ===")

    from pytorch_frustratometer.structure.residue_constants import ID_TO_AA
    AA_TO_ID = {v: k for k, v in ID_TO_AA.items()}

    B, L = batch["aa_type"].shape
    device = batch["aa_type"].device

    # Resolve 1-indexed residue number to 0-indexed batch position
    residue_index = batch.get("residue_index")
    if residue_index is not None:
        ri = residue_index[0].cpu().tolist()
        ri_to_pos0 = {int(r): i for i, r in enumerate(ri)}
        pos0 = ri_to_pos0.get(pos_1indexed)
        if pos0 is None:
            print(f"  ERROR: residue {pos_1indexed} not found in residue_index.")
            return
    else:
        pos0 = pos_1indexed - 1
    if pos0 < 0 or pos0 >= L:
        print(f"  ERROR: position {pos_1indexed} out of range [1, {L}]")
        return

    native_aa_idx = int(batch["aa_type"][0, pos0])
    native_aa = ID_TO_AA.get(native_aa_idx, "?")
    rho_pos = float(batch["residue_density"][0, pos0])
    print(f"  Native AA: {native_aa} (idx {native_aa_idx}), rho_native = {rho_pos:.4f}")

    # Count contacts
    edge_sym = torch.cat([batch["edge_index"], batch["edge_index"].flip(1)], dim=-1)
    src = edge_sym[0, 0, :]
    valid = (src >= 0)
    contacts = (src[valid] == pos0).sum().item()
    print(f"  Contacts (symmetric): {contacts}")

    with torch.no_grad():
        native_energy = model.compute_singleresidue_native_ixn(batch, batch["aa_type"])

        rho_vcb, rho_ca, vcb_hybrid = model._compute_gly_alt_densities(batch)
        stats_native, E_decoy_native, _ = model.compute_singleresidue_stats_fast(
            batch, native_energy, rho_vcb=None, rho_ca=None
        )
        stats_corr, E_decoy_corr, _ = model.compute_singleresidue_stats_fast(
            batch, native_energy, rho_vcb=rho_vcb, rho_ca=rho_ca, vcb=vcb_hybrid
        )
        # Production path: same as save_mutations=True (includes gly_water_corr)
        stats_prod, E_decoy_prod, _ = model.compute_singleresidue_stats_fast(
            batch, native_energy, rho_vcb=rho_vcb, rho_ca=rho_ca,
            vcb=vcb_hybrid, gly_water_corr=True
        )

        # Get energy components separately
        rho = batch["residue_density"]
        all20 = torch.arange(20, device=device)
        rho_exp = rho.unsqueeze(0).expand(20, B, L)
        all20_exp = all20.view(20, 1, 1).expand(20, B, L)

        burial_all = model.compute_burial_energy(batch, all20_exp, rho_exp)  # [20, B, L]
        burial_nat = model.compute_burial_energy(
            batch, batch["aa_type"].unsqueeze(0), rho.unsqueeze(0)
        )  # [1, B, L]
        delta_burial = burial_all - burial_nat  # [20, B, L]

        field = model.compute_sr_elec_field(batch)
        q_all = model.charge_lookup[all20].to(model.dtype)
        q_nat = model.charge_lookup[batch["aa_type"]].to(model.dtype)
        delta_q = q_all.view(20, 1, 1) - q_nat.unsqueeze(0)
        delta_elec = delta_q * field.unsqueeze(0)  # [20, B, L]

        delta_water = model.compute_sr_water_delta(batch, batch["aa_type"], rho)  # [20, B, L]

    nat_e_pos = float(native_energy[0, pos0])
    d_mean = float(stats_native[0, pos0, 0])
    d_std  = float(stats_native[0, pos0, 1])
    d_mean_c = float(stats_corr[0, pos0, 0])
    d_std_c  = float(stats_corr[0, pos0, 1])
    d_mean_p = float(stats_prod[0, pos0, 0])
    d_std_p  = float(stats_prod[0, pos0, 1])

    # Stats without electrostatics — zero out delta_elec for all 20 AAs at this pos
    # to see how much the normalization shifts when elec is removed.
    freq_all = torch.zeros(20, device=device, dtype=model.dtype)
    valid_mask = batch["seq_mask"].bool()[0]
    aa_native = batch["aa_type"][0]
    seq_len = valid_mask.float().sum().clamp(min=1)
    for a in range(20):
        freq_all[a] = ((aa_native == a) & valid_mask).float().sum() / seq_len
    # E_decoy without elec at this position: E_native + dBurial + dWater
    e_no_elec_pos = torch.stack([
        torch.tensor(nat_e_pos + float(delta_burial[x, 0, pos0]) + float(delta_water[x, 0, pos0]),
                     dtype=model.dtype)
        for x in range(20)
    ])
    mean_no_elec = (freq_all * e_no_elec_pos).sum().item()
    std_no_elec  = ((freq_all * (e_no_elec_pos - mean_no_elec) ** 2).sum()
                    .clamp(min=0).sqrt().item())

    elec_field_pos = float(field[0, pos0])
    print(f"\n  Native energy at pos {pos_1indexed}: {nat_e_pos:.4f}")
    print(f"  d_mean (no corr):          {d_mean:.4f}    d_std: {d_std:.4f}")
    print(f"  d_mean (burial corr only): {d_mean_c:.4f}  d_std: {d_std_c:.4f}")
    print(f"  d_mean (prod/gly_water):   {d_mean_p:.4f}  d_std: {d_std_p:.4f}  [used by mutation scanning]")
    print(f"  d_mean (no elec):          {mean_no_elec:.4f}  d_std: {std_no_elec:.4f}")

    # Show which charged neighbors contribute to the field at this position.
    # Electrostatics uses ALL non-self pairs (no sequence-separation filter).
    print(f"\n  Electrostatic field at pos {pos_1indexed}: {elec_field_pos:.6f}")
    if model.coefficients.use_electrostatics:
        from pytorch_frustratometer.structure.residue_constants import ID_TO_AA as _ID_TO_AA
        ri_all      = batch["residue_index"][0]   # [L]
        aa_all      = batch["aa_type"][0]          # [L]
        dist_row    = batch["distance"][0, pos0].to(model.dtype)   # [L]
        seq_mask_0  = batch["seq_mask"][0].bool()  # [L]
        q_j         = model.charge_lookup[aa_all].to(model.dtype)
        screening_l = float(model.coefficients.screening_length)
        k_e         = float(model.coefficients.k_PlusMinus)

        charged_rows = []
        for j in range(L):
            if j == pos0:
                continue
            if not seq_mask_0[j]:
                continue
            qj = float(q_j[j])
            if qj == 0:
                continue
            d = float(dist_row[j])
            if d < 1e-6:
                continue
            contrib = k_e * qj / d * float(torch.exp(dist_row[j] / -screening_l))
            charged_rows.append((int(ri_all[j]), _ID_TO_AA.get(int(aa_all[j]), "?"), qj, d, contrib))

        if charged_rows:
            print(f"  Charged residues contributing to field:")
            print(f"  {'Res':>5}  {'AA':>3}  {'q':>4}  {'dist_CB':>9}  {'contrib':>12}")
            for rj, aaj, qj, d, c in sorted(charged_rows, key=lambda x: abs(x[4]), reverse=True):
                print(f"  {rj:>5}  {aaj:>3}  {qj:>+4.0f}  {d:>9.3f}  {c:>12.6f}")
            total_field_check = sum(c for *_, c in charged_rows)
            print(f"  Sum of contribs: {total_field_check:.6f}  (model.compute_sr_elec_field: {elec_field_pos:.6f})")
        else:
            print(f"  No charged residues with valid seq_mask.")

    print(f"\n  Per-AA breakdown at position {pos_1indexed}:")
    hdr = (f"  {'AA':>3}  {'dBurial':>10}  {'dWater':>10}  {'dElec':>10}  "
           f"{'E_prod':>10}  {'FI_no':>8}  {'FI_prod':>8}  {'FI_noE':>7}")
    if ref_fi_by_mut:
        hdr += f"  {'FI_FR1':>8}  {'err1':>6}  {'errP1':>6}"
    if ref_fi_by_mut2:
        hdr += f"  {'FI_FR2':>8}  {'err2':>6}  {'errP2':>6}"
    print(hdr)

    for x_idx in range(20):
        aa_x = ID_TO_AA.get(x_idx, "?")
        db = float(delta_burial[x_idx, 0, pos0])
        dw = float(delta_water[x_idx, 0, pos0])
        de = float(delta_elec[x_idx, 0, pos0])
        e_x = nat_e_pos + db + dw + de
        e_x_no_elec = nat_e_pos + db + dw
        # Production E_decoy uses E_decoy_prod from gly_water_corr=True stats
        e_x_prod = float(E_decoy_prod[x_idx, 0, pos0])
        fi_no    = (d_mean - e_x) / max(d_std, 1e-8)
        fi_prod  = (d_mean_p - e_x_prod) / max(d_std_p, 1e-8)
        fi_no_e  = (mean_no_elec - e_x_no_elec) / max(std_no_elec, 1e-8)

        ref_str = ""
        if ref_fi_by_mut and aa_x in ref_fi_by_mut:
            fi_fr    = ref_fi_by_mut[aa_x]
            ref_str += f"  {fi_fr:>8.3f}  {abs(fi_no-fi_fr):>6.3f}  {abs(fi_prod-fi_fr):>6.3f}"
        elif ref_fi_by_mut:
            ref_str += f"  {'':>8}  {'':>6}  {'':>6}"
        if ref_fi_by_mut2 and aa_x in ref_fi_by_mut2:
            fi_fr2   = ref_fi_by_mut2[aa_x]
            ref_str += f"  {fi_fr2:>8.3f}  {abs(fi_no-fi_fr2):>6.3f}  {abs(fi_prod-fi_fr2):>6.3f}"

        print(f"  {aa_x:>3}  {db:>10.4f}  {dw:>10.4f}  {de:>10.4f}  "
              f"{e_x_prod:>10.4f}  {fi_no:>8.3f}  {fi_prod:>8.3f}  {fi_no_e:>7.3f}{ref_str}")


# ---------------------------------------------------------------------------
# Test 4: mutation scanning — run from PDB and compare
# ---------------------------------------------------------------------------

def run_mutations_from_pdb(model, batch):
    """
    Run mutation scanning on batch and return a DataFrame with columns:
      Res, ChainRes, AA_native, AA_mutant, NativeEnergy, MutantEnergy, FrstIndex

    The model must have save_mutations=True (set via coefficients).
    """
    from pytorch_frustratometer.structure.residue_constants import ID_TO_AA

    with torch.no_grad():
        fi, nat_e, stats, _, _, sr_mutations = model.compute_tert_frust_singleresidue(batch)

    if sr_mutations is None:
        raise RuntimeError(
            "sr_mutations is None — did you set save_mutations=True on the model? "
            "Pass save_mutations=True to get_model()."
        )

    # Extract residue numbering and chain IDs from batch
    mask = batch["seq_mask"][0].bool()

    residue_index = batch.get("residue_index")
    if residue_index is not None:
        res_nums = residue_index[0, mask].cpu().numpy()
    else:
        res_nums = np.arange(1, mask.sum().item() + 1)

    chain_index = batch.get("chain_index")
    chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if chain_index is not None:
        chain_ids = [chain_letters[min(int(c), 25)]
                     for c in chain_index[0, mask].cpu().tolist()]
    else:
        chain_ids = ["A"] * mask.sum().item()

    aa = batch["aa_type"][0, mask].cpu()

    mut_e    = sr_mutations["energies"][0, mask].cpu().float()   # [n_res, 20]
    nat_e_sr = nat_e[0, mask].cpu().float()                      # [n_res]
    fi_masked = fi[0, mask].cpu().float()                        # [n_res] native FI

    mut_stats = sr_mutations.get("stats")
    if mut_stats is not None:
        ds     = mut_stats[0, mask].cpu().float()
        d_mean = ds[:, 0]
        d_std  = ds[:, 1]
    else:
        d_mean = d_std = None

    # Case 2: corrected FI for GLY decoy at non-GLY positions
    gly_fi_masked = None
    gly_fi_all = sr_mutations.get("_gly_mutation_fi")
    if gly_fi_all is not None:
        gly_fi_masked = gly_fi_all[0, mask].cpu().float()

    # Case 1 (native GLY → non-GLY): vCB mini-batch FI from _compute_gly_mutant_fi.
    # G→G (native=GLY, mutant=GLY) uses native FI directly (fi_masked) since the mutant
    # structure is identical to native and the reference runs it as-is.
    gly_native_fi_masked = None
    gly_native_fi_all = sr_mutations.get("_gly_native_fi")
    if gly_native_fi_all is not None:
        gly_native_fi_masked = gly_native_fi_all[0, mask].cpu().float()

    n_res = mut_e.shape[0]
    rows = []
    for r in range(n_res):
        aa_nat = ID_TO_AA.get(int(aa[r]), "X")
        is_nat_gly = (int(aa[r]) == GLY_IDX)
        for d in range(20):
            if is_nat_gly and d == GLY_IDX:
                # G→G: mutant structure = native; FI = native FI at that position
                fi_val = float(fi_masked[r])
            elif is_nat_gly and gly_native_fi_masked is not None:
                fi_val = float(gly_native_fi_masked[r, d])
            elif (d == GLY_IDX and not is_nat_gly
                    and gly_fi_masked is not None
                    and not np.isnan(float(gly_fi_masked[r]))):
                fi_val = float(gly_fi_masked[r])
            else:
                fi_val = (float((d_mean[r] - mut_e[r, d]) / d_std[r].clamp(min=1e-8))
                          if d_mean is not None else float("nan"))
            rows.append({
                "Res":          int(res_nums[r]),
                "ChainRes":     chain_ids[r],
                "AA_native":    aa_nat,
                "AA_mutant":    ID_TO_AA.get(d, "X"),
                "NativeEnergy": float(nat_e_sr[r]),
                "MutantEnergy": float(mut_e[r, d]),
                "FrstIndex":    fi_val,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Alternative Test 4b: per-mutant AWSEM energies via _compute_gly_mutant_fi
# ---------------------------------------------------------------------------

def run_mutations_mutant_fi(model, batch):
    """
    Like run_mutations_from_pdb but uses _compute_gly_mutant_fi for Case 1
    (native GLY → any decoy).  The existing gly_water_corr stats are used for
    all other mutations (Case 2 and non-GLY).

    Prints a side-by-side comparison with gly_water_corr to isolate the effect
    of the per-mutant geometry for Case 1 positions.
    """
    from pytorch_frustratometer.structure.residue_constants import ID_TO_AA

    with torch.no_grad():
        fi, nat_e, stats, _, _, sr_mutations = model.compute_tert_frust_singleresidue(batch)

    if sr_mutations is None:
        raise RuntimeError("sr_mutations is None — pass save_mutations=True to get_model()")

    mask = batch["seq_mask"][0].bool()

    residue_index = batch.get("residue_index")
    res_nums = (residue_index[0, mask].cpu().numpy() if residue_index is not None
                else np.arange(1, mask.sum().item() + 1))

    chain_index = batch.get("chain_index")
    chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chain_ids = ([chain_letters[min(int(c), 25)] for c in chain_index[0, mask].cpu().tolist()]
                 if chain_index is not None else ["A"] * mask.sum().item())

    aa = batch["aa_type"][0, mask].cpu()
    mut_e    = sr_mutations["energies"][0, mask].cpu().float()   # [n_res, 20]
    nat_e_sr = nat_e[0, mask].cpu().float()

    mut_stats = sr_mutations.get("stats")
    d_mean = d_std = None
    if mut_stats is not None:
        ds     = mut_stats[0, mask].cpu().float()
        d_mean = ds[:, 0]
        d_std  = ds[:, 1]

    gly_fi_masked = None
    gly_fi_all = sr_mutations.get("_gly_mutation_fi")
    if gly_fi_all is not None:
        gly_fi_masked = gly_fi_all[0, mask].cpu().float()

    # Compute per-mutant FI for Case 1
    with torch.no_grad():
        rho_vcb, rho_ca, _ = model._compute_gly_alt_densities(batch)
        # native_energy is needed by the function signature but not used for absolute path
        native_energy_dummy = torch.zeros(
            batch["aa_type"].shape, device=batch["aa_type"].device, dtype=model.dtype
        )
        fi_mutant_all = model._compute_gly_mutant_fi(
            batch, native_energy_dummy, rho_vcb, rho_ca
        )
    fi_mutant_masked = fi_mutant_all[0, mask].cpu().float()  # [n_res, 20], NaN at non-GLY

    rows = []
    for r in range(mut_e.shape[0]):
        aa_nat = ID_TO_AA.get(int(aa[r]), "X")
        is_nat_gly = (int(aa[r]) == GLY_IDX)
        for d in range(20):
            # Case 1: use _compute_gly_mutant_fi
            _mf = float(fi_mutant_masked[r, d])
            if is_nat_gly and _mf == _mf:  # not NaN
                fi_val = _mf
            elif (d == GLY_IDX and not is_nat_gly
                    and gly_fi_masked is not None
                    and not np.isnan(float(gly_fi_masked[r]))):
                fi_val = float(gly_fi_masked[r])
            else:
                fi_val = (float((d_mean[r] - mut_e[r, d]) / d_std[r].clamp(min=1e-8))
                          if d_mean is not None else float("nan"))
            rows.append({
                "Res":          int(res_nums[r]),
                "ChainRes":     chain_ids[r],
                "AA_native":    aa_nat,
                "AA_mutant":    ID_TO_AA.get(d, "X"),
                "NativeEnergy": float(nat_e_sr[r]),
                "MutantEnergy": float(mut_e[r, d]),
                "FrstIndex":    fi_val,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 4c: exact X→G FI — each GLY mutant run as its own mini-batch
# ---------------------------------------------------------------------------

def compute_exact_gly_decoy_fi(model, batch):
    """
    Compute X→G (non-GLY → GLY decoy) FI values exactly, matching
    FrustratometeR's mutant_test_2 approach of running each GLY mutant
    structure as a separate batch item.

    For each non-GLY position q a Q-element mini-batch is built where:
      - aa_type[q] = GLY_IDX
      - centroid[q] is set to CA_q (not CB_q)
      - distance matrix is recomputed from the new centroids
      - residue_density is recomputed with NO sequence-separation filter

    Returns a [B, L] tensor with the exact FI at each non-GLY position
    (NaN at GLY / masked positions).
    """
    from pytorch_frustratometer.structure.geometry import get_CB_coords

    B, L = batch["aa_type"].shape
    device = batch["aa_type"].device
    GLY_IDX = 5

    is_gly     = (batch["aa_type"] == GLY_IDX)
    valid_mask = batch["seq_mask"].bool() & ~batch["hetatm"].bool()

    cfg  = model.coefficients
    eta  = float(cfg.eta)
    rmin = float(cfg.well_rmin[0])
    rmax = float(cfg.well_rmax[0])

    fi_exact = torch.full((B, L), float("nan"), device=device, dtype=model.dtype)

    for b in range(B):
        non_gly_idx = torch.where(~is_gly[b] & valid_mask[b])[0]
        Q = non_gly_idx.numel()
        if Q == 0:
            continue

        # native CB/CA centroid for each residue
        native_centroid = get_CB_coords(
            batch["atom_positions"][b:b+1, :, :5],
            batch["aa_type"][b:b+1],
        ).to(model.dtype)[0]  # [L, 3]

        CA_b = batch["atom_positions"][b, :, 1].to(model.dtype)  # [L, 3]

        # Q centroids: CA at mutation site, native centroid elsewhere
        centroids = native_centroid.unsqueeze(0).expand(Q, -1, -1).clone()  # [Q, L, 3]
        centroids[torch.arange(Q, device=device), non_gly_idx] = CA_b[non_gly_idx]

        # Full pairwise distance matrix for each of the Q structures [Q, L, L]
        dist_q = torch.cdist(centroids.float(), centroids.float()).to(model.dtype)

        # Density: replicate _density_intermediates in feature_calculator exactly.
        # Uses residue_index-based separation (sep <= min_seq_distance excluded)
        # AND neighbor_cutoff, matching what preprocessing computes for the GLY mutant.
        vm = valid_mask[b]  # [L] bool
        ri_b = batch["residue_index"][b].long()  # [L] PDB residue numbers
        min_sep_d = int(cfg.min_seq_distance)    # = 3
        cutoff_d  = float(getattr(cfg, 'tert_frust_cutoff', 9.5))

        sep_mat     = (ri_b.unsqueeze(1) - ri_b.unsqueeze(0)).abs()  # [L, L]
        excl        = sep_mat <= min_sep_d                            # [L, L]
        chain_idx_b = batch.get("chain_index")
        ci_b        = None
        if chain_idx_b is not None:
            ci_b = chain_idx_b[b] if chain_idx_b.dim() == 2 else chain_idx_b
            excl = excl & (ci_b.unsqueeze(1) == ci_b.unsqueeze(0))

        within_cutoff = (dist_q <= cutoff_d)                     # [Q, L, L]
        pair_vm       = (vm.unsqueeze(0) & vm.unsqueeze(1))      # [L, L]
        density_mask  = (~excl & pair_vm).unsqueeze(0) & within_cutoff  # [Q, L, L]

        vt = (0.25
              * (1 + torch.tanh(eta * (dist_q - rmin)))
              * (1 + torch.tanh(eta * (rmax - dist_q))))  # [Q, L, L]
        rho_q = (vt * density_mask.to(model.dtype)).sum(-1)  # [Q, L]

        # Recompute edge_index from dist_q — native CB-based edge_index is wrong for
        # GLY mutants because CB→CA shifts contacts in/out of the 9.5 Å cutoff.
        # min_sep=2 matches calculate_distance_matrix default used in preprocessing
        # (feature_calculator calls it without min_seq_distance, so uses 2 not cfg=3).
        iu_full  = torch.triu_indices(L, L, offset=1, device=device)
        sep_mask = (iu_full[0] - iu_full[1]).abs() >= 2  # pos-index sep, matches calculate_distance_matrix
        if ci_b is not None:
            sep_mask = sep_mask | (ci_b[iu_full[0]] != ci_b[iu_full[1]])
        iu_f = iu_full[:, sep_mask]                                # [2, n_pairs]
        pair_vm = valid_mask[b][iu_f[0]] & valid_mask[b][iu_f[1]] # [n_pairs]

        iu_dist_q  = dist_q[:, iu_f[0], iu_f[1]]                      # [Q, n_pairs]
        close_mask = (iu_dist_q < cutoff_d) & pair_vm.unsqueeze(0)   # [Q, n_pairs]

        E_max = int(close_mask.sum(dim=1).max()) if close_mask.any() else 1
        close_iu = torch.full((Q, 2, E_max), -1, device=device, dtype=torch.long)
        idx = (close_mask.cumsum(dim=1) - 1).masked_fill(~close_mask, -1)
        iu_exp  = iu_f.unsqueeze(0).expand(Q, -1, -1)
        q_idx, pair_idx = torch.where(close_mask)
        close_iu[q_idx, :, idx[q_idx, pair_idx]] = iu_exp[q_idx, :, pair_idx]

        # Build Q-element mini-batch by expanding the single structure b
        gly_batch = {}
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor) or v.dim() == 0 or v.shape[0] != B:
                gly_batch[k] = v
            else:
                gly_batch[k] = v[b:b+1].expand(Q, *v.shape[1:]).clone()

        # Patch mutation-specific fields
        aa_q = batch["aa_type"][b].unsqueeze(0).expand(Q, -1).clone()
        aa_q[torch.arange(Q, device=device), non_gly_idx] = GLY_IDX
        gly_batch["aa_type"]         = aa_q
        gly_batch["distance"]        = dist_q
        gly_batch["residue_density"] = rho_q
        gly_batch["edge_index"]      = close_iu
        if "burial_density" in batch:
            gly_batch["burial_density"] = rho_q

        # Native energy of GLY at each mutation site
        with torch.no_grad():
            ne_gly = model.compute_singleresidue_native_ixn(gly_batch, aa_q)  # [Q, L]

        # Decoy stats — gly_water_corr=False: GLY positions use all-CA naturally
        with torch.no_grad():
            stats, _, _ = model.compute_singleresidue_stats_fast(
                gly_batch, ne_gly, gly_water_corr=False,
            )  # [Q, L, 2]

        # FI for each structure at its mutation site
        fi_q = model.compute_frustration_index(ne_gly, stats)  # [Q, L]
        fi_exact[b, non_gly_idx] = fi_q[torch.arange(Q, device=device), non_gly_idx]

    return fi_exact


def compute_exact_gly_native_fi(model, batch):
    """
    Compute G→X (GLY native → any AA decoy) FI for all 20 mutants at each GLY position,
    matching FrustratometeR's mutant_test_2 approach of running each non-GLY mutant
    structure as a separate batch item.

    For each GLY position q a mini-batch is built where aa_type[q] = ALA (any
    non-GLY placeholder).  Because vCB placement depends only on backbone geometry
    (not the side-chain type), the centroid and distance/density geometry is the same
    for all 19 non-GLY mutants at q.  The decoy scan therefore produces the correct
    E_decoy[x, q] and FI(x, q) for all x simultaneously.

    Returns [B, L, 20] tensor: fi[b, q, x].  NaN at non-GLY positions.
    """
    from pytorch_frustratometer.structure.geometry import get_CB_coords

    B, L = batch["aa_type"].shape
    device = batch["aa_type"].device
    GLY_IDX = 5
    ALA_IDX = 0  # vCB geometry is identical for all non-GLY types

    is_gly     = (batch["aa_type"] == GLY_IDX)
    valid_mask = batch["seq_mask"].bool() & ~batch["hetatm"].bool()

    cfg       = model.coefficients
    eta       = float(cfg.eta)
    rmin      = float(cfg.well_rmin[0])
    rmax      = float(cfg.well_rmax[0])
    min_sep_d = int(cfg.min_seq_distance)
    cutoff_d  = float(getattr(cfg, 'tert_frust_cutoff', 9.5))

    fi_gly_x = torch.full((B, L, 20), float("nan"), device=device, dtype=model.dtype)

    for b in range(B):
        gly_pos = torch.where(is_gly[b] & valid_mask[b])[0]
        Q = gly_pos.numel()
        if Q == 0:
            continue

        # Structure q: ALA (non-GLY placeholder) at gly_pos[q], native elsewhere.
        # get_CB_coords sees ALA + missing CB (zero slot for GLY) → returns ideal vCB.
        aa_q = batch["aa_type"][b].unsqueeze(0).expand(Q, -1).clone()  # [Q, L]
        aa_q[torch.arange(Q, device=device), gly_pos] = ALA_IDX

        atom_pos_q = batch["atom_positions"][b:b+1, :, :5].expand(Q, -1, -1, -1)
        centroids  = get_CB_coords(atom_pos_q, aa_q).to(model.dtype)  # [Q, L, 3]

        dist_q = torch.cdist(centroids.float(), centroids.float()).to(model.dtype)

        # Density: match _density_intermediates (residue_index sep, 9.5 Å cutoff)
        vm   = valid_mask[b]
        ri_b = batch["residue_index"][b].long()

        sep_mat     = (ri_b.unsqueeze(1) - ri_b.unsqueeze(0)).abs()
        excl        = sep_mat <= min_sep_d
        chain_idx_b = batch.get("chain_index")
        ci_b        = None
        if chain_idx_b is not None:
            ci_b = chain_idx_b[b] if chain_idx_b.dim() == 2 else chain_idx_b
            excl = excl & (ci_b.unsqueeze(1) == ci_b.unsqueeze(0))

        within_cutoff = (dist_q <= cutoff_d)
        pair_vm       = (vm.unsqueeze(0) & vm.unsqueeze(1))
        density_mask  = (~excl & pair_vm).unsqueeze(0) & within_cutoff

        vt    = (0.25
                 * (1 + torch.tanh(eta * (dist_q - rmin)))
                 * (1 + torch.tanh(eta * (rmax - dist_q))))
        rho_q = (vt * density_mask.to(model.dtype)).sum(-1)  # [Q, L]

        # Edge_index (position sep ≥ 2, matches calculate_distance_matrix default)
        iu_full  = torch.triu_indices(L, L, offset=1, device=device)
        sep_mask = (iu_full[0] - iu_full[1]).abs() >= 2
        if ci_b is not None:
            sep_mask = sep_mask | (ci_b[iu_full[0]] != ci_b[iu_full[1]])
        iu_f = iu_full[:, sep_mask]
        pair_vm_ei = valid_mask[b][iu_f[0]] & valid_mask[b][iu_f[1]]

        iu_dist_q  = dist_q[:, iu_f[0], iu_f[1]]
        close_mask = (iu_dist_q < cutoff_d) & pair_vm_ei.unsqueeze(0)

        E_max    = int(close_mask.sum(dim=1).max()) if close_mask.any() else 1
        close_iu = torch.full((Q, 2, E_max), -1, device=device, dtype=torch.long)
        idx      = (close_mask.cumsum(dim=1) - 1).masked_fill(~close_mask, -1)
        iu_exp   = iu_f.unsqueeze(0).expand(Q, -1, -1)
        q_ei, pair_ei = torch.where(close_mask)
        close_iu[q_ei, :, idx[q_ei, pair_ei]] = iu_exp[q_ei, :, pair_ei]

        # Build Q-element mini-batch
        gly_batch = {}
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor) or v.dim() == 0 or v.shape[0] != B:
                gly_batch[k] = v
            else:
                gly_batch[k] = v[b:b+1].expand(Q, *v.shape[1:]).clone()

        gly_batch["aa_type"]         = aa_q
        gly_batch["distance"]        = dist_q
        gly_batch["residue_density"] = rho_q
        gly_batch["edge_index"]      = close_iu
        if "burial_density" in batch:
            gly_batch["burial_density"] = rho_q

        with torch.no_grad():
            ne_q = model.compute_singleresidue_native_ixn(gly_batch, aa_q)  # [Q, L]
            stats, E_decoy, _ = model.compute_singleresidue_stats_fast(
                gly_batch, ne_q, gly_water_corr=False,
            )  # stats [Q, L, 2],  E_decoy [20, Q, L]

        # FI for each (GLY position q, mutant type x)
        q_range = torch.arange(Q, device=device)
        mean_d  = stats[q_range, gly_pos, 0]                  # [Q]
        std_d   = stats[q_range, gly_pos, 1].clamp(min=1e-8)  # [Q]
        E_x     = E_decoy[:, q_range, gly_pos]                 # [20, Q]
        # fi[b, gly_pos[q], x] = (mean - E_x[x]) / std
        fi_gly_x[b, gly_pos, :] = ((mean_d - E_x) / std_d).T  # [Q, 20]

    return fi_gly_x


def run_mutations_exact_gly(model, batch):
    """
    Like run_mutations_from_pdb but replaces GLY-related FI values with
    exact per-mutant batch results:
      - Case 1 (G→X): compute_exact_gly_native_fi  — per-GLY mini-batch, vCB geometry
      - Case 2 (X→G): compute_exact_gly_decoy_fi   — per-non-GLY mini-batch, CA geometry
    All non-GLY mutations use the standard gly_water_corr stats.
    """
    from pytorch_frustratometer.structure.residue_constants import ID_TO_AA

    with torch.no_grad():
        fi, nat_e, stats, _, _, sr_mutations = model.compute_tert_frust_singleresidue(batch)

    if sr_mutations is None:
        raise RuntimeError("sr_mutations is None — pass save_mutations=True to get_model()")

    mask = batch["seq_mask"][0].bool()

    residue_index = batch.get("residue_index")
    res_nums = (residue_index[0, mask].cpu().numpy() if residue_index is not None
                else np.arange(1, mask.sum().item() + 1))

    chain_index = batch.get("chain_index")
    chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chain_ids = ([chain_letters[min(int(c), 25)] for c in chain_index[0, mask].cpu().tolist()]
                 if chain_index is not None else ["A"] * mask.sum().item())

    aa        = batch["aa_type"][0, mask].cpu()
    mut_e     = sr_mutations["energies"][0, mask].cpu().float()   # [n_res, 20]
    nat_e_sr  = nat_e[0, mask].cpu().float()

    mut_stats = sr_mutations.get("stats")
    d_mean = d_std = None
    if mut_stats is not None:
        ds     = mut_stats[0, mask].cpu().float()
        d_mean = ds[:, 0]
        d_std  = ds[:, 1]

    # Case 2 (X→G): exact FI per non-GLY position
    print("  Computing exact X→G FI values (per-mutant batch) …")
    with torch.no_grad():
        exact_fi_xg = compute_exact_gly_decoy_fi(model, batch)   # [B, L]
    exact_fi_xg_masked = exact_fi_xg[0, mask].cpu().float()  # [n_res]

    # Case 1 (G→X): exact FI for all 20 mutants at each GLY position
    print("  Computing exact G→X FI values (per-GLY mini-batch) …")
    with torch.no_grad():
        exact_fi_gx = compute_exact_gly_native_fi(model, batch)  # [B, L, 20]
    exact_fi_gx_masked = exact_fi_gx[0, mask].cpu().float()  # [n_res, 20]

    rows = []
    for r in range(mut_e.shape[0]):
        aa_nat     = ID_TO_AA.get(int(aa[r]), "X")
        is_nat_gly = (int(aa[r]) == GLY_IDX)
        for d in range(20):
            if is_nat_gly:
                # Case 1 G→X: use exact per-GLY mini-batch result
                fi_val = float(exact_fi_gx_masked[r, d])
            elif d == GLY_IDX and not np.isnan(float(exact_fi_xg_masked[r])):
                # Case 2 X→G: use exact per-non-GLY mini-batch result
                fi_val = float(exact_fi_xg_masked[r])
            else:
                fi_val = (float((d_mean[r] - mut_e[r, d]) / d_std[r].clamp(min=1e-8))
                          if d_mean is not None else float("nan"))
            rows.append({
                "Res":          int(res_nums[r]),
                "ChainRes":     chain_ids[r],
                "AA_native":    aa_nat,
                "AA_mutant":    ID_TO_AA.get(d, "X"),
                "NativeEnergy": float(nat_e_sr[r]),
                "MutantEnergy": float(mut_e[r, d]),
                "FrstIndex":    fi_val,
            })

    return pd.DataFrame(rows)


def test_gly_components(model, batch, res_1indexed_list, ref_df=None):
    """
    For each GLY position in res_1indexed_list, print a per-AA breakdown
    showing burial / water / elec contributions from both _compute_gly_native_fi
    (delta-from-native formulation) and _compute_gly_mutant_fi (absolute energies
    with vCB-corrected electrostatics).

    This helps isolate WHERE the ordering of amino acids flips between our
    computation and FrustratometeR's reference.
    """
    from pytorch_frustratometer.structure.residue_constants import ID_TO_AA
    import torch

    GLY_IDX = 5
    all20 = list(range(20))
    cfg = model.coefficients

    residue_index = batch.get("residue_index")
    ri = residue_index[0].cpu().tolist() if residue_index is not None else list(range(batch["aa_type"].shape[1]))
    ri_to_pos0 = {int(r): i for i, r in enumerate(ri)}

    with torch.no_grad():
        rho_vcb, rho_ca, _ = model._compute_gly_alt_densities(batch)

        from pytorch_frustratometer.structure.geometry import compute_ideal_vcb, get_CB_coords
        vcb = compute_ideal_vcb(batch, dtype=model.dtype)
        native_centroid = get_CB_coords(
            batch["atom_positions"][:, :, :5], batch["aa_type"]
        ).to(model.dtype)

        # Burial for all 20 at all positions
        all20_t = torch.arange(20, device=batch["aa_type"].device)
        rho_vcb_exp = rho_vcb.unsqueeze(0).expand(20, *rho_vcb.shape)
        all20_exp   = all20_t.view(20, 1, 1).expand(20, *batch["aa_type"].shape)
        burial_all  = model.compute_burial_energy(batch, all20_exp, rho_vcb_exp)  # [20, B, L]
        burial_all_nat = model.compute_burial_energy(
            batch, batch["aa_type"].unsqueeze(0), batch["residue_density"].unsqueeze(0)
        )  # [1, B, L]

        # Native energy and electrostatic field (CA-based)
        native_energy = model._compute_native_energy(batch) if hasattr(model, "_compute_native_energy") else None
        field_ca = model.compute_sr_elec_field(batch)  # [B, L]

        # Sequence frequency
        valid_mask = batch["seq_mask"].bool() & ~batch["hetatm"].bool()
        seq_len = valid_mask[0].float().sum().clamp(min=1)
        freq = torch.stack(
            [((batch["aa_type"][0] == a) & valid_mask[0]).float().sum() for a in range(20)]
        ).to(model.dtype) / seq_len

    b = 0
    for res in sorted(res_1indexed_list):
        pos0 = ri_to_pos0.get(res)
        if pos0 is None:
            continue
        aa_nat = ID_TO_AA.get(int(batch["aa_type"][b, pos0]), "?")
        if int(batch["aa_type"][b, pos0]) != GLY_IDX:
            print(f"\nRes {res} ({aa_nat}) is not GLY — skipping component breakdown")
            continue

        rho_q_vcb = float(rho_vcb[b, pos0])
        rho_q_nat = float(batch["residue_density"][b, pos0])
        print(f"\n=== Component breakdown at GLY position {res} "
              f"(rho_nat={rho_q_nat:.4f}, rho_vcb={rho_q_vcb:.4f}) ===")
        ref_hdr = "    fi_ref" if ref_df is not None else ""
        print(f"  {'AA':>3}  {'burial':>8}  {'water':>8}  {'elec_CA':>8}  {'E_native':>10}  "
              f"{'elec_vCB':>9}  {'E_mutant':>10}  {'fi_native':>9}  {'fi_mutant':>9}{ref_hdr}")

        with torch.no_grad():
            d_vcb_q = torch.cdist(
                vcb[b, pos0:pos0+1].float(), native_centroid[b].float()
            ).squeeze(0).to(model.dtype)  # [L]
            d_ca_q = batch["distance"][b, pos0, :]  # [L]

            ri_j = residue_index[b]; vm_j = valid_mask[b]
            min_sep = int(cfg.min_seq_distance)
            ri_pos = int(residue_index[b, pos0])
            sep = (ri_pos - ri_j.cpu().float()).abs()
            vp = ((sep > min_sep) & vm_j.cpu()).to(model.dtype)

            rho_j = batch["residue_density"][b]
            eta_b  = float(cfg.eta)
            rmin_b = float(cfg.well_rmin[0])
            rmax_b = float(cfg.well_rmax[0])
            def vtheta(d):
                return 0.25 * (1 + torch.tanh(eta_b*(d-rmin_b))) * (1 + torch.tanh(eta_b*(rmax_b-d)))
            delta_rho_j = vtheta(d_vcb_q) - vtheta(d_ca_q)
            rho_j_corr = (rho_j + delta_rho_j).clamp(min=0)

            water_gamma = model.get_water_gamma(batch)
            kw = float(cfg.water_kappa)
            rmin0 = cfg.well_rmin[0].to(model.dtype); rmax0 = cfg.well_rmax[0].to(model.dtype)
            rmin1 = cfg.well_rmin[1].to(model.dtype); rmax1 = cfg.well_rmax[1].to(model.dtype)

            def water_e_single(aa_x, dist, rho_src, rho_dst):
                rho_src_t = torch.as_tensor(rho_src, dtype=model.dtype)
                sw = 0.25*(1-torch.tanh(cfg.water_kappa_sigma*(rho_src_t-cfg.threshold))) \
                        * (1-torch.tanh(cfg.water_kappa_sigma*(rho_dst-cfg.threshold)))
                sp = 1.0 - sw
                gd  = water_gamma[0, aa_x, batch["aa_type"][b], :].mean(-1)
                gpm = water_gamma[1, aa_x, batch["aa_type"][b], 0]
                gwm = water_gamma[1, aa_x, batch["aa_type"][b], 1]
                td  = 0.25*(1+torch.tanh(kw*(dist-rmin0)))*(1+torch.tanh(kw*(rmax0-dist)))
                tm  = 0.25*(1+torch.tanh(kw*(dist-rmin1)))*(1+torch.tanh(kw*(rmax1-dist)))
                return (-(gd*td + (sp*gpm+sw*gwm)*tm) * vp).sum()

            # Electrostatic field at vCB
            field_vcb_q = torch.tensor(0.0, dtype=model.dtype)
            if cfg.use_electrostatics:
                q_j = model.charge_lookup[batch["aa_type"][b]].to(model.dtype)
                safe_d = d_vcb_q.clamp(min=1e-6)
                self_mask = torch.zeros(len(ri_j), dtype=torch.bool)
                self_mask[pos0] = True
                ev = (~self_mask) & vm_j.cpu() & (q_j.cpu() != 0)
                kernel = (q_j.cpu() / safe_d.cpu()) * torch.exp(-safe_d.cpu() / float(cfg.screening_length))
                field_vcb_q = float(cfg.k_PlusMinus) * (kernel * ev.float()).sum()

            # Compute per-AA components
            E_native_list = []; E_mutant_list = []
            for d in range(20):
                bur = float(burial_all[d, b, pos0])
                w   = float(water_e_single(d, d_vcb_q, rho_q_vcb, rho_j_corr))
                q_d = float(model.charge_lookup[d])
                e_ca  = q_d * float(field_ca[b, pos0])
                e_vcb = q_d * float(field_vcb_q)

                E_native_list.append(bur + w + e_ca)
                E_mutant_list.append(bur + w + e_vcb)

            E_nat_t = torch.tensor(E_native_list, dtype=model.dtype)
            E_mut_t = torch.tensor(E_mutant_list, dtype=model.dtype)
            mean_nat = (freq * E_nat_t).sum(); std_nat = (freq * (E_nat_t - mean_nat)**2).sum().clamp(min=0).sqrt()
            mean_mut = (freq * E_mut_t).sum(); std_mut = (freq * (E_mut_t - mean_mut)**2).sum().clamp(min=0).sqrt()

            for d in range(20):
                aa_x = ID_TO_AA.get(d, "X")
                bur  = float(burial_all[d, b, pos0])
                w    = float(water_e_single(d, d_vcb_q, rho_q_vcb, rho_j_corr))
                q_d  = float(model.charge_lookup[d])
                e_ca  = q_d * float(field_ca[b, pos0])
                e_vcb = q_d * float(field_vcb_q)
                fi_nat = float((mean_nat - E_nat_t[d]) / std_nat.clamp(min=1e-8))
                fi_mut = float((mean_mut - E_mut_t[d]) / std_mut.clamp(min=1e-8))
                ref_str = ""
                if ref_df is not None:
                    aa_col = "AA_mutant" if "AA_mutant" in ref_df.columns else "AA"
                    row = ref_df[(ref_df["Res"] == res) & (ref_df[aa_col] == aa_x)]
                    if len(row):
                        ref_str = f"  {float(row['FrstIndex'].iloc[0]):>8.3f}"
                print(f"  {aa_x:>3}  {bur:>8.3f}  {w:>8.3f}  {e_ca:>8.3f}  {E_native_list[d]:>10.4f}  "
                      f"{e_vcb:>9.3f}  {E_mutant_list[d]:>10.4f}  {fi_nat:>9.3f}  {fi_mut:>9.3f}{ref_str}")


def test_mutate_res(our_df_or_path, mut_ref_dir):
    """
    Compare our singleresidue_mutations output against FrustratometeR's
    MutationsData reference.

    Accepts either:
      - a file path (str/Path) to a pre-computed mutations CSV
      - a pandas DataFrame produced by run_mutations_from_pdb()

    A mutation row is "GLY-related" when:
      - AA_native == 'G'  (native is GLY; we replaced it with every other AA)
      - AA_mutant == 'G'  (native is not GLY; we're mutating TO GLY)

    Reports mean |FrstIndex error| for GLY-related vs non-GLY rows.
    If GLY errors are significantly larger, the GLY correction is likely not
    being applied during mutation scanning.
    """
    print(f"\n=== Test 4: mutation scanning GLY correction ===")
    print(f"  Reference dir  : {mut_ref_dir}")

    from pytorch_frustratometer.data.io import read_mutations_data

    # --- load our output ---
    if isinstance(our_df_or_path, pd.DataFrame):
        our_df = our_df_or_path
        print(f"  Mutations source: in-memory DataFrame")
    else:
        mutations_path = str(our_df_or_path)
        print(f"  Mutations file : {mutations_path}")
        if not os.path.isfile(mutations_path):
            print(f"  FAIL: file not found: {mutations_path}")
            return
        our_df = pd.read_csv(mutations_path, sep=r"\s+", comment="#")

    required_cols = {"Res", "ChainRes", "AA_native", "AA_mutant", "FrstIndex"}
    missing = required_cols - set(our_df.columns)
    if missing:
        print(f"  FAIL: mutations data missing columns: {missing}")
        print(f"  Found: {list(our_df.columns)}")
        return

    our_df["FrstIndex"] = pd.to_numeric(our_df["FrstIndex"], errors="coerce")
    print(f"  Our rows: {len(our_df):,}  "
          f"(non-NaN FrstIndex: {our_df['FrstIndex'].notna().sum():,})")

    # --- load FrustratometeR reference ---
    try:
        ref_df = read_mutations_data(mut_ref_dir)
    except FileNotFoundError as exc:
        print(f"  FAIL: could not read reference — {exc}")
        return

    ref_df["FrstIndex"] = pd.to_numeric(ref_df["FrstIndex"], errors="coerce")
    print(f"  Ref rows : {len(ref_df):,}  "
          f"(non-NaN FrstIndex: {ref_df['FrstIndex'].notna().sum():,})")

    # --- merge on (Res, ChainRes, AA_mutant) ---
    merged = our_df.merge(
        ref_df,
        left_on=["Res", "ChainRes", "AA_mutant"],
        right_on=["Res", "ChainRes", "AA"],
        suffixes=("_ours", "_ref"),
    )
    if len(merged) == 0:
        print("  FAIL: merge produced 0 rows — check column values "
              "(Res types, ChainRes format, AA encoding).")
        print(f"  Our  (Res sample): {our_df['Res'].head(3).tolist()}")
        print(f"  Ref  (Res sample): {ref_df['Res'].head(3).tolist()}")
        return

    print(f"  Merged rows: {len(merged):,}")

    fi_ours = merged["FrstIndex_ours"]
    fi_ref  = merged["FrstIndex_ref"]
    valid   = fi_ours.notna() & fi_ref.notna()
    err     = (fi_ours - fi_ref).abs()

    # GLY-related: native is GLY OR mutant is GLY
    is_gly = (merged["AA_native"] == "G") | (merged["AA_mutant"] == "G")

    gly_err     = err[valid &  is_gly]
    non_gly_err = err[valid & ~is_gly]

    def _stats(s):
        if len(s) == 0:
            return "n=0"
        return f"mean={s.mean():.4f}  max={s.max():.4f}  n={len(s):,}"

    print(f"\n  GLY-related  (native=G or mutant=G): {_stats(gly_err)}")
    print(f"  non-GLY                             : {_stats(non_gly_err)}")

    # Break GLY-related into sub-categories
    gly_native_err = err[valid & (merged["AA_native"] == "G")]
    gly_mutant_err = err[valid & (merged["AA_native"] != "G") & (merged["AA_mutant"] == "G")]
    print(f"\n  Breakdown:")
    print(f"    AA_native==G (GLY→X mutations) : {_stats(gly_native_err)}")
    print(f"    AA_mutant==G (X→GLY mutations) : {_stats(gly_mutant_err)}")

    # Overall R²
    if valid.sum() >= 2:
        from scipy.stats import pearsonr
        r, _ = pearsonr(fi_ours[valid], fi_ref[valid])
        print(f"\n  Overall R² = {r**2:.4f}  (n={valid.sum():,})")

    # Diagnosis
    if len(gly_err) > 0 and len(non_gly_err) > 0:
        ratio = gly_err.mean() / (non_gly_err.mean() + 1e-12)
        if ratio > 3:
            print(f"\n  DIAGNOSIS: GLY errors are {ratio:.1f}x larger than non-GLY.")
            print("             The GLY correction is likely NOT being applied in mutation scanning.")
        else:
            print(f"\n  GLY/non-GLY error ratio = {ratio:.2f} — errors are similar.")
            print("  GLY correction appears to be working in mutation scanning.")

    # Show worst GLY-related rows
    gly_rows = merged[valid & is_gly].copy()
    gly_rows["error"] = err[valid & is_gly]
    worst = gly_rows.nlargest(10, "error")[
        ["Res", "ChainRes", "AA_native", "AA_mutant",
         "FrstIndex_ours", "FrstIndex_ref", "error"]
    ]
    if len(worst):
        print(f"\n  Worst 10 GLY-related rows by |error|:")
        print(worst.to_string(index=False))

    # Return unique residue numbers of worst GLY positions for density diagnostic
    return sorted(worst["Res"].unique().tolist()) if len(worst) else []


# ---------------------------------------------------------------------------
# Test 6: X→G (non-GLY → GLY decoy) errors by position
# ---------------------------------------------------------------------------

def test_gly_decoy_by_position(our_df, ref_df, batch, model=None, top_n=20):
    """
    For X→G mutations (non-GLY native → GLY decoy), print per-position errors
    sorted by magnitude.

    Shows rho_native at each position (how buried the native CB is — removing
    it to make a GLY decoy has a larger ripple effect when rho is high).

    If model is given, also shows rho_ca (the density our code uses for the
    GLY decoy burial correction) and |Δrho| = |rho_ca − rho_native|.

    Summary at the end bins positions by rho_native to show whether high-rho
    positions have systematically larger X→G errors.
    """
    ref_renamed = ref_df.rename(columns={"AA": "AA_mutant", "FrstIndex": "FrstIndex_ref"})
    ref_renamed["FrstIndex_ref"] = pd.to_numeric(ref_renamed["FrstIndex_ref"], errors="coerce")

    merged = our_df.merge(
        ref_renamed[["Res", "ChainRes", "AA_mutant", "FrstIndex_ref"]],
        on=["Res", "ChainRes", "AA_mutant"],
        how="inner",
    )

    gly_decoy = merged[(merged["AA_native"] != "G") & (merged["AA_mutant"] == "G")].copy()
    valid = gly_decoy["FrstIndex"].notna() & gly_decoy["FrstIndex_ref"].notna()
    gly_decoy = gly_decoy[valid].copy()
    gly_decoy["error"]      = (gly_decoy["FrstIndex"] - gly_decoy["FrstIndex_ref"]).abs()
    gly_decoy["signed_err"] =  gly_decoy["FrstIndex"] - gly_decoy["FrstIndex_ref"]

    if gly_decoy.empty:
        print("\n=== X→G decoy errors: no rows found ===")
        return

    top_rows = gly_decoy.nlargest(top_n, "error")

    # Build rho lookups from batch
    residue_index = batch.get("residue_index")
    ri = (residue_index[0].cpu().tolist() if residue_index is not None
          else list(range(batch["aa_type"].shape[1])))
    ri_to_pos0 = {int(r): i for i, r in enumerate(ri)}
    rho_native = batch["residue_density"][0].cpu()

    # rho_ca: density at each position when using CA instead of CB
    rho_ca_tensor = None
    if model is not None:
        with torch.no_grad():
            _, rho_ca_all, _ = model._compute_gly_alt_densities(batch)
        rho_ca_tensor = rho_ca_all[0].cpu()

    has_rho_ca = rho_ca_tensor is not None

    print(f"\n=== Test 6: X→G decoy errors by position "
          f"(top {min(top_n, len(gly_decoy))} of {len(gly_decoy)} rows) ===")

    hdr = (f"  {'Res':>5}  {'AA_nat':>6}  {'rho_nat':>9}"
           + (f"  {'rho_ca':>9}  {'Δrho':>8}" if has_rho_ca else "")
           + f"  {'FI_ours':>9}  {'FI_ref':>9}  {'|err|':>7}  {'signed':>8}")
    print(hdr)

    for _, row in top_rows.iterrows():
        res  = int(row["Res"])
        pos0 = ri_to_pos0.get(res)
        rn   = float(rho_native[pos0]) if pos0 is not None else float("nan")
        rho_ca_str = ""
        if has_rho_ca and pos0 is not None:
            rc = float(rho_ca_tensor[pos0])
            rho_ca_str = f"  {rc:>9.4f}  {rc - rn:>+8.4f}"
        print(f"  {res:>5}  {row['AA_native']:>6}  {rn:>9.4f}"
              + rho_ca_str
              + f"  {row['FrstIndex']:>9.3f}  {row['FrstIndex_ref']:>9.3f}"
              + f"  {row['error']:>7.3f}  {row['signed_err']:>+8.3f}")

    # Summary stats by rho_native bin — shows whether buried positions are worse
    print(f"\n  Mean |error| by rho_native range (all {len(gly_decoy)} X→G pairs):")
    print(f"  {'rho range':>14}  {'n':>5}  {'mean_err':>10}  {'max_err':>10}")
    bins = [(0, 1), (1, 3), (3, 6), (6, 9), (9, 99)]
    for lo, hi in bins:
        bin_errs = []
        for _, row in gly_decoy.iterrows():
            pos0 = ri_to_pos0.get(int(row["Res"]))
            if pos0 is None:
                continue
            rn = float(rho_native[pos0])
            if lo <= rn < hi:
                bin_errs.append(float(row["error"]))
        if bin_errs:
            hi_str = f"{hi:.0f}" if hi < 99 else "∞"
            print(f"  [{lo:.0f}, {hi_str})          "
                  f"  {len(bin_errs):>5}  {np.mean(bin_errs):>10.4f}  {np.max(bin_errs):>10.4f}")

    # Signed error summary: are we systematically over- or under-estimating?
    pos_err = (gly_decoy["signed_err"] > 0).sum()
    neg_err = (gly_decoy["signed_err"] < 0).sum()
    mean_signed = gly_decoy["signed_err"].mean()
    print(f"\n  Signed error: mean={mean_signed:+.4f}  "
          f"over-estimate={pos_err} ({100*pos_err/len(gly_decoy):.0f}%)  "
          f"under-estimate={neg_err} ({100*neg_err/len(gly_decoy):.0f}%)")


# ---------------------------------------------------------------------------
# Per-position bias diagnostic for G→X mutations
# ---------------------------------------------------------------------------

def test_gly_position_bias(our_df, ref_df, batch, model=None):
    """
    Per-GLY-position signed-error diagnostic for G→X mutations.

    Shows mean(our_FI − ref_FI) across all 19 non-GLY substitutions at each
    native-GLY position.  A large, consistent (one-sign) mean indicates the
    decoy distribution statistics at that position are biased — a correctable
    column-level shift — rather than random per-mutation scatter.

    Also runs per-position linear regression (our_fi ~ ref_fi) to separate
    scale (std ratio ≠ 1) from offset (mean shift ≠ 0) effects.
    """
    from scipy.stats import pearsonr, linregress

    ref_renamed = ref_df.rename(columns={"AA": "AA_mutant", "FrstIndex": "FrstIndex_ref"})
    ref_renamed["FrstIndex_ref"] = pd.to_numeric(ref_renamed["FrstIndex_ref"], errors="coerce")

    merged = our_df.merge(
        ref_renamed[["Res", "ChainRes", "AA_mutant", "FrstIndex_ref"]],
        on=["Res", "ChainRes", "AA_mutant"],
        how="inner",
    )

    gly_native = merged[
        (merged["AA_native"] == "G") & (merged["AA_mutant"] != "G")
    ].copy()
    valid = gly_native["FrstIndex"].notna() & gly_native["FrstIndex_ref"].notna()
    gly_native = gly_native[valid].copy()
    gly_native["signed_err"] = gly_native["FrstIndex"] - gly_native["FrstIndex_ref"]
    gly_native["abs_err"]    = gly_native["signed_err"].abs()

    if gly_native.empty:
        print("\n=== GLY position bias: no G→X rows found ===")
        return

    # Density info from batch
    residue_index = batch.get("residue_index")
    ri = (residue_index[0].cpu().tolist() if residue_index is not None
          else list(range(batch["aa_type"].shape[1])))
    ri_to_pos0 = {int(r): i for i, r in enumerate(ri)}
    rho_native = batch["residue_density"][0].cpu()

    rho_vcb_tensor = None
    if model is not None:
        with torch.no_grad():
            rho_vcb_all, _, _ = model._compute_gly_alt_densities(batch)
        rho_vcb_tensor = rho_vcb_all[0].cpu()

    print(f"\n=== GLY position bias (G→X): per-position signed error ===")
    print(f"  mean_SE = mean(our_FI − ref_FI) over 19 non-GLY substitutions")
    print(f"  slope/intercept from linear regression our_FI = slope*ref_FI + intercept")
    print(f"  slope≈1, intercept≈0 → no bias; slope≠1 → std wrong; intercept≠0 → mean shift")
    print()
    print(f"  {'Res':>5}  {'rho_nat':>8}  {'rho_vcb':>8}  "
          f"{'mean_SE':>9}  {'std_SE':>8}  {'max|E|':>8}  "
          f"{'slope':>7}  {'intcpt':>7}  {'r':>6}  {'n':>3}  note")

    by_pos = gly_native.groupby("Res")
    rows_out = []
    for res, grp in by_pos:
        pos0 = ri_to_pos0.get(int(res))
        rn = float(rho_native[pos0]) if pos0 is not None else float("nan")
        rv = (float(rho_vcb_tensor[pos0])
              if (rho_vcb_tensor is not None and pos0 is not None) else float("nan"))

        se      = grp["signed_err"]
        mean_se = se.mean()
        std_se  = se.std() if len(se) > 1 else float("nan")
        max_ae  = grp["abs_err"].max()
        n = len(grp)

        # Per-position linear regression: our_fi = slope * ref_fi + intercept
        slope = intcpt = r_val = float("nan")
        if n >= 3:
            try:
                lr = linregress(grp["FrstIndex_ref"].values, grp["FrstIndex"].values)
                slope, intcpt, r_val = lr.slope, lr.intercept, lr.rvalue
            except Exception:
                pass

        # Systematic if |mean| > half the within-position std
        note = ""
        if n >= 5 and not np.isnan(std_se) and std_se > 1e-6:
            if abs(mean_se) > 0.5 * std_se:
                note = "over" if mean_se > 0 else "under"

        rows_out.append((res, rn, rv, mean_se, std_se, max_ae, slope, intcpt, r_val, n, note))

    rows_out.sort(key=lambda x: abs(x[3]), reverse=True)
    for res, rn, rv, me, se, ma, sl, ic, rv2, n, note in rows_out:
        print(f"  {res:>5}  {rn:>8.4f}  {rv:>8.4f}  "
              f"  {me:>+8.4f}  {se:>8.4f}  {ma:>8.4f}  "
              f"  {sl:>6.3f}  {ic:>+6.3f}  {rv2:>+6.3f}  {n:>3}  {note}")

    # Summary
    mean_ses = [r[3] for r in rows_out]
    slopes   = [r[6] for r in rows_out if not np.isnan(r[6])]
    n_sys    = sum(1 for r in rows_out if r[10])
    print(f"\n  {len(rows_out)} GLY positions total.  {n_sys} show systematic (consistent-sign) bias.")
    print(f"  mean(|mean_SE|) = {np.mean(np.abs(mean_ses)):.4f}")
    if slopes:
        ms = np.mean(slopes)
        print(f"  mean(slope)     = {ms:.4f}  "
              f"({'our std too small → FI range compressed' if ms > 1 else 'our std too large → FI range expanded' if ms < 1 else 'OK'})")

    # Correlation of position-level bias with density
    valid_rows = [(r[1], r[2], r[3]) for r in rows_out
                  if not np.isnan(r[1]) and not np.isnan(r[3])]
    if len(valid_rows) >= 3:
        rhos_n = [r[0] for r in valid_rows]
        rhos_v = [r[1] for r in valid_rows]
        mses   = [r[2] for r in valid_rows]
        r_nat, p_nat = pearsonr(rhos_n, mses)
        r_vcb, p_vcb = pearsonr(rhos_v, mses)
        print(f"\n  Correlation of mean_SE with rho_nat: r={r_nat:+.3f}  p={p_nat:.3f}")
        print(f"  Correlation of mean_SE with rho_vcb: r={r_vcb:+.3f}  p={p_vcb:.3f}")
        if abs(r_nat) > 0.5 and p_nat < 0.05:
            print(f"  → Significant rho_nat correlation: bias may be correctable "
                  f"by a density-dependent adjustment to the decoy mean.")


# ---------------------------------------------------------------------------
# Density diagnostic at specific residue positions
# ---------------------------------------------------------------------------

def test_density_at_positions(model, batch, res_1indexed_list):
    """
    Print rho_nat and rho_vcb at a list of 1-indexed residue positions.

    Shows whether each position is threshold-crossing (rho_nat and rho_vcb on
    opposite sides of cfg.threshold), which determines whether the burial
    correction is suppressed by the threshold-crossing protection in
    _apply_gly_correction.
    """
    if not res_1indexed_list:
        return

    print(f"\n=== Density diagnostic at worst GLY positions ===")

    threshold = float(model.coefficients.threshold)
    rho_nat   = batch["residue_density"][0].cpu()   # [L]
    aa_type   = batch["aa_type"][0].cpu()            # [L]

    residue_index = batch.get("residue_index")
    if residue_index is not None:
        ri = residue_index[0].cpu().tolist()
        ri_to_pos0 = {int(r): i for i, r in enumerate(ri)}
    else:
        ri_to_pos0 = {i + 1: i for i in range(len(aa_type))}

    with torch.no_grad():
        rho_vcb, _, _vcb_hybrid = model._compute_gly_alt_densities(batch)
    rho_vcb_0 = rho_vcb[0].cpu()

    from pytorch_frustratometer.structure.residue_constants import ID_TO_AA

    print(f"  threshold = {threshold:.2f}")
    print(f"  {'Res':>5}  {'AA':>3}  {'rho_nat':>9}  {'rho_vcb':>9}  "
          f"{'delta':>8}  {'nat<thr':>8}  {'vcb<thr':>8}  {'status':>12}")

    for res in sorted(res_1indexed_list):
        pos0 = ri_to_pos0.get(res)
        if pos0 is None:
            print(f"  {res:>5}  (residue not found)")
            continue

        aa_str   = ID_TO_AA.get(int(aa_type[pos0]), "?")
        rn       = float(rho_nat[pos0])
        rv       = float(rho_vcb_0[pos0])
        nat_low  = rn < threshold
        vcb_low  = rv < threshold
        crossing = nat_low != vcb_low

        status = "CROSSING" if crossing else ("both-low" if nat_low else "both-high")
        print(f"  {res:>5}  {aa_str:>3}  {rn:>9.4f}  {rv:>9.4f}  "
              f"  {rv - rn:>+7.4f}  {'<' if nat_low else '>=':>8}  "
              f"{'<' if vcb_low else '>=':>8}  {status:>12}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdb",  default=None, help="Path to PDB file")
    parser.add_argument("--ref",  default=None,
                        help="Path to FrustratometeR reference output file "
                             "(e.g. protein.pdb_singleresidue)")
    parser.add_argument("--mode", default="singleresidue",
                        choices=["singleresidue", "mutational", "configurational"],
                        help="Frustration mode (default: singleresidue)")
    parser.add_argument("--device", default="cpu",
                        help="torch device (default: cpu)")
    parser.add_argument("--mutations", default=None,
                        help="Path to a pre-computed *_singleresidue_mutations file "
                             "(skip running the model)")
    parser.add_argument("--mut-ref", default=None,
                        help="Path to FrustratometeR MutationsData/ directory "
                             "(or a *.done directory containing it)")
    parser.add_argument("--mut-ref2", default=None,
                        help="Optional second MutationsData/ directory for a side-by-side "
                             "comparison (e.g. original dompdb_v4.4 reference)")
    parser.add_argument("--diagnose-pos", type=int, nargs="+", default=None,
                        help="Run energy breakdown diagnostic at these 1-indexed positions "
                             "(space-separated, e.g. --diagnose-pos 39 22 175)")
    parser.add_argument("--mutant-fi", action="store_true", default=False,
                        help="Use _compute_gly_mutant_fi (absolute AWSEM energies at vCB) "
                             "for Case 1 GLY mutations instead of gly_water_corr. "
                             "Requires --mut-ref to compare against reference.")
    parser.add_argument("--components-pos", type=int, nargs="+", default=None,
                        help="Print burial/water/elec component breakdown at these GLY "
                             "positions (space-separated, e.g. --components-pos 175 207). "
                             "Requires --pdb and --mut-ref.")
    # Burial density hypothesis flags
    parser.add_argument("--burial-rmin", type=float, default=None,
                        help="Lower distance cutoff for burial density (default: 4.5 = same as water). "
                             "Set to 2.0 to test hypothesis 1 (AWSEM uses tighter contacts).")
    parser.add_argument("--burial-rmax", type=float, default=None,
                        help="Upper distance cutoff for burial density (default: 6.5 = same as water). "
                             "Set to 9.0 to test hypothesis 2 (AWSEM uses wider cutoff).")
    parser.add_argument("--burial-min-sep", type=int, default=None,
                        help="Minimum sequence separation for burial density (default: 3 = same as water). "
                             "Set to 1 to include nearest-neighbor contacts in burial density.")
    parser.add_argument("--seq-dist", type=int, default=None,
                        help="Minimum sequence separation for water/contact pairs (default: 3). "
                             "Set to 12 to test long-range-only contacts.")
    parser.add_argument("--no-electrostatics", action="store_true", default=False,
                        help="Disable electrostatic energy term.")
    parser.add_argument("--diagnose-gly-decoy", type=int, nargs="?", const=20, default=None,
                        metavar="N",
                        help="Show per-position X→G (non-GLY→GLY decoy) error table, "
                             "sorted by |error|. Optional N = rows to show (default 20). "
                             "Requires --mut-ref and --pdb.")
    parser.add_argument("--diagnose-gly-bias", action="store_true", default=False,
                        help="Per-position bias diagnostic for G→X mutations: shows mean "
                             "signed error and linear regression (our_FI ~ ref_FI) at each "
                             "native-GLY position to identify column-level mean/std offsets. "
                             "Requires --mut-ref and --pdb.")
    parser.add_argument("--exact-gly", action="store_true", default=False,
                        help="Run each X→G (non-GLY→GLY decoy) mutation as its own "
                             "mini-batch for exact FI computation. Slow but matches "
                             "FrustratometeR's per-mutant-structure approach. "
                             "Requires --mut-ref and --pdb.")
    args = parser.parse_args()

    # --mutations: load a pre-computed file and compare (no PDB needed)
    if args.mutations and not args.pdb:
        if not os.path.isfile(args.mutations):
            sys.exit(f"Mutations file not found: {args.mutations}")
        if not args.mut_ref:
            sys.exit("--mut-ref is required when --mutations is given")
        if not os.path.exists(args.mut_ref):
            sys.exit(f"MutationsData reference not found: {args.mut_ref}")
        test_mutate_res(args.mutations, args.mut_ref)
        print("\nDone.")
        return

    if not args.pdb:
        sys.exit("--pdb is required (or use --mutations without --pdb for file-only test 4)")

    if not os.path.isfile(args.pdb):
        sys.exit(f"PDB not found: {args.pdb}")

    print(f"PDB:  {args.pdb}")
    print(f"Mode: {args.mode}")

    # Decide whether we need save_mutations
    need_mutations = bool(args.mut_ref) or bool(args.mutations) or args.exact_gly

    model = get_model(args.mode, save_mutations=need_mutations,
                      burial_rmin=args.burial_rmin,
                      burial_rmax=args.burial_rmax,
                      burial_min_sep=args.burial_min_sep,
                      seq_dist=args.seq_dist,
                      no_electrostatics=args.no_electrostatics)
    model = model.to(args.device)
    model.eval()

    print("\nLoading batch …")
    batch = load_batch(args.pdb, model=model)
    batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    rho_vcb, rho_ca = test_densities(model, batch)
    test_energy_change(model, batch, rho_vcb, rho_ca)

    if args.ref:
        if not os.path.isfile(args.ref):
            print(f"\nReference file not found: {args.ref}")
        else:
            test_vs_reference(model, batch, args.ref, args.mode)

    # Pre-load reference DataFrames for any diagnostic that needs them
    from pytorch_frustratometer.data.io import read_mutations_data
    ref_df_cached = None
    ref_df2_cached = None
    if args.mut_ref and os.path.exists(args.mut_ref):
        try:
            ref_df_cached = read_mutations_data(args.mut_ref)
        except Exception as e:
            print(f"  (could not load mut-ref: {e})")
    if args.mut_ref2 and os.path.exists(args.mut_ref2):
        try:
            ref_df2_cached = read_mutations_data(args.mut_ref2)
        except Exception as e:
            print(f"  (could not load mut-ref2: {e})")

    # Test 5: energy breakdown diagnostic (one section per position)
    if args.diagnose_pos is not None:
        for pos in args.diagnose_pos:
            ref_fi_by_mut = None
            if ref_df_cached is not None:
                pos_rows = ref_df_cached[ref_df_cached["Res"] == pos]
                if len(pos_rows) > 0:
                    ref_fi_by_mut = dict(zip(pos_rows["AA"], pos_rows["FrstIndex"].astype(float)))
            ref_fi_by_mut2 = None
            if ref_df2_cached is not None:
                pos_rows2 = ref_df2_cached[ref_df2_cached["Res"] == pos]
                if len(pos_rows2) > 0:
                    ref_fi_by_mut2 = dict(zip(pos_rows2["AA"], pos_rows2["FrstIndex"].astype(float)))
            test_energy_breakdown(model, batch, pos,
                                  ref_fi_by_mut=ref_fi_by_mut,
                                  ref_fi_by_mut2=ref_fi_by_mut2)

    # Test 4: mutation scanning comparison
    our_df_cached = None
    if args.mut_ref:
        if not os.path.exists(args.mut_ref):
            print(f"\nMutationsData reference not found: {args.mut_ref}")
        else:
            if args.mutations:
                # User provided a pre-computed file — use that instead of re-running
                print("\nRunning Test 4 from pre-computed mutations file …")
                worst_positions = test_mutate_res(args.mutations, args.mut_ref)
            else:
                print("\nRunning mutation scanning from PDB …")
                our_df_cached = run_mutations_from_pdb(model, batch)
                print(f"  Generated {len(our_df_cached):,} mutation rows in memory.")
                worst_positions = test_mutate_res(our_df_cached, args.mut_ref)

            # Density diagnostic: print rho_nat / rho_vcb at worst GLY positions
            if worst_positions and args.pdb:
                test_density_at_positions(model, batch, worst_positions)

    # Test 6: X→G decoy errors by position
    if args.diagnose_gly_decoy is not None:
        top_n = args.diagnose_gly_decoy if args.diagnose_gly_decoy else 20
        if our_df_cached is None:
            print("\n--diagnose-gly-decoy: no mutations DataFrame available; "
                  "run with --mut-ref and --pdb to generate one.")
        elif not args.mut_ref or not os.path.exists(args.mut_ref):
            print("\n--diagnose-gly-decoy: --mut-ref is required.")
        elif ref_df_cached is None:
            print("\n--diagnose-gly-decoy: could not load reference; check --mut-ref path.")
        else:
            df_for_decoy = (our_df_cached if isinstance(our_df_cached, pd.DataFrame)
                            else pd.read_csv(our_df_cached, sep=r"\s+", comment="#"))
            test_gly_decoy_by_position(df_for_decoy, ref_df_cached, batch,
                                       model=model, top_n=top_n)

    # Test: per-position bias for G→X (column-level error investigation)
    if args.diagnose_gly_bias:
        if our_df_cached is None:
            print("\n--diagnose-gly-bias: no mutations DataFrame; run with --mut-ref and --pdb.")
        elif ref_df_cached is None:
            print("\n--diagnose-gly-bias: could not load reference; check --mut-ref path.")
        else:
            df_for_bias = (our_df_cached if isinstance(our_df_cached, pd.DataFrame)
                           else pd.read_csv(our_df_cached, sep=r"\s+", comment="#"))
            test_gly_position_bias(df_for_bias, ref_df_cached, batch, model=model)

    # Test 4 (second reference): compare same mutations against --mut-ref2
    if args.mut_ref2:
        if not os.path.exists(args.mut_ref2):
            print(f"\nSecond MutationsData reference not found: {args.mut_ref2}")
        else:
            if our_df_cached is None:
                if args.mutations:
                    # reuse the pre-computed file
                    our_df_cached = args.mutations
                else:
                    print("\nRunning mutation scanning from PDB (for --mut-ref2) …")
                    our_df_cached = run_mutations_from_pdb(model, batch)
                    print(f"  Generated {len(our_df_cached):,} mutation rows in memory.")
            print(f"\n--- Comparison against second reference: {args.mut_ref2} ---")
            test_mutate_res(our_df_cached, args.mut_ref2)

    # Test 4c: exact X→G FI via per-mutant mini-batch
    if args.exact_gly:
        if not args.mut_ref or not os.path.exists(args.mut_ref):
            print("\n--exact-gly: --mut-ref is required for comparison; skipping.")
        else:
            print("\n=== Test 4c: exact X→G FI (per-mutant mini-batch) ===")
            exact_df = run_mutations_exact_gly(model, batch)
            print(f"  Generated {len(exact_df):,} mutation rows.")
            test_mutate_res(exact_df, args.mut_ref)

    # Test 4b: per-mutant AWSEM energies via _compute_gly_mutant_fi
    if args.mutant_fi:
        if not args.mut_ref or not os.path.exists(args.mut_ref):
            print("\n--mutant-fi: --mut-ref is required for comparison; skipping.")
        else:
            print("\n=== Test 4b: _compute_gly_mutant_fi (absolute AWSEM at vCB) ===")
            mutfi_df = run_mutations_mutant_fi(model, batch)
            print(f"  Generated {len(mutfi_df):,} mutation rows.")
            test_mutate_res(mutfi_df, args.mut_ref)

    # Component breakdown diagnostic at specified GLY positions
    if args.components_pos is not None:
        print(f"\n=== Component breakdown at positions {args.components_pos} ===")
        ref_df_comp = None
        if args.mut_ref and os.path.exists(args.mut_ref):
            from pytorch_frustratometer.data.io import read_mutations_data
            try:
                ref_df_comp = read_mutations_data(args.mut_ref)
            except Exception as e:
                print(f"  (could not load mut-ref for component breakdown: {e})")
        test_gly_components(model, batch, args.components_pos, ref_df=ref_df_comp)

    print("\nDone.")


if __name__ == "__main__":
    main()

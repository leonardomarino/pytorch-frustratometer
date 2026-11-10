"""Mutational frustration — all mutational logic in one class."""
from __future__ import annotations
import dataclasses
from pathlib import Path

import click
import numpy as np
import torch

from ..utils.utils import frustration_loss as _frustration_loss

from ..utils.utils import index_to_pseudo_logits, compute_frustration_metrics
from ..coefficients import Coefficients
from .decoy_statistics import decoy_aa_frequencies, decoy_aa_weights, decoy_statistics

DECOY_CHUNK_ELEMENTS = 1 << 22  # edges x decoys evaluated per pass in compute_mutational_decoy_energies


# ---------------------------------------------------------------------------
# Free function (was in awsem.py)
# ---------------------------------------------------------------------------

def sample_mutational_decoys(
    aa_type: torch.Tensor,   # [B, L]
    seq_mask: torch.Tensor,  # [B, L]
    n_edges: int,
    n_decoys: int,
    use_c_rand: bool = False,
    native_edge_aa: tuple | None = None,  # deprecated: use decoy_distribution
    decoy_distribution: str = "sequence",
) -> torch.Tensor:           # [2, B, n_edges, n_decoys]
    """For each edge, sample n_decoys AA type pairs from the protein sequence.
    decoy_distribution controls the sampling strategy (see sample_decoy_aa_from_sequence).
    The legacy native_edge_aa tuple is equivalent to "sequence_exclude_native"."""
    # Back-compat
    if native_edge_aa is not None and decoy_distribution == "sequence":
        decoy_distribution = "sequence_exclude_native"

    exclude_native = decoy_distribution.endswith("_exclude_native")
    use_pseudocount = decoy_distribution.startswith("pseudocount")

    B, L = aa_type.shape
    device = aa_type.device

    valid_mask = seq_mask & (aa_type < 20)
    seq_lens = valid_mask.sum(dim=1)
    valid_indices = [torch.where(valid_mask[b])[0] for b in range(B)]

    if use_pseudocount:
        aa_probs = decoy_aa_frequencies(aa_type, seq_mask, "pseudocount", torch.float32)  # [B, 20]
        flat1 = torch.stack([
            torch.multinomial(aa_probs[b], n_edges * n_decoys, replacement=True)
            for b in range(B)
        ], dim=0).view(B, n_edges, n_decoys)  # [B, E, D]
        flat2 = torch.stack([
            torch.multinomial(aa_probs[b], n_edges * n_decoys, replacement=True)
            for b in range(B)
        ], dim=0).view(B, n_edges, n_decoys)
        dec_aa1, dec_aa2 = flat1, flat2

        if exclude_native:
            nat_src, nat_dst = native_edge_aa if native_edge_aa is not None else (
                aa_type[torch.arange(B, device=device).view(B,1), torch.zeros(B, n_edges, dtype=torch.long, device=device)],
                aa_type[torch.arange(B, device=device).view(B,1), torch.zeros(B, n_edges, dtype=torch.long, device=device)],
            )
            # nat_src/nat_dst must be supplied via native_edge_aa for meaningful per-edge exclusion
            if native_edge_aa is not None:
                for _ in range(20):
                    bad1 = dec_aa1 == nat_src.unsqueeze(-1)
                    bad2 = dec_aa2 == nat_dst.unsqueeze(-1)
                    if not bad1.any() and not bad2.any():
                        break
                    if bad1.any():
                        r = torch.stack([torch.multinomial(aa_probs[b], n_edges * n_decoys, replacement=True)
                                         for b in range(B)], dim=0).view(B, n_edges, n_decoys)
                        dec_aa1 = torch.where(bad1, r, dec_aa1)
                    if bad2.any():
                        r = torch.stack([torch.multinomial(aa_probs[b], n_edges * n_decoys, replacement=True)
                                         for b in range(B)], dim=0).view(B, n_edges, n_decoys)
                        dec_aa2 = torch.where(bad2, r, dec_aa2)
        return torch.stack([dec_aa1, dec_aa2], dim=0)

    # ── sequence distribution ────────────────────────────────────────────────
    if use_c_rand:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.srand(1)
        aa_cpu = aa_type.cpu()
        valid_cpu = [vi.cpu() for vi in valid_indices]
        dec_aa1 = torch.zeros(B, n_edges, n_decoys, dtype=torch.long)
        dec_aa2 = torch.zeros(B, n_edges, n_decoys, dtype=torch.long)
        for e in range(n_edges):
            for d in range(n_decoys):
                for b in range(B):
                    vi = valid_cpu[b]
                    n = len(vi)
                    dec_aa1[b, e, d] = aa_cpu[b, vi[libc.rand() % n].item()]
                    dec_aa2[b, e, d] = aa_cpu[b, vi[libc.rand() % n].item()]
        return torch.stack([dec_aa1.to(device), dec_aa2.to(device)], dim=0)

    max_n = int(seq_lens.max().item())
    vi_table = (~valid_mask.bool()).long().argsort(dim=1, stable=True)[:, :max_n]
    n_valid = seq_lens.view(B, 1, 1)
    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, n_edges, n_decoys)

    def rand_res():
        u    = torch.rand(B, n_edges, n_decoys, device=device)
        slot = (u * n_valid).long().clamp(max=n_valid - 1)
        return vi_table.gather(1, slot.view(B, -1)).view(B, n_edges, n_decoys)

    dec_aa1 = aa_type[b_idx, rand_res()]
    dec_aa2 = aa_type[b_idx, rand_res()]

    if exclude_native and native_edge_aa is not None:
        nat_src, nat_dst = native_edge_aa  # both [B, E]
        for _ in range(20):
            bad1 = (dec_aa1 == nat_src.unsqueeze(-1))
            bad2 = (dec_aa2 == nat_dst.unsqueeze(-1))
            if not bad1.any() and not bad2.any():
                break
            if bad1.any():
                dec_aa1 = torch.where(bad1, aa_type[b_idx, rand_res()], dec_aa1)
            if bad2.any():
                dec_aa2 = torch.where(bad2, aa_type[b_idx, rand_res()], dec_aa2)

    return torch.stack([dec_aa1, dec_aa2], dim=0)


# ---------------------------------------------------------------------------
# Mutational Mixin — all mutational computation methods
# ---------------------------------------------------------------------------

class MutationalMixin:
    """
    Mixin providing mutational frustration computation methods.
    Injected into AWSEM via __bases__ reassignment in awsem.py.
    """

    def compute_tert_frust_mutational(self, batch):
        batch["bg_i"], batch["bg_j"] = self.build_mutational_edges(batch)

        aa_src, aa_dst = self._gather_node(batch["aa_type"], batch["edge_index"])
        native_energy = self.compute_mutational_ixn(batch, torch.stack([aa_src, aa_dst], dim=0))

        B, _, E = batch["edge_index"].shape
        results = self.compute_mutational_decoy_ixns(batch, B, E)

        frustration_index = self.compute_frustration_index(native_energy, results["stats"])

        mut_metrics = None
        mut_losses  = None
        if "mutational_frust_mask" in batch and "mutational_frust_true" in batch:
            true_fi = batch["mutational_frust_true"].to(frustration_index.device)
            mask    = batch["mutational_frust_mask"].to(frustration_index.device).bool()
            mut_metrics = compute_frustration_metrics(
                frustration_index, true_fi, mask,
                threshold=self.coefficients.frustration_thresholds[1],
            )
            pseudo_logits = index_to_pseudo_logits(
                frustration_index, self.coefficients.frustration_thresholds[1]
            )
            idx_loss, cls_loss = _frustration_loss(
                mode=1,
                pred_frustration_index=frustration_index.unsqueeze(-1),
                pred_frustration_class=pseudo_logits,
                true_frustration_index=true_fi,
                seq_mask=mask,
                welltype=batch.get("mutational_welltype"),
                stage="test",
            )
            mut_losses = (idx_loss.item(), cls_loss.item())

        mut_mutations = None
        if self.run_dmfs:
            mut_mutations = {"energies": results["all_energies"], "aas": results["all_aas"]}

        return frustration_index, native_energy, results["stats"], mut_metrics, mut_losses, mut_mutations

    def build_mutational_edges_dense(self, batch):
        """
        Original dense implementation — kept as reference.
        Allocates [B, E, L] intermediates; OOMs on large proteins.
        Use build_mutational_edges (sparse) instead.
        """
        device = batch["aa_type"].device
        B, L = batch["aa_type"].shape
        _, _, E = batch["edge_index"].shape
        cutoff = self.coefficients.tert_frust_cutoff

        with torch.no_grad():
            src = batch["edge_index"][:, 0, :]
            dst = batch["edge_index"][:, 1, :]
            b_idx_E = torch.arange(B, device=device)[:, None].expand(B, E)

            valid_edge = (src >= 0) & (dst >= 0)
            src_safe = src.clamp(min=0)
            dst_safe = dst.clamp(min=0)

            dist = batch["distance"]

            src_dists = dist[b_idx_E, src_safe, :]
            dst_dists = dist[b_idx_E, dst_safe, :]

            k_range = torch.arange(L, device=device).view(1, 1, L)
            seq_mask_k = batch["seq_mask"].unsqueeze(1).expand(B, E, L)
            src_exp = src.unsqueeze(-1).expand(B, E, L)
            dst_exp = dst.unsqueeze(-1).expand(B, E, L)

            valid_edge_exp = valid_edge.unsqueeze(-1).expand(B, E, L)
            bg_i_mask = ((src_dists < cutoff) & seq_mask_k
                         & (k_range != src_exp) & (k_range != dst_exp)
                         & valid_edge_exp)
            bg_i_b, bg_i_e, bg_i_k = bg_i_mask.nonzero(as_tuple=True)
            bg_i_src = src_safe[bg_i_b, bg_i_e]
            bg_i = dict(
                b=bg_i_b, e=bg_i_e,
                dist=dist[bg_i_b, bg_i_src, bg_i_k],
                rho_src=batch["residue_density"][bg_i_b, bg_i_src],
                rho_k=batch["residue_density"][bg_i_b, bg_i_k],
                native_k=batch["aa_type"][bg_i_b, bg_i_k],
            )

            bg_j_mask = ((dst_dists < cutoff) & seq_mask_k
                         & (k_range != src_exp) & (k_range != dst_exp)
                         & valid_edge_exp)
            bg_j_b, bg_j_e, bg_j_k = bg_j_mask.nonzero(as_tuple=True)
            bg_j_dst = dst_safe[bg_j_b, bg_j_e]
            bg_j = dict(
                b=bg_j_b, e=bg_j_e,
                dist=dist[bg_j_b, bg_j_dst, bg_j_k],
                rho_src=batch["residue_density"][bg_j_b, bg_j_dst],
                rho_k=batch["residue_density"][bg_j_b, bg_j_k],
                native_k=batch["aa_type"][bg_j_b, bg_j_k],
            )

        return bg_i, bg_j

    def build_mutational_edges(self, batch):
        """
        Sparse implementation — avoids [B, E, L] intermediates.

        For each native edge e=(i,j), background contacts of i (or j) are its
        neighbors in the contact graph, looked up directly from edge_index rather
        than thresholding a full [B, E, L] distance slice.  Peak allocation is
        O(N_bg) where N_bg << E*L.

        Strategy: symmetrize edge_index into a sorted neighbor list keyed by
        (b*L + r).  For each edge, searchsorted finds the neighbor block for its
        fixed residue; repeat_interleave expands without any [B, E, L] tensor.
        """
        device = batch["aa_type"].device
        B, L = batch["aa_type"].shape
        _, _, E = batch["edge_index"].shape
        dist = batch["distance"]

        with torch.no_grad():
            src = batch["edge_index"][:, 0, :]    # [B, E]
            dst = batch["edge_index"][:, 1, :]    # [B, E]
            valid_edge = (src >= 0) & (dst >= 0)  # [B, E]
            src_safe = src.clamp(min=0)
            dst_safe = dst.clamp(min=0)

            b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, E)
            v_b   = b_idx[valid_edge]       # [N_valid]
            v_src = src_safe[valid_edge]    # [N_valid]
            v_dst = dst_safe[valid_edge]    # [N_valid]
            v_e   = valid_edge.nonzero(as_tuple=False)[:, 1]  # edge idx in [0,E)

            seq_mask = batch["seq_mask"]   # [B, L]

            # Symmetrize contacts: both directions so every residue appears as center.
            nb_b = torch.cat([v_b,   v_b  ])   # [2*N_valid]
            nb_r = torch.cat([v_src, v_dst])    # center residue
            nb_k = torch.cat([v_dst, v_src])    # neighbor

            # edge_index excludes sep < min_seq_distance (typically sep=1 adjacent pairs).
            # The dense background-contact mask includes ALL k within cutoff, so augment
            # with adjacent (sep=1) pairs that edge_index omits.
            cutoff  = self.coefficients.tert_frust_cutoff
            r_adj   = torch.arange(L - 1, device=device)         # [L-1]
            k_adj   = r_adj + 1                                   # adjacent neighbor
            d_adj   = dist[:, r_adj, k_adj]                      # [B, L-1]
            adj_ok  = (seq_mask[:, :-1].bool() & seq_mask[:, 1:].bool()
                       & (d_adj < cutoff))                        # [B, L-1]
            b_adj   = torch.arange(B, device=device).unsqueeze(1).expand(B, L - 1)[adj_ok]
            r_adj_f = r_adj.unsqueeze(0).expand(B, L - 1)[adj_ok]
            k_adj_f = k_adj.unsqueeze(0).expand(B, L - 1)[adj_ok]
            # Add both (center=r, neighbor=k) and (center=k, neighbor=r)
            nb_b = torch.cat([nb_b, b_adj, b_adj])
            nb_r = torch.cat([nb_r, r_adj_f, k_adj_f])
            nb_k = torch.cat([nb_k, k_adj_f, r_adj_f])

            # Sort by (batch, center) → enables searchsorted lookup per edge.
            sort_key = nb_b * L + nb_r
            order    = sort_key.argsort(stable=True)
            sk_sort  = sort_key[order]
            nb_k_s   = nb_k[order]

            def _collect(v_fix, v_oth):
                """
                v_fix, v_oth: [N_valid] — fixed and other endpoint per valid edge.
                Returns bg dict with flat triplets (b, e, k).
                """
                query_key = v_b * L + v_fix   # [N_valid]
                lo = torch.searchsorted(sk_sort, query_key)
                hi = torch.searchsorted(sk_sort, query_key, right=True)
                counts = (hi - lo).clamp(min=0)   # neighbors per edge
                total  = int(counts.sum().item())

                if total == 0:
                    empty = torch.zeros(0, device=device, dtype=torch.long)
                    return dict(b=empty, e=empty,
                                dist=torch.zeros(0, device=device, dtype=self.dtype),
                                rho_src=torch.zeros(0, device=device, dtype=self.dtype),
                                rho_k=torch.zeros(0, device=device, dtype=self.dtype),
                                native_k=empty)

                # Expand each edge's metadata to match its neighbor count.
                rep_b   = v_b.repeat_interleave(counts)    # [total]
                rep_e   = v_e.repeat_interleave(counts)    # [total]
                rep_fix = v_fix.repeat_interleave(counts)  # [total]
                rep_oth = v_oth.repeat_interleave(counts)  # [total]

                # Index into the sorted neighbor table: lo[i] + offset_within_block.
                block_id = torch.arange(len(counts), device=device).repeat_interleave(counts)
                cs = torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                                counts.cumsum(0)[:-1]])
                within = torch.arange(total, device=device) - cs[block_id]
                k_t = nb_k_s[lo[block_id] + within]   # [total]

                # Exclude the fixed residue itself, the other endpoint, and masked residues.
                keep = ((k_t != rep_fix) & (k_t != rep_oth) & seq_mask[rep_b, k_t])
                b_t   = rep_b[keep]
                e_t   = rep_e[keep]
                fix_t = rep_fix[keep]
                k_t   = k_t[keep]

                return dict(
                    b=b_t, e=e_t,
                    dist=dist[b_t, fix_t, k_t],
                    rho_src=batch["residue_density"][b_t, fix_t],
                    rho_k=batch["residue_density"][b_t, k_t],
                    native_k=batch["aa_type"][b_t, k_t],
                )

            bg_i = _collect(v_src, v_dst)
            bg_j = _collect(v_dst, v_src)

        return bg_i, bg_j

    def compute_mutational_ixn(self, batch, res_type_native):
        """
        Full mutational native energy per edge (matches C++):
          E(e) = water(i,j) + elec(i,j) + burial(i) + burial(j)
               + sum_{k<cutoff}[water(i,k)] + sum_{k<cutoff}[water(j,k)]
               + sum_{all k}[elec(i,k)]    + sum_{all k}[elec(j,k)]
        Returns: [B, E]
        """
        B, _, E = batch["edge_index"].shape
        device = batch["edge_index"].device

        aa_src = res_type_native[0]
        aa_dst = res_type_native[1]
        rho_src, rho_dst = self._gather_node(batch["residue_density"], batch["edge_index"])
        edge_distances = self._gather_edge_distances(batch["distance"], batch["edge_index"])

        water_direct = self.compute_water_energy(
            batch, edge_distances, rho_src, rho_dst, aa_src, aa_dst)
        elec_direct = self.calculate_electrostatics(aa_src, aa_dst, edge_distances)
        burial_i = self.compute_burial_energy(batch, aa_src, rho_src)
        burial_j = self.compute_burial_energy(batch, aa_dst, rho_dst)

        E_total = (water_direct + elec_direct + burial_i + burial_j).to(self.dtype)

        if "bg_i" in batch and "bg_j" in batch:
            E_total = E_total + self._scatter_bg_water(batch, batch["bg_i"], aa_src, B, E)
            E_total = E_total + self._scatter_bg_water(batch, batch["bg_j"], aa_dst, B, E)
            field_i = self.compute_bg_elec_field(batch, is_src=True)
            field_j = self.compute_bg_elec_field(batch, is_src=False)
            q_src = self.charge_lookup[aa_src].to(self.dtype)
            q_dst = self.charge_lookup[aa_dst].to(self.dtype)
            E_total = E_total + q_src * field_i * (q_src != 0).to(self.dtype)
            E_total = E_total + q_dst * field_j * (q_dst != 0).to(self.dtype)

        return E_total

    @torch.no_grad()
    def compute_bg_elec_field(self, batch, is_src: bool,
                               k_chunk: int = 512) -> torch.Tensor:
        """
        Per-edge electrostatic field from ALL residues k!=i, k!=j.
        Returns: [B, E] float64

        Precompute-subtract: total_field[i] = Σ_{k≠i} q_k/d_ik * exp(-d/sl),
        then bg_field(i,j) = k_PlusMinus * (total_field[i] - edge_kernel(i,j)).
        Chunks over k so peak allocation is [B, L, k_chunk] instead of [B, E, L].
        """
        B, L = batch["aa_type"].shape
        _, _, E = batch["edge_index"].shape
        device = batch["edge_index"].device

        if not self.coefficients.use_electrostatics:
            return torch.zeros(B, E, device=device, dtype=self.dtype)

        src = batch["edge_index"][:, 0, :]
        dst = batch["edge_index"][:, 1, :]
        fixed_res = src if is_src else dst
        other_res  = dst if is_src else src

        q    = self.charge_lookup[batch["aa_type"]].to(self.dtype)  # [B, L]
        mask = batch["seq_mask"].to(self.dtype)                       # [B, L]
        dist = batch["distance"].to(self.dtype)                       # [B, L, L]
        sl   = self.coefficients.screening_length
        i_idx = torch.arange(L, device=device)

        # Step 1: total field at each residue i, summed over all valid k≠i.
        total_field = torch.zeros(B, L, device=device, dtype=self.dtype)
        for k0 in range(0, L, k_chunk):
            k1    = min(k0 + k_chunk, L)
            k_idx = i_idx[k0:k1]                          # [C]
            d_ik  = dist[:, :, k0:k1].clamp(min=1e-6)    # [B, L, C]
            q_k   = q[:, k0:k1]                            # [B, C]
            m_k   = mask[:, k0:k1]                         # [B, C]
            kernel = (q_k[:, None, :] / d_ik
                      * torch.exp(-d_ik / sl)
                      * m_k[:, None, :])                   # [B, L, C]
            not_self = (i_idx[:, None] != k_idx[None, :]).unsqueeze(0)
            total_field += (kernel * not_self).sum(dim=-1)

        # Step 2: gather field at fixed_res, subtract the direct edge contribution.
        b_idx = torch.arange(B, device=device)[:, None].expand(B, E)
        fr    = fixed_res.clamp(0)
        oth   = other_res.clamp(0)

        field_fr = total_field[b_idx, fr]                             # [B, E]
        d_edge   = dist[b_idx, fr, oth].clamp(min=1e-6)
        subtract = q[b_idx, oth] / d_edge * torch.exp(-d_edge / sl)  # [B, E]

        valid = (src >= 0) & (dst >= 0)
        result = self.coefficients.k_PlusMinus * (field_fr - subtract) * valid.to(self.dtype)
        return result

    def _scatter_bg_water(self, batch, bg, src_type, B, E):
        """Water energy for k < cutoff background edges, scattered to [B, E]."""
        N_bg = bg['dist'].shape[0]
        device = bg['dist'].device

        result_flat = torch.zeros(B * E, device=device, dtype=self.dtype)
        if N_bg == 0:
            return result_flat.view(B, E)

        src_t = src_type[bg['b'], bg['e']]
        k_t   = bg['native_k']

        water_bg = self.compute_water_energy(
            batch, bg['dist'], bg['rho_src'], bg['rho_k'], src_t, k_t)

        result_flat.scatter_add_(0, bg['b'] * E + bg['e'], water_bg.to(self.dtype))
        return result_flat.view(B, E)

    def _scatter_bg_water_decoy(self, batch, bg, decoy_type, B, E, n_decoys,
                                 bg_chunk: int = 1 << 20):
        """Water energy for decoy background edges, scattered to [B, E, n_decoys].

        Processes background contacts in chunks to keep peak GPU allocation bounded.
        The default bg_chunk (1 M) is used on CPU or when VRAM is plentiful; on
        CUDA the chunk is automatically clamped based on currently free VRAM so that
        the ~12 simultaneous [chunk, n_decoys] float64 tensors created inside
        compute_water_energy don't exceed one-quarter of free memory.  The chunk
        never falls below 8 192 entries regardless of pressure.
        """
        N_bg = bg['dist'].shape[0]
        device = bg['dist'].device

        # compute_water_energy holds ~12 simultaneous [chunk, n_decoys] tensors
        # (sigma, theta, gamma lookups, intermediates).  Cap so they fit in VRAM.
        if device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            bytes_per_entry = n_decoys * 12 * 8  # 12 simultaneous float64 tensors
            max_chunk = max(8192, int(free_bytes // (4 * bytes_per_entry)))
            bg_chunk = min(bg_chunk, max_chunk)

        result = torch.zeros(B * E, n_decoys, device=device, dtype=self.dtype)
        if N_bg == 0:
            return result.view(B, E, n_decoys)

        for c0 in range(0, N_bg, bg_chunk):
            c1 = min(c0 + bg_chunk, N_bg)
            sl = slice(c0, c1)

            src_t = decoy_type[bg['b'][sl], bg['e'][sl], :]           # [C, n_decoys]
            k_t   = bg['native_k'][sl].unsqueeze(-1).expand(c1 - c0, n_decoys)

            rik_exp     = bg['dist'][sl].unsqueeze(-1).expand(c1 - c0, n_decoys)
            rho_src_exp = bg['rho_src'][sl].unsqueeze(-1).expand(c1 - c0, n_decoys)
            rho_k_exp   = bg['rho_k'][sl].unsqueeze(-1).expand(c1 - c0, n_decoys)

            water_bg = self.compute_water_energy(
                batch, rik_exp, rho_src_exp, rho_k_exp, src_t, k_t)

            flat_idx = (bg['b'][sl] * E + bg['e'][sl]).unsqueeze(-1).expand(c1 - c0, n_decoys)
            result.scatter_add_(0, flat_idx, water_bg.to(self.dtype))

        return result.view(B, E, n_decoys)

    # ------------------------------------------------------------------
    # Precomputation helpers: evaluate energy over all 20 AA types once
    # so _eval_decoys only needs a .gather() call per decoy sample.
    # ------------------------------------------------------------------

    def _precompute_bg_water_20(self, batch, bg, B, E):
        """Background water energy for all 20 AA decoy types at every edge.

        Evaluates compute_water_energy N_bg × 20 times instead of
        N_bg × n_decoys times.  _eval_decoys retrieves the result for a
        specific decoy type with a single .gather(-1, d_src.long()) call.

        Returns: [B, E, 20]
        """
        N_bg   = bg['dist'].shape[0]
        device = bg['dist'].device
        result = torch.zeros(B * E, 20, device=device, dtype=self.dtype)
        if N_bg == 0:
            return result.view(B, E, 20)

        all_t    = torch.arange(20, device=device, dtype=torch.long)
        flat_idx = bg['b'] * E + bg['e']
        rows     = max(1, DECOY_CHUNK_ELEMENTS // 20)
        for start in range(0, N_bg, rows):
            sl = slice(start, start + rows)
            n  = flat_idx[sl].shape[0]
            w = self.compute_water_energy(
                batch,
                bg['dist'][sl].unsqueeze(-1).expand(n, 20),
                bg['rho_src'][sl].unsqueeze(-1).expand(n, 20),
                bg['rho_k'][sl].unsqueeze(-1).expand(n, 20),
                all_t.unsqueeze(0).expand(n, 20),
                bg['native_k'][sl].unsqueeze(-1).expand(n, 20),
            )
            result.scatter_add_(0, flat_idx[sl].unsqueeze(-1).expand(n, 20), w.to(self.dtype))
        return result.view(B, E, 20)

    def _precompute_burial_20(self, batch, B):
        """Burial energy for all 20 AA types at every residue.

        Returns: [B, L, 20]
        """
        L      = batch["aa_type"].shape[1]
        device = batch["aa_type"].device
        all_t  = torch.arange(20, device=device, dtype=torch.long)
        all_t_exp = all_t.view(1, 1, 20).expand(B, L, 20)
        rho_exp   = batch["residue_density"].unsqueeze(-1).expand(B, L, 20)
        return self.compute_burial_energy(batch, all_t_exp, rho_exp)  # [B, L, 20]

    def _water_energy_from_precomputed(self, batch, sigma_prot, sigma_wat,
                                        theta_dir, theta_med, d_src, d_dst):
        """Direct water energy using structure-precomputed sigma/theta.

        Avoids allocating large [B, E, n_decoys] sigma and theta tensors by
        keeping sigma/theta as [B, E] and broadcasting over the decoy dim.

        sigma_prot, sigma_wat, theta_dir, theta_med : [B, E]
        d_src, d_dst                                : [B, E, D]
        Returns                                     : [B, E, D]
        """
        D  = d_src.shape[-1]
        wg = self.get_water_gamma(batch)             # [n_wells, 21, 21, 2]

        s_flat = d_src.long().reshape(-1, D)          # [B*E, D]
        d_flat = d_dst.long().reshape(-1, D)

        wg_dir = wg[0, s_flat, d_flat, :].mean(-1).reshape(*d_src.shape)  # [B, E, D]
        wg_pm  = wg[1, s_flat, d_flat, 0].reshape(*d_src.shape)
        wg_wm  = wg[1, s_flat, d_flat, 1].reshape(*d_src.shape)

        sp = sigma_prot.unsqueeze(-1)   # [B, E, 1] → broadcasts over D
        sw = sigma_wat.unsqueeze(-1)
        td = theta_dir.unsqueeze(-1)
        tm = theta_med.unsqueeze(-1)

        sigma_gam_med = sp * wg_pm + sw * wg_wm
        return -(wg_dir * td + sigma_gam_med * tm)

    def _compute_gly_edge_data(self, batch, B, E):
        """
        Pre-compute per-edge alternative distances used by the GLY pair correction.

        For mutational decoys the backbone geometry is fixed, but the correct
        CB/CA reference point depends on the decoy AA type:
          native=GLY  → decoy=non-GLY : should use vCB (ideal CB placed on native backbone)
          native≠GLY  → decoy=GLY     : should use CA  (GLY has no CB)

        Returns:
          gly_src, gly_dst  [B, E] bool  — which endpoints are natively GLY
          d_vcb_src         [B, E]       — |vCB_src − native_dst|
          d_ca_src          [B, E]       — |CA_src  − native_dst|
          d_vcb_dst         [B, E]       — |native_src − vCB_dst|
          d_ca_dst          [B, E]       — |native_src − CA_dst|
          d_ca_ca           [B, E]       — |CA_src − CA_dst|  (both endpoints swap to GLY)
        """
        from ..structure.geometry import compute_ideal_vcb, get_CB_coords

        device = batch["edge_index"].device
        GLY_IDX = 5

        src = batch["edge_index"][:, 0, :].clamp(min=0)  # [B, E]
        dst = batch["edge_index"][:, 1, :].clamp(min=0)  # [B, E]
        b_idx = torch.arange(B, device=device)[:, None].expand(B, E)

        native_aa = batch["aa_type"]
        gly_src = (native_aa[b_idx, src] == GLY_IDX)  # [B, E]
        gly_dst = (native_aa[b_idx, dst] == GLY_IDX)

        atom_pos = batch["atom_positions"]
        CA  = atom_pos[:, :, 1].to(self.dtype)                            # [B, L, 3]
        vCB = compute_ideal_vcb(batch, dtype=self.dtype)                   # [B, L, 3]
        nat = get_CB_coords(atom_pos[:, :, :5], native_aa).to(self.dtype)  # [B, L, 3]

        ca_s  = CA[b_idx, src];   ca_d  = CA[b_idx, dst]   # [B, E, 3]
        vcb_s = vCB[b_idx, src];  vcb_d = vCB[b_idx, dst]
        nat_s = nat[b_idx, src];  nat_d = nat[b_idx, dst]

        d_vcb_src = (vcb_s - nat_d).norm(dim=-1)   # |vCB_src − native_dst|
        d_ca_src  = (ca_s  - nat_d).norm(dim=-1)   # |CA_src  − native_dst|
        d_vcb_dst = (nat_s - vcb_d).norm(dim=-1)   # |native_src − vCB_dst|
        d_ca_dst  = (nat_s - ca_d ).norm(dim=-1)   # |native_src − CA_dst|
        d_ca_ca   = (ca_s  - ca_d ).norm(dim=-1)   # |CA_src − CA_dst|

        return gly_src, gly_dst, d_vcb_src, d_ca_src, d_vcb_dst, d_ca_dst, d_ca_ca

    def compute_mutational_decoy_energies(self, batch, B, E, decoy_aa):
        """
        Energies of the given decoy identities on every edge.
        decoy_aa: [2, B, E, D] -> returns [B, E, D]
        """
        device   = batch["edge_index"].device
        GLY_IDX  = 5

        rho_src, rho_dst = self._gather_node(batch["residue_density"], batch["edge_index"])
        edge_dist = self._gather_edge_distances(batch["distance"], batch["edge_index"])

        # --- Structure-fixed precomputation (shared across all n_decoys) ---

        # Per-edge residue indices for burial lookup
        src_r   = batch["edge_index"][:, 0, :].clamp(min=0)                 # [B, E]
        dst_r   = batch["edge_index"][:, 1, :].clamp(min=0)                 # [B, E]
        b_idx_e = torch.arange(B, device=device).unsqueeze(1).expand(B, E)  # [B, E]

        # Burial energy for all 20 AA types at every residue: [B, L, 20]
        burial_20 = self._precompute_burial_20(batch, B)

        # Background water [B, E, 20] and electrostatic field [B, E]
        if "bg_i" in batch and "bg_j" in batch:
            bg_water_i_20 = self._precompute_bg_water_20(batch, batch["bg_i"], B, E)
            bg_water_j_20 = self._precompute_bg_water_20(batch, batch["bg_j"], B, E)
            field_i = self.compute_bg_elec_field(batch, is_src=True)
            field_j = self.compute_bg_elec_field(batch, is_src=False)
        else:
            bg_water_i_20 = bg_water_j_20 = field_i = field_j = None

        use_gly_correction = getattr(self.coefficients, 'gly_pair_correction', False)
        if use_gly_correction:
            gly_src, gly_dst, d_vcb_src, d_ca_src, d_vcb_dst, d_ca_dst, d_ca_ca = \
                self._compute_gly_edge_data(batch, B, E)

            def _get_dist(d_src, d_dst):
                D = d_src.shape[-1]
                dg_s = (d_src == GLY_IDX)  # [B, E, D]
                dg_d = (d_dst == GLY_IDX)
                gs = gly_src.unsqueeze(-1).expand(B, E, D)
                gd = gly_dst.unsqueeze(-1).expand(B, E, D)
                dist = edge_dist.unsqueeze(-1).expand(B, E, D).clone()
                # native GLY src → non-GLY decoy: vCB replaces CA at src
                dist = torch.where(gs & ~dg_s & ~dg_d, d_vcb_src.unsqueeze(-1).expand(B, E, D), dist)
                # native GLY dst → non-GLY decoy: vCB replaces CA at dst
                dist = torch.where(~dg_s & gd & ~dg_d, d_vcb_dst.unsqueeze(-1).expand(B, E, D), dist)
                # native non-GLY src → GLY decoy: CA replaces CB at src
                dist = torch.where(~gs & dg_s & ~dg_d, d_ca_src.unsqueeze(-1).expand(B, E, D), dist)
                # native non-GLY dst → GLY decoy: CA replaces CB at dst
                dist = torch.where(~dg_s & ~gd & dg_d, d_ca_dst.unsqueeze(-1).expand(B, E, D), dist)
                # both non-GLY native → both GLY decoy: CA−CA
                dist = torch.where(~gs & dg_s & ~gd & dg_d, d_ca_ca.unsqueeze(-1).expand(B, E, D), dist)
                return dist

            # With gly correction, dist_x varies per decoy → sigma/theta cannot be precomputed
            direct_sigma_prot = direct_sigma_wat = direct_theta_dir = direct_theta_med = None
        else:
            def _get_dist(d_src, d_dst):
                D = d_src.shape[-1]
                return edge_dist.unsqueeze(-1).expand(B, E, D)

            # Precompute direct-term sigma and theta (structure-only): [B, E]
            kw     = self.coefficients.water_kappa_sigma
            thresh = self.coefficients.threshold
            s_wi = 1.0 - torch.tanh(kw * (rho_src.to(self.dtype) - thresh))
            s_wj = 1.0 - torch.tanh(kw * (rho_dst.to(self.dtype) - thresh))
            direct_sigma_wat  = 0.25 * s_wi * s_wj
            direct_sigma_prot = 1.0 - direct_sigma_wat

            d_e   = edge_dist.to(self.dtype)
            kappa = self.coefficients.water_kappa
            rmin  = self.coefficients.well_rmin.to(device, dtype=self.dtype)
            rmax  = self.coefficients.well_rmax.to(device, dtype=self.dtype)
            direct_theta_dir = 0.25 * (1 + torch.tanh(kappa * (d_e - rmin[0]))) * \
                                       (1 + torch.tanh(kappa * (rmax[0] - d_e)))
            direct_theta_med = 0.25 * (1 + torch.tanh(kappa * (d_e - rmin[1]))) * \
                                       (1 + torch.tanh(kappa * (rmax[1] - d_e)))

        def _eval_decoys(decoy_aa_tensor):
            D     = decoy_aa_tensor.shape[-1]
            d_src = decoy_aa_tensor[0]   # [B, E, D]
            d_dst = decoy_aa_tensor[1]   # [B, E, D]
            dist_x = _get_dist(d_src, d_dst)

            # Burial: lookup from precomputed [B, L, 20] → [B, E, D]
            burial_src = burial_20[b_idx_e, src_r, :].gather(-1, d_src.long())
            burial_dst = burial_20[b_idx_e, dst_r, :].gather(-1, d_dst.long())

            # Direct water energy
            if use_gly_correction:
                rho_s_x = rho_src.unsqueeze(-1).expand(B, E, D)
                rho_d_x = rho_dst.unsqueeze(-1).expand(B, E, D)
                direct_water = self.compute_water_energy(
                    batch, dist_x, rho_s_x, rho_d_x, d_src, d_dst)
            else:
                direct_water = self._water_energy_from_precomputed(
                    batch, direct_sigma_prot, direct_sigma_wat,
                    direct_theta_dir, direct_theta_med, d_src, d_dst)

            E_d = (direct_water
                   + self.calculate_electrostatics(d_src, d_dst, dist_x)
                   + burial_src + burial_dst)

            if bg_water_i_20 is not None:
                # Background water: lookup from precomputed [B, E, 20] → [B, E, D]
                E_d = E_d + bg_water_i_20.gather(-1, d_src.long())
                E_d = E_d + bg_water_j_20.gather(-1, d_dst.long())
                q_s = self.charge_lookup[d_src].to(self.dtype)
                q_d = self.charge_lookup[d_dst].to(self.dtype)
                E_d = E_d + q_s * field_i.unsqueeze(-1) * (q_s != 0).to(self.dtype)
                E_d = E_d + q_d * field_j.unsqueeze(-1) * (q_d != 0).to(self.dtype)
            return E_d

        # Decoys are independent: evaluate them in chunks to bound the [B, E, chunk] intermediates.
        D = decoy_aa.shape[-1]
        chunk = max(1, DECOY_CHUNK_ELEMENTS // max(B * E, 1))
        if D <= chunk:
            return _eval_decoys(decoy_aa)
        energies = None
        for start in range(0, D, chunk):
            part = _eval_decoys(decoy_aa[..., start:start + chunk])
            if energies is None:
                energies = torch.empty(B, E, D, dtype=part.dtype, device=part.device)
            energies[..., start:start + chunk] = part
        return energies

    def compute_mutational_pair_energies(self, batch, B, E):
        """Energies of all 20 x 20 identity pairs on every edge: [B, E, 20, 20]."""
        aa_grid = torch.arange(20, device=batch["edge_index"].device)
        pairs   = torch.cartesian_prod(aa_grid, aa_grid).T  # [2, 400], column a*20+b = (a, b)
        exh_aa  = pairs.view(2, 1, 1, 400).expand(2, B, E, 400)
        return self.compute_mutational_decoy_energies(batch, B, E, exh_aa).view(B, E, 20, 20)

    def compute_mutational_decoy_ixns(self, batch, B, E):
        """
        Mean and std of the decoy energy distribution on every edge.
        Exact (weighted over all 400 pairs) unless use_c_rand is set, in which
        case n_decoys pairs are sampled from the C random stream.
        Returns {"stats": [B, E, 2]} plus "all_energies" [B, E, 400] and
        "all_aas" [2, 400] when run_dmfs is set.
        """
        device = batch["edge_index"].device
        decoy_dist = getattr(self.coefficients, 'tert_frust_decoy_distribution', 'sequence')
        if getattr(self.coefficients, 'tert_frust_exclude_native', False) and decoy_dist == 'sequence':
            decoy_dist = 'sequence_exclude_native'
        native_src, native_dst = self._gather_node(batch["aa_type"], batch["edge_index"])  # [B, E]

        table = None
        if not self.use_c_rand or self.run_dmfs:
            table = self.compute_mutational_pair_energies(batch, B, E)

        if self.use_c_rand:
            decoy_aa = sample_mutational_decoys(
                aa_type=batch["aa_type"],
                seq_mask=batch["seq_mask"],
                n_edges=E,
                n_decoys=self.n_decoys,
                use_c_rand=True,
                native_edge_aa=(native_src, native_dst) if decoy_dist.endswith('_exclude_native') else None,
                decoy_distribution=decoy_dist,
            )
            stats = decoy_statistics(self.compute_mutational_decoy_energies(batch, B, E, decoy_aa))
        else:
            w_src = decoy_aa_weights(batch["aa_type"], batch["seq_mask"], native_src, decoy_dist, table.dtype)
            w_dst = decoy_aa_weights(batch["aa_type"], batch["seq_mask"], native_dst, decoy_dist, table.dtype)
            stats = decoy_statistics(table, w_src[..., :, None] * w_dst[..., None, :], dim=(-2, -1))

        out = {"stats": stats}
        if self.run_dmfs:
            aa_grid = torch.arange(20, device=device)
            out["all_energies"] = table.reshape(B, E, 400)
            out["all_aas"]      = torch.cartesian_prod(aa_grid, aa_grid).T
        return out


# ---------------------------------------------------------------------------
# TODO: engine refactor sketch (not wired — existing code above is live)
#
# Once algorithm/energy/ and algorithm/decoys/ are fully wired, the mixin
# method above collapses to:
#
#   def compute_mutational_decoy_energies(self, batch, B, E):
#       energy_fn = AWsemEnergy(self)           # or MJEnergy, OpenMMEnergy, …
#       decoy_aa  = sample_mutational_decoys(
#           aa_type=batch["aa_type"], seq_mask=batch["seq_mask"],
#           n_edges=E, n_decoys=self.n_decoys, use_c_rand=self.use_c_rand,
#       )
#       decoy_info = {"decoy_aa": decoy_aa}
#       E_decoy = energy_fn.score_decoys(batch, decoy_info)   # [B, E, n_decoys]
#
#       if self.run_dmfs:
#           aa_grid = torch.arange(20, device=batch["edge_index"].device)
#           g1, g2  = torch.meshgrid(aa_grid, aa_grid, indexing="ij")
#           exh_aa  = torch.stack([g1, g2]).flatten(1).view(2,1,1,400).expand(2,B,E,400).clone()
#           batch["_mut_all_energies"] = energy_fn.score_decoys(batch, {"decoy_aa": exh_aa})
#           batch["_mut_all_aas"]      = torch.stack([g1.flatten(), g2.flatten()])
#
#       return E_decoy
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mutational Frustratometer
# ---------------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Forward and output methods (used by MutationalFrustratometer)
    # ------------------------------------------------------------------

    def forward(self, batch):
        B = batch["aa_type"].shape[0]
        if B == 1:
            return self._forward_single(batch)
        if self.use_c_rand:
            results = [self._forward_single(self._slice_batch(batch, i)) for i in range(B)]
            return self._merge_results(results)
        return self._forward_single(batch)

    def _forward_single(self, batch):
        with torch.no_grad():
            self.compute_densities(batch)
            fi, nat_e, stats, mut_metrics, mut_losses, mut_mutations = \
                self.compute_tert_frust_mutational(batch)
        result = {
            "mutational":                 fi,
            "_mutational_native_energy":  nat_e,
            "_mutational_decoy_stats":    stats,
        }
        if mut_mutations is not None: result["_mutational_mutations"] = mut_mutations
        if mut_metrics   is not None: result["_mutational_metrics"]   = mut_metrics
        if mut_losses    is not None: result["_mutational_losses"]    = mut_losses
        return result

    def _write_extra_outputs(self, batch, result, frust_bases, names, done_dirs,
                             residue_index, chain_index, output_dir):
        if self.run_dmfs:
            self._write_mutation_files(batch, result, frust_bases, names,
                                       residue_index, chain_index)

    def _write_mutation_files(self, batch, result, frust_bases, names,
                              residue_index, chain_index):
        """Write per-edge × 20×20 mutation energy tables."""
        import pandas as pd
        from ..structure.residue_constants import ID_TO_AA
        from ..structure.protein import PDB_CHAIN_IDS

        mut_data = result.get("_mutational_mutations")
        if mut_data is None:
            return

        _cid_map_list = batch.get("chain_id_map")

        for i, (frust_base, name) in enumerate(zip(frust_bases, names)):
            all_e   = mut_data["energies"][i].cpu().float()
            all_aas = mut_data["aas"].cpu()
            nat_e   = result["_mutational_native_energy"][i].cpu().float()
            ei      = batch["edge_index"][i].cpu()  # all_e/nat_e are on CPU; a CUDA mask can't index them
            valid_e = (ei >= 0).all(dim=0)
            ei_v    = ei[:, valid_e]
            all_e   = all_e[valid_e]
            nat_e   = nat_e[valid_e]

            res1 = (residue_index[i, ei_v[0]].cpu().numpy() if residue_index is not None
                    else (ei_v[0] + 1).numpy())
            res2 = (residue_index[i, ei_v[1]].cpu().numpy() if residue_index is not None
                    else (ei_v[1] + 1).numpy())
            aa_src = batch["aa_type"][i, ei_v[0]].cpu()
            aa_dst = batch["aa_type"][i, ei_v[1]].cpu()
            _cid_map_i = _cid_map_list[i] if _cid_map_list is not None else None
            ch1 = ([(_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
                    for c in chain_index[i, ei_v[0]].cpu().tolist()]
                   if chain_index is not None else ["A"] * ei_v.shape[1])
            ch2 = ([(_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
                    for c in chain_index[i, ei_v[1]].cpu().tolist()]
                   if chain_index is not None else ["A"] * ei_v.shape[1])

            rows = []
            for e in range(all_e.shape[0]):
                aa1_nat = ID_TO_AA.get(int(aa_src[e]), "X")
                aa2_nat = ID_TO_AA.get(int(aa_dst[e]), "X")
                for d in range(400):
                    rows.append({
                        "Res1": int(res1[e]), "Res2": int(res2[e]),
                        "ChainRes1": ch1[e],  "ChainRes2": ch2[e],
                        "AA1_native": aa1_nat, "AA2_native": aa2_nat,
                        "AA1_mutant": ID_TO_AA.get(int(all_aas[0, d]), "X"),
                        "AA2_mutant": ID_TO_AA.get(int(all_aas[1, d]), "X"),
                        "NativeEnergy": float(nat_e[e]),
                        "MutantEnergy": float(all_e[e, d]),
                    })
            pd.DataFrame(rows).to_csv(
                f"{frust_base}.{name}_mutational_mutations", sep=" ", index=False)


# ---------------------------------------------------------------------------
# Lazy MutationalFrustratometer (defers FrustratometerBase / lightning)
# ---------------------------------------------------------------------------

_MutationalFrustratometer = None


def _get_MutationalFrustratometer():
    global _MutationalFrustratometer
    if _MutationalFrustratometer is not None:
        return _MutationalFrustratometer
    from ..base import FrustratometerBase

    class MutationalFrustratometer(FrustratometerBase):
        """All mutational frustration logic.  Computation via MutationalMixin;
        Lightning training wrapper via FrustratometerBase.

        AWSEM already holds every mode mixin, so the methods MutationalMixin shares
        with other classes in the MRO are taken from it explicitly."""

        forward               = MutationalMixin.forward
        _forward_single       = MutationalMixin._forward_single
        _write_extra_outputs  = MutationalMixin._write_extra_outputs
        _write_mutation_files = MutationalMixin._write_mutation_files

        def __init__(self, coefficients: Coefficients, output_dir=None,
                     dtype: torch.dtype = torch.float32):
            coefficients = dataclasses.replace(coefficients, mode="mutational")
            super().__init__(coefficients, output_dir=output_dir, dtype=dtype)

        def _check_r_threshold(self, batch, result, r_min: float = 0.99, acc_min: float = 0.9):
            mode_cfg = [
                ("mutational", "mutational", "mutational_frust_true",
                 "mutational_frust_mask", self.coefficients.frustration_thresholds[1]),
            ]
            self._run_mode_check(batch, result, mode_cfg, r_min, acc_min)

    _MutationalFrustratometer = MutationalFrustratometer
    return _MutationalFrustratometer


def __getattr__(name):
    if name == "MutationalFrustratometer":
        return _get_MutationalFrustratometer()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option('--structure', '-s', required=True, multiple=True,
              help='Structure file(s), directory, or text file of paths [PDB/mmCIF]')
@click.option('--output_dir', '-o', default=None)
@click.option('--ndecoys', '-n', default=None, type=int)
@click.option('--cutoff', default=None, type=float)
@click.option('--min_seq_dist', default=None, type=int)
@click.option('--config', '-c', default=None, type=click.Path(exists=True))
@click.option('--set', 'overrides', multiple=True, metavar='KEY=VALUE')
@click.option('--compare_to_original', is_flag=True, default=False)
@click.option('--run_dmfs', is_flag=True, default=False,
              help='Write per-edge×20×20 mutation energy tables.')
@click.option('--use_c_rand', is_flag=True, default=False)
@click.option('--dtype', default='float32',
              type=click.Choice(['float32', 'float64'], case_sensitive=False))
@click.option('--no_electrostatics', is_flag=True, default=False)
@click.option('--wandb', 'use_wandb', is_flag=True, default=False)
@click.option('--wandb_project', default='frustratometer')
@click.option('--wandb_run_name', default='mutational_baseline')
@click.option('--wandb_offline', is_flag=True, default=False)
@click.option('--batch_size', '-b', default=1, type=int)
@click.option('--num_workers', default=4, type=int)
@click.option('--use_completed_pdb', is_flag=True, default=False)
def cli(structure, output_dir, ndecoys, cutoff, min_seq_dist,
        config, overrides, compare_to_original, use_completed_pdb, use_c_rand, dtype,
        no_electrostatics, run_dmfs, use_wandb, wandb_project, wandb_run_name,
        wandb_offline, batch_size, num_workers):
    cfg_dict = {"mode": "mutational"}
    if config:
        import yaml
        with open(config) as fh:
            cfg_dict.update(yaml.safe_load(fh) or {})
    cfg_dict["mode"] = "mutational"
    if ndecoys is not None:          cfg_dict["tert_frust_ndecoys"] = ndecoys
    if cutoff is not None:           cfg_dict["tert_frust_cutoff"]  = cutoff
    if min_seq_dist is not None:     cfg_dict["min_seq_distance"]   = min_seq_dist
    if run_dmfs:               cfg_dict["run_dmfs"]      = True
    if no_electrostatics:            cfg_dict["use_electrostatics"] = False
    for kv in overrides:
        if '=' not in kv:
            raise click.BadParameter(f"KEY=VALUE required, got: {kv!r}")
        k, raw = kv.split('=', 1)
        cfg_dict[k] = Coefficients._parse_cli_value(raw)
    cfg_dict["mode"] = "mutational"

    coefficients = Coefficients.from_dict(cfg_dict)
    torch_dtype  = torch.float64 if dtype == 'float64' else torch.float32

    from torch.utils.data import DataLoader
    from lightning.pytorch import Trainer
    from ..data.dataset import FrustratometerDataset, collate_frustratometer
    MutationalFrustratometer = _get_MutationalFrustratometer()

    dataset = FrustratometerDataset(structure, coefficients,
                                    compare_to_original=compare_to_original,
                                    use_completed_pdb=use_completed_pdb,
                                    dtype=torch_dtype)
    model = MutationalFrustratometer(coefficients, output_dir=output_dir, dtype=torch_dtype)
    model.use_c_rand = use_c_rand
    if torch.cuda.is_available():
        model = model.to("cuda")

    logger = True
    if use_wandb:
        from lightning.pytorch.loggers import WandbLogger
        import os
        if wandb_offline:
            os.environ.setdefault("WANDB_MODE", "offline")
        logger = WandbLogger(project=wandb_project, name=wandb_run_name, offline=wandb_offline)

    trainer = Trainer(logger=logger)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=collate_frustratometer,
                            prefetch_factor=2 if num_workers > 0 else None,
                            persistent_workers=num_workers > 0,
                            pin_memory=torch.cuda.is_available(), shuffle=False)

    if compare_to_original:
        trainer.test(model, dataloader)
    else:
        trainer.predict(model, dataloader)


if __name__ == "__main__":
    cli()

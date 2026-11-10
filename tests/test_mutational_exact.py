"""Contract: exact (decoy-free) mutational frustration.

Mutational frustration compares the native energy of each contact (i, j) with the energies of decoys in which both
residues of the contact are mutated while the structure stays fixed. Exact statistics evaluate all 20 x 20 amino-acid
pairs of every contact and weight them with the decoy amino-acid distribution: the limit of infinitely many sampled
decoys, with the population standard deviation. Single-residue statistics are exact the same way over the 20 amino
acids of each residue. Both modes sample tert_frust_ndecoys decoys only with use_c_rand, which reproduces the C random
stream of the AWSEM LAMMPS code.

The decoy weights of a protein count its valid standard residues (inside seq_mask, aa_type < 20): "sequence" gives
n_a / N, "pseudocount" (n_a + 1) / (N + 20), and "*_exclude_native" sets the native amino acid of the position to 0
and renormalizes. The tests compute them from this definition, independently of the package.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from click.testing import CliRunner

from pytorch_frustratometer import Frustratometer, FrustratometerDataset, collate_frustratometer
from pytorch_frustratometer.algorithm import decoy_statistics
from pytorch_frustratometer.algorithm import mutational
from pytorch_frustratometer.algorithm.mutational import sample_mutational_decoys
from pytorch_frustratometer.cli import cli

DATA = Path(__file__).parent / "data"
STRUCTURE = DATA / "1r69.pdb"
DISTRIBUTIONS = ("sequence", "sequence_exclude_native", "pseudocount", "pseudocount_exclude_native")


def model_for(mode: str = "mutational", energy_function: str = "awsem", **settings):
    return Frustratometer(mode, energy_function=energy_function, dtype=torch.float64, **settings)


def energy_model(model):
    """The module that computes the energies: Frustratometer3 delegates them to its AWSEM model."""
    return getattr(model, "awsem", model)


def load(model, *paths: Path) -> dict:
    """One batch of the given structures, in the given order."""
    items = [FrustratometerDataset([str(path)], model.coefficients, dtype=torch.float64)[0] for path in paths]
    return collate_frustratometer(items)


def run(model, *paths: Path, seed: int = 0) -> tuple[dict, dict]:
    """The result of a forward pass under a seed, and the batch, which keeps the densities and contacts."""
    batch = load(model, *(paths or (STRUCTURE,)))
    torch.manual_seed(seed)
    with torch.no_grad():
        result = model(batch)
    return result, batch


def valid_contacts(batch: dict) -> torch.Tensor:  # [B, E]
    return (batch["edge_index"] >= 0).all(dim=1)


def contact_natives(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:  # [B, E] each
    edges = batch["edge_index"].clamp(min=0)
    return (batch["aa_type"].gather(1, edges[:, 0]), batch["aa_type"].gather(1, edges[:, 1]))


def decoy_weights(batch: dict, distribution: str, native: torch.Tensor) -> torch.Tensor:
    """The decoy weights [B, N, 20] of positions whose native amino acids are native [B, N]."""
    aa = batch["aa_type"]
    valid = batch["seq_mask"].bool() & (aa < 20)
    counts = torch.stack([torch.bincount(a[v], minlength=20) for a, v in zip(aa, valid)]).double()
    pseudo = 1.0 if distribution.startswith("pseudocount") else 0.0
    freq = (counts + pseudo) / (counts.sum(-1, keepdim=True) + 20 * pseudo)
    weights = freq[:, None, :].expand(*native.shape, 20)
    if not distribution.endswith("_exclude_native"):
        return weights
    removed = weights * (native[..., None] != torch.arange(20))
    total = removed.sum(-1, keepdim=True)
    return torch.where(total > 0, removed / total, weights)


def contact_weights(batch: dict, distribution: str) -> tuple[torch.Tensor, torch.Tensor]:  # [B, E, 20] each
    return tuple(decoy_weights(batch, distribution, native) for native in contact_natives(batch))


def weighted_moments(energies: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted mean and population standard deviation over the last dimension: [..., 2]."""
    mean = (weights * energies).sum(-1) / weights.sum(-1)
    var = (weights * (energies - mean[..., None]) ** 2).sum(-1) / weights.sum(-1)
    return torch.stack([mean, var.sqrt()], dim=-1)


def pair_moments(table: torch.Tensor, w_src: torch.Tensor, w_dst: torch.Tensor) -> torch.Tensor:
    """Weighted moments of [..., 20, 20] pair energies, the pair (a, b) weighted w_src(a) w_dst(b)."""
    pair_weights = w_src[..., :, None] * w_dst[..., None, :]
    return weighted_moments(table.flatten(-2), pair_weights.flatten(-2))


def pair_table(model, batch: dict) -> torch.Tensor:  # [B, E, 20, 20]
    B, _, E = batch["edge_index"].shape
    return energy_model(model).compute_mutational_pair_energies(batch, B, E)


def write_truncated_pdb(source: Path, destination: Path, n_residues: int) -> Path:
    """The ATOM records of the first n_residues residues of source."""
    kept, residues = [], []
    for line in source.read_text().splitlines():
        if line.startswith("ATOM"):
            if line[22:27] not in residues:
                residues.append(line[22:27])
            if len(residues) > n_residues:
                break
            kept.append(line)
    destination.write_text("\n".join(kept + ["END"]) + "\n")
    return destination


def decoy_stats(mode: str = "mutational", seed: int = 0, use_c_rand: bool = False,
                **settings) -> tuple[torch.Tensor, torch.Tensor]:
    """The decoy stats of 1r69, [B, L, 2] or [B, E, 2], and the residues [B, L] or contacts [B, E] they hold."""
    model = model_for(mode, **settings)
    energy_model(model).use_c_rand = use_c_rand
    result, batch = run(model, seed=seed)
    valid = batch["seq_mask"].bool() if mode == "singleresidue" else valid_contacts(batch)
    return result[f"_{mode}_decoy_stats"], valid


@pytest.mark.parametrize(("settings", "weights"), [
    ({"tert_frust_decoy_distribution": "sequence"}, "sequence"),
    ({"tert_frust_decoy_distribution": "sequence_exclude_native"}, "sequence_exclude_native"),
    ({"tert_frust_decoy_distribution": "pseudocount"}, "pseudocount"),
    ({"tert_frust_decoy_distribution": "pseudocount_exclude_native"}, "pseudocount_exclude_native"),
    ({"tert_frust_exclude_native": True}, "sequence_exclude_native"),
])
def test_exact_statistics_are_weighted_moments_of_pair_table(settings: dict, weights: str) -> None:
    """Feature: exact statistics of a contact are the weighted mean and population std of its 400 pair energies,
    each endpoint weighted with the protein's decoy weights. The deprecated tert_frust_exclude_native flag reads
    "sequence" as "sequence_exclude_native"."""
    model = model_for(**settings)
    result, batch = run(model)
    expected = pair_moments(pair_table(model, batch), *contact_weights(batch, weights))
    valid = valid_contacts(batch)
    torch.testing.assert_close(result["_mutational_decoy_stats"][valid], expected[valid], rtol=1e-10, atol=1e-12)
    mean, std = expected[..., 0], expected[..., 1]
    frustration = (mean - result["_mutational_native_energy"]) / std
    torch.testing.assert_close(result["mutational"][valid], frustration[valid], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("electrostatics", [False, True])
def test_pair_table_holds_sampled_decoy_energies(electrostatics: bool) -> None:
    """Feature: entry (a, b) of a contact's pair table is the energy the sampled path evaluates for the decoy (a, b):
    burial, the contact itself and the background of both residues. The native pair is the native energy."""
    model = model_for(use_electrostatics=electrostatics)
    result, batch = run(model)
    table = pair_table(model, batch)
    B, _, E = batch["edge_index"].shape
    decoys = sample_mutational_decoys(batch["aa_type"], batch["seq_mask"], E, 50, decoy_distribution="pseudocount")
    sampled = energy_model(model).compute_mutational_decoy_energies(batch, B, E, decoys)
    b, e = torch.arange(B)[:, None], torch.arange(E)[None, :]
    valid = valid_contacts(batch)
    torch.testing.assert_close(table[b[..., None], e[..., None], decoys[0], decoys[1]][valid], sampled[valid],
                               rtol=1e-12, atol=1e-12)
    native_src, native_dst = contact_natives(batch)
    torch.testing.assert_close(table[b, e, native_src, native_dst][valid], result["_mutational_native_energy"][valid])


@pytest.mark.parametrize("distribution", ["sequence", "pseudocount_exclude_native"])
def test_sampled_statistics_converge_to_exact(distribution: str) -> None:
    """Feature: exact statistics are the limit of sampled statistics as the number of decoys grows."""
    n = 20000
    model = model_for(tert_frust_decoy_distribution=distribution)
    result, batch = run(model)
    B, _, E = batch["edge_index"].shape
    natives = contact_natives(batch) if distribution.endswith("_exclude_native") else None
    torch.manual_seed(1)
    decoys = sample_mutational_decoys(batch["aa_type"], batch["seq_mask"], E, n, native_edge_aa=natives,
                                      decoy_distribution=distribution)
    energies = energy_model(model).compute_mutational_decoy_energies(batch, B, E, decoys)
    valid = valid_contacts(batch)
    exact = result["_mutational_decoy_stats"][valid]
    sampled = torch.stack([energies.mean(dim=-1), energies.std(dim=-1, correction=0)], dim=-1)[valid]
    std = exact[:, 1].clamp(min=1e-6)
    mean_errors = (sampled[:, 0] - exact[:, 0]).abs() / (std / math.sqrt(n))
    std_errors = (sampled[:, 1] - exact[:, 1]).abs() / std
    assert (mean_errors < 6).double().mean() >= 0.99
    assert (std_errors < 0.05).double().mean() >= 0.99


def test_exact_statistics_are_deterministic_and_independent_of_batching(tmp_path: Path) -> None:
    """Feature: exact results do not depend on the seed or on the structures batched together, and draw no random
    numbers."""
    model = model_for()
    results = []
    for seed in (0, 1):
        batch = load(model, STRUCTURE)
        torch.manual_seed(seed)
        state = torch.get_rng_state()
        with torch.no_grad():
            results.append(model(batch))
        assert torch.equal(torch.get_rng_state(), state)
    short = write_truncated_pdb(STRUCTURE, tmp_path / "short.pdb", 40)
    together, batch = run(model, STRUCTURE, short)
    n_contacts = int(valid_contacts(batch)[0].sum())
    for result in (results[1], together):
        for key in ("mutational", "_mutational_decoy_stats"):
            torch.testing.assert_close(result[key][0, :n_contacts], results[0][key][0], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["singleresidue", "mutational"])
@pytest.mark.parametrize(("settings", "identical"), [
    ({"tert_frust_ndecoys": 50}, True),
    ({"use_c_rand": True, "tert_frust_ndecoys": 20}, False),  # the C stream is drawn in Python, one number at a time
], ids=["50", "use_c_rand"])
def test_statistics_are_exact_unless_use_c_rand(mode: str, settings: dict, identical: bool) -> None:
    """Feature: single-residue and mutational statistics are exact whatever tert_frust_ndecoys is. Only use_c_rand,
    which reproduces the C random stream, samples tert_frust_ndecoys decoys."""
    default, valid = decoy_stats(mode, seed=5)
    stats, _ = decoy_stats(mode, seed=3, **settings)
    if identical:
        assert torch.equal(stats, default)
    else:
        assert not torch.allclose(stats[valid], default[valid])


def test_c_rand_statistics_are_moments_of_c_stream_decoys() -> None:
    """Feature: with use_c_rand, mutational statistics are the mean and the population standard deviation (dividing by
    N, as LAMMPS) of the energies of the decoys drawn from the C random stream."""
    n = 20
    model = model_for(tert_frust_ndecoys=n)
    energy_model(model).use_c_rand = True
    result, batch = run(model)
    B, _, E = batch["edge_index"].shape
    decoys = sample_mutational_decoys(batch["aa_type"], batch["seq_mask"], E, n, use_c_rand=True)
    energies = energy_model(model).compute_mutational_decoy_energies(batch, B, E, decoys)
    mean = energies.mean(dim=-1)
    std = ((energies - mean[..., None]) ** 2).sum(dim=-1).div(n).sqrt()
    valid = valid_contacts(batch)
    torch.testing.assert_close(result["_mutational_decoy_stats"][valid], torch.stack([mean, std], dim=-1)[valid],
                               rtol=1e-12, atol=1e-12)


def test_mutational_results_are_the_same_alone_and_with_other_modes() -> None:
    """Feature: one pipeline computes mutational frustration, so it gives the same results alone and next to the
    single-residue and configurational modes, which draw random numbers before it."""
    settings = {"tert_frust_decoy_distribution": "pseudocount_exclude_native", "run_dmfs": True}
    first, *others = [run(model_for(mode, **settings))[0] for mode in ("mutational", "all")]
    for result in others:
        for key in ("mutational", "_mutational_native_energy", "_mutational_decoy_stats"):
            torch.testing.assert_close(result[key], first[key], rtol=0, atol=0)
        torch.testing.assert_close(result["_mutational_mutations"]["energies"],
                                   first["_mutational_mutations"]["energies"], rtol=0, atol=0)


def test_saved_mutation_table_is_the_exact_table() -> None:
    """Feature: run_dmfs saves the pair table the exact statistics come from, column a * 20 + b holding the
    pair (a, b), so the saved table reproduces the reported decoy stats."""
    model = model_for(run_dmfs=True)
    result, batch = run(model)
    saved = result["_mutational_mutations"]
    B, _, E = batch["edge_index"].shape
    pairs = torch.arange(400)
    assert torch.equal(saved["aas"], torch.stack([pairs // 20, pairs % 20]))
    table = saved["energies"].view(B, E, 20, 20)
    valid = valid_contacts(batch)
    torch.testing.assert_close(table[valid], pair_table(model, batch)[valid], rtol=0, atol=0)
    w_src, w_dst = contact_weights(batch, "sequence")
    stats = decoy_statistics(table, w_src[..., :, None] * w_dst[..., None, :], dim=(-2, -1))
    torch.testing.assert_close(stats[valid], result["_mutational_decoy_stats"][valid], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("distribution", ["sequence", "pseudocount_exclude_native"])
def test_mj_exact_statistics_are_weighted_moments_of_mj_matrix(distribution: str) -> None:
    """Feature: MJ exact statistics are the moments of the MJ matrix under the decoy distribution's endpoint
    weights."""
    model = model_for(energy_function="mj", tert_frust_decoy_distribution=distribution)
    result, batch = run(model)
    w_src, w_dst = contact_weights(batch, distribution)
    expected = pair_moments(model.mj_matrix.expand(*w_src.shape[:-1], 20, 20), w_src, w_dst)
    valid = valid_contacts(batch)
    torch.testing.assert_close(result["_mj_decoy_stats"][valid], expected[valid], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_single_residue_statistics_are_weighted_moments(distribution: str) -> None:
    """Feature: single-residue statistics are the weighted mean and population std of each residue's 20 decoy
    energies, weighted with the decoy distribution."""
    model = model_for("singleresidue", tert_frust_decoy_distribution=distribution)
    result, batch = run(model)
    _, decoy_energies, _ = energy_model(model).compute_singleresidue_stats_fast(
        batch, result["_singleresidue_native_energy"])
    weights = decoy_weights(batch, distribution, batch["aa_type"])
    expected = weighted_moments(decoy_energies.permute(1, 2, 0), weights)
    residues = batch["seq_mask"].bool()
    torch.testing.assert_close(result["_singleresidue_decoy_stats"][residues], expected[residues],
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("evaluation", ["exact", "2000 decoys"])
def test_chunked_decoy_evaluation_equals_single_pass(monkeypatch, evaluation: str) -> None:
    """Feature: decoys and the background water of every amino acid are evaluated in chunks of
    DECOY_CHUNK_ELEMENTS contacts x decoys, which bounds memory and gives the energies and stats of one pass."""
    model = model_for()
    _, batch = run(model)
    awsem = energy_model(model)
    B, _, E = batch["edge_index"].shape
    torch.manual_seed(4)
    decoys = sample_mutational_decoys(batch["aa_type"], batch["seq_mask"], E, 2000)

    def evaluate() -> tuple[torch.Tensor, ...]:
        if evaluation == "exact":
            return (awsem.compute_mutational_pair_energies(batch, B, E),
                    awsem.compute_mutational_decoy_ixns(batch, B, E)["stats"])
        return (awsem.compute_mutational_decoy_energies(batch, B, E, decoys),)

    single = evaluate()
    monkeypatch.setattr(mutational, "DECOY_CHUNK_ELEMENTS", 7 * B * E)  # 7 decoys or 7 * B * E / 20 rows per chunk
    chunked = evaluate()
    for chunked_values, single_values in zip(chunked, single):
        torch.testing.assert_close(chunked_values, single_values, rtol=1e-12, atol=1e-12)


def test_cli_writes_exact_mutational_frustration_whatever_the_decoy_count(tmp_path: Path) -> None:
    """Feature: the CLI writes a finite mutational frustration index for every contact, and --decoys, which sets the
    configurational and c-rand decoy count, does not change it."""
    written = []
    for options in ([], ["--decoys", "50"]):
        output_dir = tmp_path / f"decoys{len(options)}"
        args = ["run", "-s", str(STRUCTURE), "-m", "mutational", "-o", str(output_dir), "--accelerator", "cpu",
                "--num-workers", "0", "--no-pdb-write", *options]
        output = CliRunner().invoke(cli, args)
        assert output.exit_code == 0, output.output
        (path,) = output_dir.glob("**/*.pdb_mutational")
        written.append(pd.read_csv(path, sep=r"\s+"))
    _, batch = run(model_for())
    assert len(written[0]) == int(valid_contacts(batch).sum())
    assert np.isfinite(written[0]["FrstIndex"]).all()
    pd.testing.assert_frame_equal(written[1], written[0])

"""Contract: Frustratometer(mode) builds the model the CLI uses, for every AWSEM mode.

For the AWSEM energy function the factory returns Frustratometer3, configured for the requested mode, and
the model computes exactly that mode. For the MJ energy function it returns MJFrustratometer.

The per-mode classes (SingleResidueFrustratometer, MutationalFrustratometer, ConfigurationalFrustratometer)
build too, compute the same results as the factory's model, and back the module CLIs
(python -m pytorch_frustratometer.algorithm.<mode>).
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

import pytorch_frustratometer
from pytorch_frustratometer import Coefficients, Frustratometer, Frustratometer3, MJFrustratometer

DATA = Path(__file__).parent / "data"
MODES = ("singleresidue", "mutational", "configurational", "smooth")
PER_MODE_CLASSES = [
    ("SingleResidueFrustratometer", "singleresidue"),
    ("MutationalFrustratometer", "mutational"),
    ("ConfigurationalFrustratometer", "configurational"),
]


@pytest.mark.parametrize("mode", MODES)
def test_factory_builds_frustratometer3_for_every_awsem_mode(mode):
    """Feature: Frustratometer(mode) returns a Frustratometer3 whose coefficients have that mode."""
    model = Frustratometer(mode, dtype=torch.float64)

    assert isinstance(model, Frustratometer3)
    assert model.coefficients.mode == mode


@pytest.mark.parametrize("mode", ["singleresidue", "mutational", "configurational"])
def test_model_computes_exactly_the_requested_mode(mode):
    """Feature: the factory's model computes the requested mode and no other."""
    model = Frustratometer(mode, dtype=torch.float64)

    torch.manual_seed(0)
    result = model.from_pdb(DATA / "1r69.pdb", dtype=torch.float64)

    assert set(result) & set(MODES) == {mode}


def test_given_coefficients_keep_their_settings_and_take_the_requested_mode():
    """Feature: given coefficients are copied with the requested mode, and their other settings are kept."""
    coefficients = Coefficients(mode="all", burial_min_sep=13)

    model = Frustratometer("mutational", coefficients=coefficients)

    assert model.coefficients.mode == "mutational"
    assert model.coefficients.burial_min_sep == 13
    assert coefficients.mode == "all"


def test_mj_energy_function_builds_mj_frustratometer():
    """Feature: energy_function="mj" returns MJFrustratometer."""
    assert isinstance(Frustratometer("mutational", energy_function="mj"), MJFrustratometer)


@pytest.mark.parametrize("structure", ["1r69", "1r69_split"])
@pytest.mark.parametrize(("name", "mode"), PER_MODE_CLASSES)
def test_per_mode_class_builds_and_computes_its_mode(name, mode, structure):
    """Feature: each per-mode class builds, takes its mode, and computes exactly the results of Frustratometer(mode)
    under the same seed, also for a structure with two chains."""
    model = getattr(pytorch_frustratometer, name)(Coefficients(mode="all"), dtype=torch.float64)
    assert model.coefficients.mode == mode

    results = []
    for each in (model, Frustratometer(mode, dtype=torch.float64)):
        torch.manual_seed(0)
        results.append(each.from_pdb(DATA / f"{structure}.pdb", dtype=torch.float64))
    ours, factory = results

    assert set(ours) & set(MODES) == {mode}
    for key in (mode, f"_{mode}_native_energy", f"_{mode}_decoy_stats"):
        assert torch.equal(ours[key], factory[key]), key


@pytest.mark.parametrize(("mode", "options", "mutations"), [
    ("singleresidue", ["--run_dmfs"], True),
    ("mutational", ["--run_dmfs"], True),
    ("configurational", [], False),  # its CLI has no --run_dmfs
])
def test_module_cli_writes_its_mode(tmp_path, mode, options, mutations):
    """Feature: each module CLI builds its per-mode class and writes the mode's frustration file and, with
    --run_dmfs, its mutation table."""
    cli = importlib.import_module(f"pytorch_frustratometer.algorithm.{mode}").cli
    args = ["-s", str(DATA / "1r69.pdb"), "-o", str(tmp_path), "--num_workers", "0", *options]
    output = CliRunner().invoke(cli, args)
    assert output.exit_code == 0, output.output

    written = tmp_path / "1r69.done" / "FrustrationData"
    assert (written / f"1r69.pdb_{mode}").is_file()
    assert (written / f"1r69.pdb_{mode}_mutations").is_file() == mutations

"""Tests for workflow_generator.py — Pegasus DAG generation."""

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
import yaml

pytest.importorskip("Pegasus.api", reason="Pegasus API required for workflow generator tests")

WF_DIR = os.path.join(os.path.dirname(__file__), "..")
SCRIPT = os.path.join(WF_DIR, "workflow_generator.py")


@pytest.fixture
def stage1_inputs(tmp_path):
    """Create two 500x500 source images for Stage 1 testing."""
    paths = []
    for i in range(2):
        img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        p = str(tmp_path / f"s2_vis_{i:02d}.png")
        cv2.imwrite(p, img)
        paths.append(p)
    return paths


@pytest.fixture
def stage2_inputs(tmp_path):
    """Create training images/masks directories for Stage 2 testing."""
    img_dir = tmp_path / "train_images"
    mask_dir = tmp_path / "train_masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    rng = np.random.RandomState(0)
    for i in range(4):
        cv2.imwrite(str(img_dir / f"img_{i:02d}.png"),
                     rng.randint(0, 255, (256, 256), dtype=np.uint8))
        cv2.imwrite(str(mask_dir / f"mask_{i:02d}.png"),
                     rng.choice([0, 128, 255], size=(256, 256)).astype(np.uint8))
    return str(img_dir), str(mask_dir)


def test_workflow_generator_help():
    """--help should exit 0 and show usage."""
    result = subprocess.run(
        [sys.executable, SCRIPT, "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Generate Pegasus workflow" in result.stdout


def test_stage1_only(stage1_inputs, tmp_path):
    """Stage 1 only (--no-auto-label): split/segment/merge jobs, no Stage 2."""
    output = str(tmp_path / "workflow.yml")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--images"] + stage1_inputs + [
         "--tile-size", "250",
         "--original-size", "500",
         "--scene-size", "0",
         "--no-auto-label",
         "--output", output,
         "--skip-sites-catalog"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert os.path.exists(output)

    with open(output) as f:
        content = f.read()
    # Verify key job types are present
    assert "image_split" in content
    assert "color_segment" in content
    assert "image_merge" in content
    # Stage 2 jobs should NOT be present
    assert "preprocess_data" not in content
    assert "train_unet" not in content


def test_stage1_job_counts(stage1_inputs, tmp_path):
    """2 images at 500x500 with tile_size=250: 2 split + 8 segment + 2 merge = 12 jobs."""
    output = str(tmp_path / "workflow.yml")
    subprocess.run(
        [sys.executable, SCRIPT,
         "--images"] + stage1_inputs + [
         "--tile-size", "250", "--original-size", "500",
         "--scene-size", "0", "--no-auto-label",
         "--output", output, "--skip-sites-catalog"],
        capture_output=True, text=True, check=True,
    )

    with open(output) as f:
        wf_data = yaml.safe_load(f)

    job_ids = [j["id"] for j in wf_data.get("jobs", [])]
    split_jobs = [j for j in job_ids if j.startswith("split_")]
    seg_jobs = [j for j in job_ids if j.startswith("seg_")]
    merge_jobs = [j for j in job_ids if j.startswith("merge_")]

    assert len(split_jobs) == 2, f"Expected 2 split jobs, got {len(split_jobs)}"
    assert len(seg_jobs) == 8, f"Expected 8 segment jobs, got {len(seg_jobs)}"
    assert len(merge_jobs) == 2, f"Expected 2 merge jobs, got {len(merge_jobs)}"


def test_both_stages(stage1_inputs, stage2_inputs, tmp_path):
    """Both stages: should include all 6 job types."""
    img_dir, mask_dir = stage2_inputs
    output = str(tmp_path / "workflow.yml")

    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--images"] + stage1_inputs + [
         "--tile-size", "250", "--original-size", "500",
         "--scene-size", "0", "--no-auto-label",
         "--train-images-dir", img_dir,
         "--train-masks-dir", mask_dir,
         "--epochs", "1",
         "--output", output,
         "--skip-sites-catalog"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    with open(output) as f:
        content = f.read()

    for job_type in ["image_split", "color_segment", "image_merge",
                     "preprocess_data", "train_unet", "evaluate_model"]:
        assert job_type in content, f"Missing job type: {job_type}"


def test_fails_on_missing_image(tmp_path):
    """Should fail if input image does not exist."""
    output = str(tmp_path / "workflow.yml")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--images", "/nonexistent/image.png",
         "--output", output,
         "--skip-sites-catalog"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_unique_job_ids(stage1_inputs, tmp_path):
    """All job IDs in the workflow should be unique."""
    output = str(tmp_path / "workflow.yml")
    subprocess.run(
        [sys.executable, SCRIPT,
         "--images"] + stage1_inputs + [
         "--tile-size", "250", "--original-size", "500",
         "--output", output, "--skip-sites-catalog"],
        capture_output=True, text=True, check=True,
    )

    with open(output) as f:
        wf_data = yaml.safe_load(f)

    job_ids = [j["id"] for j in wf_data.get("jobs", [])]
    assert len(job_ids) == len(set(job_ids)), \
        f"Duplicate job IDs found: {[x for x in job_ids if job_ids.count(x) > 1]}"


def test_auto_label_mode(stage1_inputs, tmp_path):
    """--auto-label should add split_images + split_masks jobs and wire them to preprocess."""
    output = str(tmp_path / "workflow.yml")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--images"] + stage1_inputs + [
         "--tile-size", "250", "--original-size", "500",
         "--scene-size", "0",
         "--auto-label",
         "--epochs", "1",
         "--output", output,
         "--skip-sites-catalog"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    with open(output) as f:
        wf_data = yaml.safe_load(f)

    job_ids = [j["id"] for j in wf_data.get("jobs", [])]

    # Should have split_images and split_masks jobs (one per source image)
    split_img_jobs = [j for j in job_ids if j.startswith("split_images_")]
    split_mask_jobs = [j for j in job_ids if j.startswith("split_masks_")]
    assert len(split_img_jobs) == 2, f"Expected 2 split_images jobs, got {len(split_img_jobs)}"
    assert len(split_mask_jobs) == 2, f"Expected 2 split_masks jobs, got {len(split_mask_jobs)}"

    # Should have preprocess, train, evaluate for both branches
    # (--paths both is the default; stage-2 ids carry the branch suffix)
    for branch in ("orig", "filtered"):
        assert f"preprocess_{branch}" in job_ids
        assert f"train_{branch}" in job_ids
        assert f"evaluate_{branch}" in job_ids

    # All job IDs should be unique
    assert len(job_ids) == len(set(job_ids)), \
        f"Duplicate job IDs: {[x for x in job_ids if job_ids.count(x) > 1]}"

    # Check that split_images and split_masks jobs use --grayscale flag
    for job in wf_data["jobs"]:
        if job["id"].startswith(("split_masks_", "split_images_")):
            args_str = " ".join(str(a) for a in job["arguments"])
            assert "--grayscale" in args_str, \
                f"Job {job['id']} missing --grayscale flag"
            assert "--tile-size 256" in args_str or "--tile-size\n256" in args_str, \
                f"Job {job['id']} missing --tile-size 256"

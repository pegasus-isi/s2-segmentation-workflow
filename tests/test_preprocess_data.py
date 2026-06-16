"""Tests for bin/preprocess_data.py."""

import glob
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="TensorFlow required for preprocess_data tests")

BIN_DIR = os.path.join(os.path.dirname(__file__), "..", "bin")
SCRIPT = os.path.join(BIN_DIR, "preprocess_data.py")


def _run_preprocess(img_dir, mask_dir, tmp_dir, n_classes=3, test_size=0.20):
    """Helper: run preprocess_data.py and return output paths."""
    x_train = os.path.join(tmp_dir, "X_train.npy")
    x_test = os.path.join(tmp_dir, "X_test.npy")
    y_train = os.path.join(tmp_dir, "y_train_cat.npy")
    y_test = os.path.join(tmp_dir, "y_test_cat.npy")

    cmd = [sys.executable, SCRIPT]
    for f in sorted(glob.glob(os.path.join(img_dir, "*.png"))):
        cmd.extend(["--image", f])
    for f in sorted(glob.glob(os.path.join(mask_dir, "*.png"))):
        cmd.extend(["--mask", f])
    cmd.extend([
        "--x-train", x_train,
        "--x-test", x_test,
        "--y-train", y_train,
        "--y-test", y_test,
        "--n-classes", str(n_classes),
        "--test-size", str(test_size),
        "--random-state", "42",
    ])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result, x_train, x_test, y_train, y_test


def test_preprocess_creates_all_outputs(synthetic_training_data, tmp_dir):
    """Should create all 4 .npy output files."""
    img_dir, mask_dir = synthetic_training_data
    result, x_train, x_test, y_train, y_test = _run_preprocess(img_dir, mask_dir, tmp_dir)
    assert result.returncode == 0, f"stderr: {result.stderr}"
    for path in [x_train, x_test, y_train, y_test]:
        assert os.path.exists(path), f"Missing output: {path}"


def test_preprocess_train_test_split_ratio(synthetic_training_data, tmp_dir):
    """With 4 samples and test_size=0.25, should get 3 train + 1 test."""
    img_dir, mask_dir = synthetic_training_data
    result, x_train, x_test, y_train, y_test = _run_preprocess(
        img_dir, mask_dir, tmp_dir, test_size=0.25,
    )
    assert result.returncode == 0

    X_train = np.load(x_train)
    X_test = np.load(x_test)
    assert X_train.shape[0] == 3, f"Expected 3 train samples, got {X_train.shape[0]}"
    assert X_test.shape[0] == 1, f"Expected 1 test sample, got {X_test.shape[0]}"


def test_preprocess_output_shapes(synthetic_training_data, tmp_dir):
    """X arrays should be (N, 256, 256, 1), y arrays should be (N, 256, 256, 3)."""
    img_dir, mask_dir = synthetic_training_data
    result, x_train, x_test, y_train, y_test = _run_preprocess(img_dir, mask_dir, tmp_dir)
    assert result.returncode == 0

    X_tr = np.load(x_train)
    y_tr = np.load(y_train)

    assert len(X_tr.shape) == 4, f"X_train should be 4D, got {X_tr.shape}"
    assert X_tr.shape[1:3] == (256, 256), f"Spatial dims should be 256x256, got {X_tr.shape}"
    assert X_tr.shape[3] == 1, f"X should have 1 channel, got {X_tr.shape[3]}"
    assert y_tr.shape[-1] == 3, f"y should have 3 classes, got {y_tr.shape[-1]}"


def test_preprocess_x_values_normalized(synthetic_training_data, tmp_dir):
    """X values should be normalized (not raw 0-255 range)."""
    img_dir, mask_dir = synthetic_training_data
    result, x_train, x_test, y_train, y_test = _run_preprocess(img_dir, mask_dir, tmp_dir)
    X_tr = np.load(x_train)
    # After keras normalize, values should not all be in [0, 255] integer range
    assert X_tr.max() <= 255.0  # sanity check
    # At least some values should be non-integer (normalized)
    non_int = np.sum(X_tr != X_tr.astype(int))
    assert non_int > 0, "Expected normalized (non-integer) values"


def test_preprocess_y_is_one_hot(synthetic_training_data, tmp_dir):
    """y arrays should be one-hot encoded (values 0 or 1 only)."""
    img_dir, mask_dir = synthetic_training_data
    result, x_train, x_test, y_train, y_test = _run_preprocess(img_dir, mask_dir, tmp_dir)
    y_tr = np.load(y_train)
    unique_vals = np.unique(y_tr)
    assert set(unique_vals).issubset({0.0, 1.0}), \
        f"One-hot should only contain 0 and 1, got {unique_vals}"


def test_preprocess_zero_cloud_fraction_preserved(synthetic_training_data, tmp_dir):
    """A cloud fraction of exactly 0.0 must be kept, not dropped to -1.

    Regression for the ``frac or -1.0`` truthiness bug: a perfectly clear
    tile (fraction 0.0) is falsy and was wrongly marked missing, excluding
    the cleanest tiles from stratified evaluation.
    """
    import json
    img_dir, mask_dir = synthetic_training_data
    # All tiles get fraction 0.0 so whichever lands in the test split is 0.0.
    cf = {f"img_{i:02d}": 0.0 for i in range(4)}
    cf_path = os.path.join(tmp_dir, "cloud_fraction.json")
    with open(cf_path, "w") as f:
        json.dump(cf, f)
    out_npy = os.path.join(tmp_dir, "test_cloud_fractions.npy")

    cmd = [sys.executable, SCRIPT]
    for f in sorted(glob.glob(os.path.join(img_dir, "*.png"))):
        cmd.extend(["--image", f])
    for f in sorted(glob.glob(os.path.join(mask_dir, "*.png"))):
        cmd.extend(["--mask", f])
    cmd.extend([
        "--x-train", os.path.join(tmp_dir, "X_train.npy"),
        "--x-test", os.path.join(tmp_dir, "X_test.npy"),
        "--y-train", os.path.join(tmp_dir, "y_train.npy"),
        "--y-test", os.path.join(tmp_dir, "y_test.npy"),
        "--n-classes", "3", "--test-size", "0.25", "--random-state", "0",
        "--cloud-fraction", cf_path,
        "--test-cloud-fractions", out_npy,
    ])
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"stderr: {result.stderr}"

    fracs = np.load(out_npy)
    assert (fracs >= 0).all(), \
        f"0.0 fractions wrongly dropped to -1: {fracs}"
    assert (fracs == 0.0).all(), f"expected all 0.0, got {fracs}"

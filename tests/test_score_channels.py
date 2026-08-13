"""
`score_channels` must mean the same thing in all three places a detector runs:
the torch detector, the distilled NumPy evaluator, and the generated C.
"""

import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("torch")

from telemetry_anomdet.models.deep.codegen import generate_c, write_c  # noqa: E402
from telemetry_anomdet.models.deep.distill import (  # noqa: E402
    KANGDNNumpy,
    extract_kan_gdn,
)
from telemetry_anomdet.models.deep.gdn import GDN  # noqa: E402
from telemetry_anomdet.models.deep.kan_gdn import KANGDN  # noqa: E402

CC = shutil.which("cc") or shutil.which("gcc")
WINDOW_SIZE, N_FEATURES = 6, 5


def _cc_env() -> dict:
    env = dict(os.environ)
    if CC is not None:
        env["PATH"] = str(pathlib.Path(CC).parent) + os.pathsep + env.get("PATH", "")
    return env


def _fit(score_channels=None):
    X = np.random.default_rng(0).normal(size=(40, WINDOW_SIZE, N_FEATURES))
    det = KANGDN(
        embed_dim=8,
        topk=3,
        epochs=3,
        batch_size=16,
        random_state=0,
        score_channels=score_channels,
    )
    det.fit(X)
    return det, X


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------


def test_restricting_channels_changes_the_score():
    plain, X = _fit()
    restricted, _ = _fit(score_channels=[0])
    assert not np.allclose(plain.decision_function(X), restricted.decision_function(X))


def test_all_channels_matches_the_default():
    """Naming every channel must reproduce the unrestricted behaviour."""
    plain, X = _fit()
    explicit, _ = _fit(score_channels=list(range(N_FEATURES)))
    np.testing.assert_allclose(plain.decision_function(X), explicit.decision_function(X))


def test_score_is_the_max_over_named_channels_only():
    det, X = _fit(score_channels=[1, 3])
    errors = det._forecast_errors(det._scale_transform(X))
    normed = np.abs(errors - det._err_median_) / (det._err_iqr_ + 1e-9)
    np.testing.assert_allclose(det.decision_function(X), normed[:, [1, 3]].max(axis=1))


@pytest.mark.parametrize("bad", [[], [0, 0], [-1]])
def test_rejects_malformed_channel_lists(bad):
    with pytest.raises(ValueError, match="score_channels"):
        GDN(score_channels=bad)


def test_rejects_channels_beyond_the_fitted_width():
    X = np.random.default_rng(0).normal(size=(30, WINDOW_SIZE, N_FEATURES))
    with pytest.raises(ValueError, match="exceed"):
        KANGDN(epochs=1, score_channels=[0, N_FEATURES]).fit(X)


def test_recorded_in_params():
    det, _ = _fit(score_channels=[0, 2])
    assert det._get_params()["score_channels"] == [0, 2]


# ---------------------------------------------------------------------------
# The distilled evaluator and the generated C
# ---------------------------------------------------------------------------


def test_distilled_evaluator_honours_score_channels():
    det, X = _fit(score_channels=[0])
    spec = extract_kan_gdn(det)
    assert spec["score_channels"] == [0]
    np.testing.assert_allclose(
        KANGDNNumpy(spec).decision_function(X),
        det.decision_function(X),
        rtol=1e-4,
        atol=1e-5,
    )


def test_generated_c_declares_only_the_named_channels():
    det, _ = _fit(score_channels=[0, 2])
    src = generate_c(extract_kan_gdn(det))["kangdn.c"]
    assert "N_SCORE = 2" in src
    assert "static const unsigned char score_channels[2]" in src


def test_generated_c_defaults_to_every_channel():
    det, _ = _fit()
    src = generate_c(extract_kan_gdn(det))["kangdn.c"]
    assert f"N_SCORE = {N_FEATURES}" in src


@pytest.mark.skipif(CC is None, reason="no C compiler available")
def test_compiled_c_matches_the_restricted_detector(tmp_path):
    """The whole chain must agree: torch, NumPy, and the compiled artifact."""
    det, X = _fit(score_channels=[0])
    spec = extract_kan_gdn(det)
    windows = X[:6]
    write_c(spec, tmp_path, dtype="double", test_windows=windows)

    main = r"""
#include <stdio.h>
#include "kangdn.h"
#include "kangdn_vectors.h"
int main(void) {
    int i; kangdn_real_t s;
    for (i = 0; i < KANGDN_N_VECTORS; ++i) {
        if (kangdn_score(KANGDN_TEST_WINDOWS[i], &s) != KANGDN_OK) return 1;
        printf("%.17g\n", (double)s);
    }
    return 0;
}
"""
    (tmp_path / "main.c").write_text(main, encoding="utf-8")
    exe = tmp_path / "sc.exe"
    subprocess.run(
        [
            CC,
            "-std=c99",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(tmp_path),
            str(tmp_path / "kangdn.c"),
            str(tmp_path / "main.c"),
            "-o",
            str(exe),
            "-lm",
        ],
        check=True,
        capture_output=True,
        env=_cc_env(),
    )
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True, env=_cc_env())
    got = np.array([float(v) for v in out.stdout.split()])
    np.testing.assert_allclose(got, det.decision_function(windows), rtol=1e-4, atol=1e-5)

import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

# Code generation reads a fitted torch detector; skip if the deep extra is absent.
pytest.importorskip("torch")

from telemetry_anomdet.models.deep.codegen import generate_c, write_c  # noqa: E402
from telemetry_anomdet.models.deep.distill import (  # noqa: E402
    KANGDNNumpy,
    extract_kan_gdn,
)
from telemetry_anomdet.models.deep.kan_gdn import KANGDN  # noqa: E402

CC = shutil.which("cc") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(CC is None, reason="no C compiler available")


def _cc_env() -> dict:
    """
    Environment for invoking the compiler.

    A toolchain such as MSYS2 installs the driver alongside the DLLs its
    backend needs. Launched from a shell those are already on PATH, but a bare
    subprocess inherits no such guarantee and the compile fails with no
    diagnostics at all. Prepending the driver's own directory fixes it.
    """
    env = dict(os.environ)
    if CC is not None:
        env["PATH"] = str(pathlib.Path(CC).parent) + os.pathsep + env.get("PATH", "")
    return env


WINDOW_SIZE = 6
N_FEATURES = 5


def _fitted(
    n_windows=40,
    window_size=WINDOW_SIZE,
    n_features=N_FEATURES,
    embed_dim=8,
    topk=3,
    epochs=3,
    seed=0,
    **kwargs,
):
    """A fitted detector plus the windows it was trained on."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_windows, window_size, n_features))
    det = KANGDN(
        embed_dim=embed_dim,
        topk=topk,
        epochs=epochs,
        batch_size=16,
        random_state=0,
        **kwargs,
    )
    det.fit(X)
    return det, X


def _harness(windows: np.ndarray, prefix: str = "kangdn") -> str:
    """A main() that scores each window and prints one score per line."""
    n, w, f = windows.shape
    rows = []
    for win in windows:
        rows.append(
            "  {"
            + ", ".join("{" + ", ".join(f"{v:.17g}" for v in step) + "}" for step in win)
            + "}"
        )
    data = ",\n".join(rows)
    return f"""
#include <stdio.h>
#include "{prefix}.h"

static const {prefix}_real_t windows[{n}][{w}][{f}] = {{
{data}
}};

int main(void) {{
    int i;
    int status;
    {prefix}_real_t score;

    for (i = 0; i < {n}; ++i) {{
        score = ({prefix}_real_t)0;
        status = {prefix}_score(&windows[i][0][0], &score);
        if (status != {prefix.upper()}_OK) {{
            printf("ERR %d\\n", status);
            return 1;
        }}
        printf("%.17g\\n", (double)score);
    }}
    return 0;
}}
"""


# Power of Ten rule 10: the generated code must survive strict warnings.
CFLAGS = ["-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic"]


def _compile(tmp_path, extracted, windows, dtype):
    """Generate and compile the C detector; return the executable path."""
    write_c(extracted, tmp_path, dtype=dtype)
    (tmp_path / "main.c").write_text(_harness(windows), encoding="utf-8")

    exe = tmp_path / "detector.exe"
    subprocess.run(
        [
            CC,
            *CFLAGS,
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
    return exe


def _compile_and_run(tmp_path, extracted, windows, dtype):
    """Generate, compile and execute the C detector; return its scores."""
    exe = _compile(tmp_path, extracted, windows, dtype)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True, env=_cc_env())
    assert "ERR" not in out.stdout, f"detector returned an error status: {out.stdout}"
    return np.array([float(line) for line in out.stdout.split()])


# ---------------------------------------------------------------------------
# Generation itself (no compiler needed)
# ---------------------------------------------------------------------------


def test_generates_header_and_source():
    det, _ = _fitted()
    files = generate_c(extract_kan_gdn(det))
    assert set(files) == {"kangdn.h", "kangdn.c"}
    assert "kangdn_score" in files["kangdn.h"]
    assert "#include <math.h>" in files["kangdn.c"]
    # Parameters are static const so they land in flash, not RAM.
    assert "static const float act_spline_w" in files["kangdn.c"]


# ---------------------------------------------------------------------------
# Power of Ten conformance of the generated source
# ---------------------------------------------------------------------------


def _source() -> str:
    det, _ = _fitted()
    return generate_c(extract_kan_gdn(det))["kangdn.c"]


def test_no_recursion_or_complex_flow():
    """Rule 1: no goto, setjmp, or longjmp."""
    src = _source()
    for banned in ("goto", "setjmp", "longjmp"):
        assert banned not in src


def test_no_heap_allocation():
    """Rule 3: nothing is allocated at runtime."""
    src = _source()
    for banned in ("malloc", "calloc", "realloc", "free("):
        assert banned not in src


def test_functions_stay_short():
    """Rule 4: no function body runs past roughly one printed page."""
    src = _source().splitlines()
    longest, current, name = 0, None, ""
    for line in src:
        if line.startswith(("static int ", "static float ", "static double ", "int kangdn_")):
            current, name = 0, line
        elif current is not None:
            current += 1
            if line == "}":
                longest = max(longest, current)
                current = None
    assert longest <= 60, f"a generated function exceeds 60 lines (longest {longest}, {name})"


def _function_bodies(src: str) -> list[str]:
    """Every generated function body, split on the column-zero closing brace."""
    return [chunk for chunk in src.split("\n}\n") if "\n{\n" in chunk]


def test_every_function_has_assertions():
    """Rule 5: at least two assertions per generated function."""
    bodies = _function_bodies(_source())
    assert len(bodies) >= 10, f"expected the full scoring path, found {len(bodies)}"
    for body in bodies:
        signature = body[body.rfind("\n\n") :].strip().splitlines()[0]
        assert body.count("KANGDN_ASSERT") >= 2, f"too few assertions in: {signature}"


def test_no_file_scope_mutable_state():
    """Rule 6: every non-const file-scope object is inside a function."""
    for line in _source().splitlines():
        if line.startswith("static ") and "(" not in line:
            assert line.startswith("static const "), f"mutable file-scope object: {line}"


def test_every_call_site_checks_status():
    """Rule 7: each internal call is followed by a status check."""
    lines = _source().splitlines()
    for i, line in enumerate(lines):
        if "status = " in line and "int status" not in line:
            following = " ".join(lines[i + 1 : i + 3])
            assert "if (status !=" in following, f"unchecked call: {line.strip()}"


def test_preprocessor_used_sparingly():
    """Rule 8: no function-like macros beyond the assertion hook."""
    src = _source()
    defines = [ln for ln in src.splitlines() if ln.startswith("#define")]
    for line in defines:
        assert "ASSERT" in line, f"unexpected macro definition: {line}"
    # Knot vectors are data, not macros.
    assert "static const float act_knots" in src


def test_no_function_pointers():
    """Rule 9: no function pointer declarations."""
    assert "(*" not in _source()


def test_rejects_unknown_dtype():
    det, _ = _fitted()
    with pytest.raises(ValueError, match="dtype"):
        generate_c(extract_kan_gdn(det), dtype="half")


def test_rejects_per_feature_knots():
    """One knot vector is emitted per layer, so features must agree on it."""
    det, _ = _fitted()
    spec = extract_kan_gdn(det)
    spec["net"]["activation"]["grid"][2, :] += 0.5  # give one feature its own grid
    with pytest.raises(NotImplementedError, match="shared across input features"):
        generate_c(spec)


def test_handles_non_uniform_knots():
    """Spans are tabulated per index, so unevenly spaced knots are supported."""
    det, X = _fitted()
    spec = extract_kan_gdn(det)
    spec["net"]["activation"]["grid"][:, 4] += 0.1  # perturb one interior knot
    files = generate_c(spec)
    assert "act_inv_span" in files["kangdn.c"]
    assert np.all(np.isfinite(KANGDNNumpy(spec).decision_function(X[:4])))


def test_write_c_creates_both_files(tmp_path):
    det, _ = _fitted()
    written = write_c(extract_kan_gdn(det), tmp_path / "out")
    assert sorted(p.name for p in written) == ["kangdn.c", "kangdn.h"]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


# ---------------------------------------------------------------------------
# Compiled C agrees with the NumPy evaluator
# ---------------------------------------------------------------------------


@needs_cc
def test_c_matches_numpy_in_double(tmp_path):
    """In double precision the port should be exact to near machine epsilon."""
    det, X = _fitted()
    spec = extract_kan_gdn(det)
    windows = X[:8]

    c_scores = _compile_and_run(tmp_path, spec, windows, "double")
    np_scores = KANGDNNumpy(spec).decision_function(windows)

    np.testing.assert_allclose(c_scores, np_scores, rtol=1e-10, atol=1e-12)


@needs_cc
def test_c_matches_numpy_in_float(tmp_path):
    """Single precision is the deployment mode; it should track NumPy closely."""
    det, X = _fitted()
    spec = extract_kan_gdn(det)
    windows = X[:8]

    c_scores = _compile_and_run(tmp_path, spec, windows, "float")
    np_scores = KANGDNNumpy(spec).decision_function(windows)

    np.testing.assert_allclose(c_scores, np_scores, rtol=1e-4, atol=1e-5)


@needs_cc
def test_c_labels_match_the_detector(tmp_path):
    """The compiled detector flags the same unseen windows as the torch model."""
    det, _ = _fitted()
    spec = extract_kan_gdn(det)
    windows = np.random.default_rng(7).normal(size=(12, WINDOW_SIZE, N_FEATURES))

    c_scores = _compile_and_run(tmp_path, spec, windows, "double")
    c_labels = (c_scores > spec["threshold"]).astype(int)

    np.testing.assert_array_equal(c_labels, det.predict(windows))


# ---------------------------------------------------------------------------
# Deployment-scale and branch coverage
# ---------------------------------------------------------------------------


@needs_cc
def test_c_matches_numpy_at_deployment_scale(tmp_path):
    """
    The small fixtures prove the port; this proves it at the size that ships.

    Index arithmetic that happens to work when the dimensions are similar can
    fail once they diverge, and float32 error accumulates with width and node
    count, so the drift measured on a toy model does not transfer.
    """
    det, X = _fitted(n_windows=120, window_size=30, n_features=25, embed_dim=32, topk=10, epochs=2)
    spec = extract_kan_gdn(det)
    windows = X[:8]
    ref = KANGDNNumpy(spec).decision_function(windows)

    exact = _compile_and_run(tmp_path / "d", spec, windows, "double")
    np.testing.assert_allclose(exact, ref, rtol=1e-10, atol=1e-12)

    # Single-precision drift stays at about one float32 epsilon (1.2e-07) here,
    # so it does not grow with node count or width as might be expected.
    single = _compile_and_run(tmp_path / "f", spec, windows, "float")
    np.testing.assert_allclose(single, ref, rtol=1e-5, atol=1e-6)


@needs_cc
def test_c_matches_numpy_without_scaling(tmp_path):
    """The unscaled branch emits different index expressions and needs covering."""
    det, X = _fitted(scale=False)
    spec = extract_kan_gdn(det)
    assert spec["scaler_mean"] is None
    assert "scaler_mean" not in generate_c(spec)["kangdn.c"]

    got = _compile_and_run(tmp_path, spec, X[:8], "double")
    np.testing.assert_allclose(
        got, KANGDNNumpy(spec).decision_function(X[:8]), rtol=1e-10, atol=1e-12
    )


@needs_cc
def test_c_matches_numpy_outside_the_spline_grid(tmp_path):
    """
    Inputs far outside the knot range drive every B-spline basis to zero, so the
    SiLU residual carries the output. That is the path an actual anomaly takes.
    """
    det, _ = _fitted()
    spec = extract_kan_gdn(det)
    windows = np.random.default_rng(3).normal(size=(6, WINDOW_SIZE, N_FEATURES)) * 30.0

    got = _compile_and_run(tmp_path, spec, windows, "double")
    ref = KANGDNNumpy(spec).decision_function(windows)
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-11)


# ---------------------------------------------------------------------------
# Status codes
# ---------------------------------------------------------------------------


_NULL_MAIN = r"""
#include <stdio.h>
#include <stddef.h>
#include "kangdn.h"

int main(void) {
    kangdn_real_t score = (kangdn_real_t)0;
    int flag = 0;
    static kangdn_real_t win[KANGDN_WINDOW_SIZE][KANGDN_N_FEATURES];

    printf("%d\n", kangdn_score(NULL, &score));
    printf("%d\n", kangdn_score(&win[0][0], NULL));
    printf("%d\n", kangdn_is_anomaly(NULL, &flag));
    printf("%d\n", kangdn_is_anomaly(&win[0][0], NULL));
    printf("%d\n", kangdn_score(&win[0][0], &score));
    return 0;
}
"""


@needs_cc
def test_null_arguments_return_err_null(tmp_path):
    """Rule 7 in practice: a bad pointer is reported, not dereferenced."""
    det, _ = _fitted()
    write_c(extract_kan_gdn(det), tmp_path, dtype="double")
    (tmp_path / "main.c").write_text(_NULL_MAIN, encoding="utf-8")

    exe = tmp_path / "null.exe"
    subprocess.run(
        [
            CC,
            *CFLAGS,
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
    codes = [int(line) for line in out.stdout.split()]
    # Four null-argument calls report ERR_NULL (1); the valid call succeeds (0).
    assert codes == [1, 1, 1, 1, 0]


_RANGE_MAIN = r"""
#include <stdio.h>
#include "kangdn.h"

static kangdn_real_t win[KANGDN_WINDOW_SIZE][KANGDN_N_FEATURES];

int main(void) {
    kangdn_real_t score = (kangdn_real_t)0;
    int t;
    int f;
    for (t = 0; t < KANGDN_WINDOW_SIZE; ++t) {
        for (f = 0; f < KANGDN_N_FEATURES; ++f) {
            win[t][f] = (kangdn_real_t)1e30;
        }
    }
    printf("%d\n", kangdn_score(&win[0][0], &score));
    return 0;
}
"""


@needs_cc
def test_implausible_score_returns_err_range(tmp_path):
    """
    A corrupted or wildly out-of-range input must not become a detection.

    Assertions are compiled out here so the range check is what reports the
    problem, which is also how an integrator overrides the hook.
    """
    det, _ = _fitted()
    write_c(extract_kan_gdn(det), tmp_path, dtype="double")
    (tmp_path / "main.c").write_text(_RANGE_MAIN, encoding="utf-8")

    exe = tmp_path / "range.exe"
    subprocess.run(
        [
            CC,
            *CFLAGS,
            "-DKANGDN_ASSERT(cond)=",
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
    assert int(out.stdout.strip()) == 2  # KANGDN_ERR_RANGE


# ---------------------------------------------------------------------------
# Degenerate graphs and generator limits
# ---------------------------------------------------------------------------


@needs_cc
def test_c_matches_numpy_with_topk_clamped(tmp_path):
    """topk larger than the graph is clamped, changing the neighbour count."""
    det, X = _fitted(n_features=3, topk=99)
    spec = extract_kan_gdn(det)
    n_nbr = int(np.asarray(spec["net"]["adj"]).sum(axis=1)[0])
    assert n_nbr == 3  # every node, including itself

    got = _compile_and_run(tmp_path, spec, X[:6], "double")
    np.testing.assert_allclose(
        got, KANGDNNumpy(spec).decision_function(X[:6]), rtol=1e-10, atol=1e-12
    )


@needs_cc
def test_c_matches_numpy_on_a_two_node_graph(tmp_path):
    det, X = _fitted(n_features=2, topk=1, embed_dim=4)
    spec = extract_kan_gdn(det)
    got = _compile_and_run(tmp_path, spec, X[:6], "double")
    np.testing.assert_allclose(
        got, KANGDNNumpy(spec).decision_function(X[:6]), rtol=1e-10, atol=1e-12
    )


def test_rejects_more_nodes_than_a_byte_index_holds():
    """Neighbour lists are emitted as bytes, so the node count is bounded."""
    det, _ = _fitted()
    spec = extract_kan_gdn(det)
    spec["net"]["n_nodes"] = 300
    with pytest.raises(NotImplementedError, match="bytes"):
        generate_c(spec)


def test_generation_is_deterministic():
    """Regenerating the same detector must produce identical text."""
    det, _ = _fitted()
    spec = extract_kan_gdn(det)
    assert generate_c(spec)["kangdn.c"] == generate_c(spec)["kangdn.c"]


_MUTABLE_MAIN = r"""
#include <stdio.h>
#include "kangdn.h"

/* The realistic caller: a mutable buffer filled from sensor readings, then
   scored. A two-dimensional const array parameter would reject this under
   ISO C before C23, which is why the window is passed as a flat pointer. */
int main(void) {
    static kangdn_real_t win[KANGDN_WINDOW_SIZE][KANGDN_N_FEATURES];
    kangdn_real_t score = (kangdn_real_t)0;
    int t;
    int f;

    for (t = 0; t < KANGDN_WINDOW_SIZE; ++t) {
        for (f = 0; f < KANGDN_N_FEATURES; ++f) {
            win[t][f] = (kangdn_real_t)0.25;
        }
    }
    printf("%d\n", kangdn_score(&win[0][0], &score));
    return 0;
}
"""


@needs_cc
def test_caller_can_pass_a_mutable_buffer(tmp_path):
    """Regression: a caller filling its own window must not need a cast."""
    det, _ = _fitted()
    write_c(extract_kan_gdn(det), tmp_path, dtype="double")
    (tmp_path / "main.c").write_text(_MUTABLE_MAIN, encoding="utf-8")

    exe = tmp_path / "mutable.exe"
    subprocess.run(
        [
            CC,
            *CFLAGS,
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
    assert int(out.stdout.strip()) == 0  # KANGDN_OK

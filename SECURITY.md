# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/JulioAnzaldo/telemetry-anomdet/security/advisories/new)
(the "Report a vulnerability" button under the repository's **Security** tab).
Do not open a public issue for security problems.

We aim to acknowledge a report within a few days and to coordinate a fix and
disclosure with you. As a small project we follow a good-faith disclosure
timeline of up to 90 days.

## Supported Versions

telemetry-anomdet is pre-1.0 and moving quickly. Security fixes are applied to
the **latest released version only**; older versions are not patched
retroactively. Please upgrade to the newest release.

| Version    | Supported |
|------------|-----------|
| latest 0.x | yes       |
| older      | no        |

## Issues That Are Not Security Vulnerabilities

The following are handled as regular bugs, not security reports:

- Crashes, exceptions, or hangs caused by malformed or out-of-spec input.
- Numerical differences or detection-quality (precision/recall) issues.
- Resource exhaustion (memory or CPU) from very large inputs.

Because running the toolkit already implies local code execution, anything that
only affects the machine already running it is a bug, not a vulnerability.

## Using telemetry-anomdet Securely

- **Untrusted model files.** Detectors are serialized with Python `pickle`
  (`BaseDetector.save` / `load`). Unpickling executes arbitrary code, so only
  load model files you produced or fully trust.
- **Untrusted data files.** The SMAP loader reads NumPy `.npy` arrays and the CSV
  loader reads CSV files. Only load telemetry from sources you trust, keep NumPy
  up to date, and treat third-party `.npy` or pickled files as untrusted code.
- **Downstream services.** When the FastAPI service and LLM layer land, validate
  and authenticate all inputs at the service boundary and do not expose them
  unauthenticated to untrusted networks.

## Backporting Security Fixes

Security patches are applied to the current release line only and are not
backported to older versions.

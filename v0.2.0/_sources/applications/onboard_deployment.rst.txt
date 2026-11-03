Onboard Deployment Architecture
================================

This page describes how a trained detector is distilled, sized, and deployed onto
spacecraft flight hardware, and how responsibility is divided between the
spacecraft and the ground segment.

The target artifact is a KANGDN detector reduced to closed-form arithmetic: a
set of spline coefficients plus a deviation threshold, emitted as C with no
dependencies beyond ``math.h``. See
:mod:`telemetry_anomdet.models.deep.distill` for the extraction step and
:mod:`telemetry_anomdet.models.deep.codegen` for C generation.


Design principle
----------------

Perform every computation on the ground that can be performed on the ground.
Place onboard only what cannot wait for a ground contact.

Detection and the resulting trigger must be onboard, because a fault requiring
emergency operations cannot wait for the next pass. Threshold estimation,
retraining, explanation, and reporting are all slow-varying or offline
activities, so they belong to the ground segment, where model size, runtime, and
power are not constrained.

A consequence worth stating plainly: the ground segment can run models far too
large to fly, and use their output to calibrate the small onboard detector. The
flight artifact is a fast, auditable tripwire, tuned by a much larger process
that never leaves the ground.


Responsibility tiers
--------------------

.. list-table::
   :header-rows: 1
   :widths: 12 20 68

   * - Tier
     - Location
     - Responsibility
   * - 0
     - Onboard, hard real time
     - Distilled detector and fixed threshold. Deterministic, bounded runtime,
       no allocation. Raises the anomaly flag that gates emergency operations.
   * - 1
     - Onboard, background
     - EWMA smoothing of the error signal, accumulation of deviation
       statistics, emission of event packets, downlink of compact summaries
       rather than raw telemetry.
   * - 2
     - Ground segment
     - Full-precision models and ensembles, threshold recomputation from
       downlinked statistics, retraining, attribution and reporting, and
       parameter uplink.


Target hardware
---------------

Radiation-hardened flight processors trade process technology for
survivability, so they are typically slower than contemporary commercial parts
rather than faster. Representative classes:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Part
     - Architecture
     - Approximate clock
   * - RAD750
     - PowerPC, radiation hardened
     - 130 to 200 MHz
   * - RAD5500 family
     - PowerPC e5500, multicore
     - up to roughly 466 MHz
   * - GR712RC, GR740
     - LEON3 and LEON4 SPARC
     - roughly 80 MHz and 250 MHz
   * - Vorago VA416xx
     - Radiation hardened Cortex-M4
     - roughly 100 MHz

Many missions pair a radiation-hardened command and data handling computer with
a commercial payload processor that carries watchdog and power-cycling
mitigation. The distilled detector is designed for the first of these, where a
full neural network framework cannot be deployed at all.

**Portability constraint.** The generated C is C99 with no intrinsics, no
dynamic allocation, and no platform APIs, so it compiles unchanged for SPARC,
PowerPC, Cortex-M, RISC-V, and Xtensa targets. Platform-specific integration
(flash partitioning, packet handling, scheduling) belongs in a separate glue
layer and must not be introduced into the generated core.


Resource budget
---------------

Measured for a 25-channel model with a 29-sample context, across the embedding
widths the sizing sweep covers. Coefficients are ``static const`` and therefore
resident in flash; the RAM column is the scratch working set the generated code
declares.

.. list-table::
   :header-rows: 1
   :widths: 16 18 18 16 16 16

   * - embed_dim
     - Parameters
     - Flash (float32)
     - RAM scratch
     - MMACs
     - Approx. ms at 240 MHz
   * - 128
     - 159,232
     - 622.0 KB
     - 28.1 KB
     - 3.84
     - 16.0
   * - 64
     - 42,752
     - 167.0 KB
     - 15.6 KB
     - 1.00
     - 4.2
   * - 32
     - 12,160
     - 47.5 KB
     - 9.4 KB
     - 0.27
     - 1.1
   * - 16
     - 3,776
     - 14.8 KB
     - 6.2 KB
     - 0.08
     - 0.3
   * - 8
     - 1,312
     - 5.1 KB
     - 4.7 KB
     - 0.02
     - 0.1

Parameter count is dominated by the spline coefficients of the KAN activation
layer and grows as ``embed_dim**2 * (grid_size + spline_order)``, so embedding
width is the primary size control.

The millisecond column above is derived from the multiply-accumulate count and
assumes roughly one per cycle. Measurement does not support that assumption, so
treat it as a lower bound rather than an estimate; see the measured figures
below.


Measured on hardware
--------------------

An ESP32-S3 running the conformance harness in ``targets/esp32s3``, with a
detector trained on a single SMAP channel at ``embed_dim=16``, 25 channels and a
30 sample window, generated as ``float``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Measurement
     - Result
   * - Worst absolute score error
     - 2.78e-04, identical to the host build
   * - Anomaly flag mismatches
     - 0 of 8 golden vectors
   * - Heap consumed while scoring
     - 0 bytes
   * - Time per window
     - 19.3 ms
   * - RAM used by the firmware
     - 19,608 bytes
   * - Flash used by the firmware
     - 251,321 bytes, of which roughly 11 KB is coefficients

Two results are worth drawing out.

The score error matches the host build digit for digit, so the Xtensa single
precision unit and an x86 host agree exactly. Combined with the double precision
mode being exact against the reference evaluator, the whole chain from the
trained model to the deployed binary is verified rather than assumed.

Throughput is far below what the operation count suggests. The same model scores
a window in about 40 microseconds on an x86 host and 19.3 milliseconds on the
ESP32-S3, a factor of roughly 480. Coefficients are read from flash through the
cache rather than from RAM, the exponential is a library call made once per input
per layer, and the spline recursion is scalar. Which of those dominates has not
been profiled. At telemetry rates of about 1 Hz the detector still occupies only
a few percent of the processor, and triple modular redundancy remains affordable
at roughly 58 milliseconds, but a faster duty cycle would need the cost
investigated first.


Numeric precision
-----------------

``codegen`` emits either ``float`` or ``double``.

``float``
    The deployment mode. It matches single-precision FPUs on the relevant
    targets, and halves the flash footprint relative to ``double``.

``double``
    A verification mode. It isolates whether a discrepancy originates in the
    port or in reduced precision. It is not intended for flight, and on parts
    without double-precision hardware it is emulated in software.

Measured against the NumPy reference evaluator on the same windows, at two
model sizes: a small fixture (5 channels, ``embed_dim=8``) and a deployment
shaped model (25 channels, ``embed_dim=32``, 30 sample window).

.. list-table::
   :header-rows: 1
   :widths: 22 24 24 30

   * - Mode
     - Small model
     - Deployment model
     - Interpretation
   * - ``double``
     - 0.0
     - 1.5e-16
     - Exact to within one unit in the last place of float64. The port
       contributes no error of its own.
   * - ``float``
     - 3.0e-07
     - 1.2e-07
     - About one float32 epsilon (1.19e-07).

Single-precision error does not grow with node count or embedding width; the
deployment sized model tracks the reference more closely than the small one.
Because the double mode is exact, any discrepancy observed on a target can be
attributed to precision or to the platform's math library rather than to the
generated code. Drift of order 1e-07 is far below the margin separating nominal
from anomalous scores, so it does not affect the flag.

Some radiation-hardened microcontrollers have no floating point unit at all. The
spline bases are piecewise polynomials, so a fixed-point generation mode is
tractable and is a planned addition for those targets.


Threshold strategy
------------------

The onboard threshold is a constant, computed on the ground and uplinked. It is
not estimated onboard.

Rationale:

Auditability
    The constant that decides whether emergency operations are triggered is a
    reviewed, versioned, revertible value. An onboard decision can be reproduced
    exactly on the ground from the same inputs.

Determinism
    Identical input always produces identical output, which makes verification
    and regression testing meaningful.

Cost is not the deciding factor
    Onboard threshold estimation is inexpensive in absolute terms, requiring a
    rolling error buffer of a few kilobytes and well under a millisecond of
    computation. The argument for uplink rests on auditability, not resources.

The known limitation is adaptation lag. If the error distribution drifts through
thermal cycling, sensor ageing, or a mode change, a stale threshold degrades
until the next uplink. A planned refinement is bounded onboard adaptation, in
which the threshold may move only within a ground-approved envelope. That
preserves reviewability while recovering most of the responsiveness.

EWMA smoothing of the error signal runs onboard in all cases. It is part of the
score rather than the threshold, and costs one multiply-add per step.


Parameter updates
-----------------

Coefficients and thresholds are stored in a dedicated flash partition rather
than compiled into the application image. Updating a threshold then does not
require rebuilding or replacing executable code.

This mirrors established practice, where parameter and table uploads are a
lower-risk category than code patches, and where firmware images are held in
redundant banks with checksum verification and automatic rollback to a
known-good image.

Update sizes are modest:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Payload
     - Size
     - At 1 kbps
     - At 9.6 kbps
   * - Threshold and deviation statistics
     - 204 bytes
     - 1.6 s
     - 0.2 s
   * - Full model, ``embed_dim=16``
     - 14.8 KB
     - 120.8 s
     - 12.6 s
   * - Full model, ``embed_dim=32``
     - 47.5 KB
     - 389.1 s
     - 40.5 s

A threshold update is a single command packet. A complete model replacement is a
routine operation measured in minutes of contact time.


Radiation tolerance
-------------------

Qualification is a hardware and process activity. What the software layer
contributes is tolerance to corruption and evidence of correctness.

Required mitigations:

Parameter integrity
    Store the coefficient blob with a CRC and at least one redundant copy.
    Verify on boot and periodically during operation, and reload from a valid
    copy when a mismatch is detected. Single event upsets flip bits in both
    flash and RAM, and a corrupted coefficient silently changes the detector's
    behaviour.

Output plausibility checks
    Reject non-finite or out-of-range scores before acting on them. Such a value
    indicates corruption rather than an anomaly and must not reach the trigger
    path.

Redundant evaluation
    Where the fault budget requires it, evaluate three times and vote.

Watchdog and safe state
    Define the behaviour on timeout, and ensure the default is inaction rather
    than a spurious trigger.

The generated code already satisfies much of the NASA/JPL Power of Ten
guidance by construction: no dynamic allocation, no recursion, fixed loop
bounds, and no function pointers. Alignment with MISRA C requires explicit
types, no implicit conversions, mandatory braces, and no floating point equality
comparisons. Relevant process standards include NPR 7150.2 and NASA-STD-8739.8,
and ECSS-Q-ST-80C for European missions.


CCSDS integration
-----------------

Three separate interfaces make the detector usable within existing ground
infrastructure rather than requiring bespoke plumbing.

Telemetry input
    CCSDS Space Packet Protocol (133.0-B) carries the telemetry. XTCE describes
    how packets decode into named parameters, which maps onto the long-form
    ``[timestamp, variable, value]`` schema this toolkit uses throughout.

Event output
    Detections are emitted as event packets so that existing ground software can
    consume them directly.

Parameter and model upload
    CFDP (727.0-B) provides reliable, chunked, resumable file delivery across
    contact windows, which suits both threshold updates and full model
    replacement.


Target ports
------------

Platform specific code lives in ``targets``, one directory per target, and never
in the generated sources. ``targets/host`` is the reference implementation of a
port and ``targets/esp32s3`` is a bench validation of the software chain.

Generation can emit golden vectors alongside the detector: input windows, the
reference evaluator's score for each, and the expected anomaly flags. A port is
then validated on the target itself, by scoring each window and comparing, with
no host tooling in the loop. A port is complete when it compiles clean under the
strictest warnings its toolchain supports, matches the golden vectors with the
flags exactly right, and reports timing and memory.

See ``targets/README.md`` for the per target instructions.


Provenance and interface stability
----------------------------------

Flight software has to be traceable to what produced it, so every generated file
records the toolkit version that emitted it in its banner, and ``kangdn.h``
exposes it as ``KANGDN_VERSION`` for a running build to report. Record that
version alongside any artifact that is flashed or flown.

The generated interface is unstable until version 1.0.0. Symbol names, macros
and entry point signatures may change in any 0.x release, per the
`Semantic Versioning <https://semver.org/>`_ rule for major version zero.
Upgrading means regenerating and rebuilding rather than hand-patching, then
re-running the golden vectors, which are regenerated with the sources and are
the evidence that the new version behaves as the old one did.

Generation is deterministic and carries no timestamp. The same fitted detector
reproduces the sources byte for byte, so a diff shows only real changes and a
build is reproducible from a stored detector.


Open items
----------

The following are identified but not yet resolved.

- Profiling the inference cost on the ESP32-S3, which is far above what the
  operation count predicts.
- Fixed-point generation mode for targets without a floating point unit.
- Bounded onboard threshold adaptation within a ground-approved envelope.
- MISRA C conformance pass over the generator output.
- Separation of the coefficient blob into an independently updatable partition,
  with CRC and redundancy, replacing the current compiled-in arrays.
- Selection of the deployed embedding width, which depends on detection quality
  measured under a metric that is not inflated by point adjustment.

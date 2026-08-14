# ESP32-S3 port

Bench validation of the distilled detector on real hardware: the Xtensa
toolchain, the single precision FPU path, access to flash resident `static
const` parameters through the cache, and measured timing.

## Build and run

Requires ESP-IDF v5.x on the path.

```
python examples/export_kangdn_c.py targets/esp32s3/main/generated
cd targets/esp32s3
idf.py set-target esp32s3
idf.py build flash monitor
```

Substitute another chip in `set-target` to try a different part; nothing in the
detector is specific to the S3.

## Expected output

```
=== distilled KANGDN conformance ===
window: 30 x 25  real_t: 4 bytes
vectors: 8  rtol: 0.0001  atol: 1e-05
worst absolute score error: ...
flag mismatches: 0 of 8
time per window: ... us
heap delta across scoring: 0 bytes
PASS
```

`flag mismatches: 0` and `PASS` are the results that matter. A non-zero heap
delta would mean something in the scoring path allocates, which it must not.

## Notes

The S3 has a single precision FPU and no double precision hardware, so generate
with the default `float`. A `double` build works but is emulated in software and
much slower; it is a diagnostic mode, useful only for deciding whether a
discrepancy comes from the port or from precision.

Coefficients are `static const` and land in flash rather than RAM. Confirm with
the size report after a build:

```
idf.py size-components
```

`partitions.csv` reserves a 256K `model` partition. It is unused today and
exists for the planned split of the coefficient blob out of the application
image, so that a threshold update becomes a data write instead of a full
firmware replacement.

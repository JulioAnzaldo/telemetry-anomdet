/* Conformance harness for the distilled detector, host build.
 *
 * Scores every golden vector and checks two things:
 *
 *   1. the score matches the reference within the generated tolerances, and
 *   2. the anomaly flag matches exactly.
 *
 * The flag is the decisive check. A score may drift in the last digits at
 * single precision, particularly when it is small compared with the
 * intermediate magnitudes, but the flag is what the rest of the system acts on
 * and it must be reproduced exactly.
 *
 * This is the reference implementation of a target port. A new target should do
 * the same three things and report the same three results.
 */
#include <math.h>
#include <stdio.h>
#include <time.h>

#include "kangdn.h"
#include "kangdn_vectors.h"

#define TIMING_REPEATS 100

int main(void)
{
    kangdn_real_t score = (kangdn_real_t)0;
    double worst_absolute = 0.0;
    int failures = 0;
    int flag_failures = 0;
    int i;
    clock_t start;
    double elapsed_us;

    printf("vectors: %d  rtol: %g  atol: %g\n",
           KANGDN_N_VECTORS, KANGDN_TEST_RTOL, KANGDN_TEST_ATOL);

    for (i = 0; i < KANGDN_N_VECTORS; ++i) {
        const double want = KANGDN_TEST_SCORES[i];
        int flag = 0;
        int status;
        double difference;
        double allowed;

        status = kangdn_score(KANGDN_TEST_WINDOWS[i], &score);
        if (status != KANGDN_OK) {
            printf("  [%d] FAIL score status %d\n", i, status);
            ++failures;
            continue;
        }
        status = kangdn_is_anomaly(KANGDN_TEST_WINDOWS[i], &flag);
        if (status != KANGDN_OK) {
            printf("  [%d] FAIL flag status %d\n", i, status);
            ++failures;
            continue;
        }

        difference = fabs((double)score - want);
        allowed = KANGDN_TEST_ATOL + (KANGDN_TEST_RTOL * fabs(want));
        if (difference > worst_absolute) {
            worst_absolute = difference;
        }
        if (difference > allowed) {
            printf("  [%d] FAIL score got %.17g want %.17g (diff %.3g > %.3g)\n",
                   i, (double)score, want, difference, allowed);
            ++failures;
        }
        if (flag != KANGDN_TEST_FLAGS[i]) {
            printf("  [%d] FAIL flag got %d want %d\n", i, flag, KANGDN_TEST_FLAGS[i]);
            ++flag_failures;
            ++failures;
        }
    }

    /* Timing, averaged so a single noisy sample does not dominate. */
    start = clock();
    for (i = 0; i < TIMING_REPEATS; ++i) {
        (void)kangdn_score(KANGDN_TEST_WINDOWS[i % KANGDN_N_VECTORS], &score);
    }
    elapsed_us = (double)(clock() - start) / CLOCKS_PER_SEC * 1e6 / TIMING_REPEATS;

    printf("worst absolute score error: %.3g\n", worst_absolute);
    printf("flag mismatches: %d of %d\n", flag_failures, KANGDN_N_VECTORS);
    printf("time per window: %.1f us\n", elapsed_us);
    printf("%s\n", (failures == 0) ? "PASS" : "FAIL");
    return (failures == 0) ? 0 : 1;
}

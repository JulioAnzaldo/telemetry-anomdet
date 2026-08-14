/* Conformance harness for the distilled detector, ESP32-S3 build.
 *
 * Mirrors targets/host/main.c: score every golden vector, check the score
 * against the reference within the generated tolerances, and check the anomaly
 * flag exactly. The flag is the decisive result, since it is what the rest of
 * the system acts on.
 *
 * The detector itself is plain C99 and contains nothing platform specific. All
 * ESP-IDF dependence lives in this file, which is what allows the same
 * generated sources to build for a flight processor without modification.
 */
#include <math.h>
#include <stdio.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "kangdn.h"
#include "kangdn_vectors.h"

#define TIMING_REPEATS 100

void app_main(void)
{
    kangdn_real_t score = (kangdn_real_t)0;
    double worst_absolute = 0.0;
    int failures = 0;
    int flag_failures = 0;
    int i;
    int64_t start;
    double elapsed_us;
    size_t heap_before;
    size_t heap_after;

    printf("\n=== distilled KANGDN conformance ===\n");
    printf("window: %d x %d  real_t: %u bytes\n",
           KANGDN_WINDOW_SIZE, KANGDN_N_FEATURES, (unsigned)sizeof(kangdn_real_t));
    printf("vectors: %d  rtol: %g  atol: %g\n",
           KANGDN_N_VECTORS, KANGDN_TEST_RTOL, KANGDN_TEST_ATOL);

    heap_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

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
            printf("  [%d] FAIL score got %.9g want %.9g (diff %.3g > %.3g)\n",
                   i, (double)score, want, difference, allowed);
            ++failures;
        }
        if (flag != KANGDN_TEST_FLAGS[i]) {
            printf("  [%d] FAIL flag got %d want %d\n", i, flag, KANGDN_TEST_FLAGS[i]);
            ++flag_failures;
            ++failures;
        }
    }

    heap_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

    start = esp_timer_get_time();
    for (i = 0; i < TIMING_REPEATS; ++i) {
        (void)kangdn_score(KANGDN_TEST_WINDOWS[i % KANGDN_N_VECTORS], &score);
    }
    elapsed_us = (double)(esp_timer_get_time() - start) / (double)TIMING_REPEATS;

    printf("worst absolute score error: %.3g\n", worst_absolute);
    printf("flag mismatches: %d of %d\n", flag_failures, KANGDN_N_VECTORS);
    printf("time per window: %.1f us\n", elapsed_us);
    /* Scoring allocates nothing, so this difference should be zero. */
    printf("heap delta across scoring: %d bytes\n", (int)(heap_before - heap_after));
    printf("%s\n", (failures == 0) ? "PASS" : "FAIL");

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}

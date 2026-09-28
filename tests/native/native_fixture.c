#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DE_OK 0
#define DE_INVALID_ARGUMENT 1
#define DE_MODEL_NOT_FOUND 2
#define DE_ENGINE_ERROR 6

typedef struct de_engine {
    char model[128];
    unsigned int calls;
} de_engine_t;

static _Thread_local char LAST_ERROR[256] = "";

static int fail(int status, const char *message) {
    snprintf(LAST_ERROR, sizeof(LAST_ERROR), "%s", message);
    return status;
}

uint32_t de_abi_version(void) { return 1u; }

const char *de_version(void) { return "test-native-1"; }

int de_engine_create(
    const char *model_id,
    const char *options_json,
    de_engine_t **out_engine
) {
    (void)options_json;
    if (model_id == NULL || out_engine == NULL) {
        return fail(DE_INVALID_ARGUMENT, "model_id and out_engine are required");
    }
    *out_engine = NULL;
    if (strcmp(model_id, "missing/model") == 0) {
        return fail(DE_MODEL_NOT_FOUND, "model is not installed");
    }
    de_engine_t *engine = calloc(1, sizeof(de_engine_t));
    if (engine == NULL) {
        return fail(DE_ENGINE_ERROR, "allocation failed");
    }
    snprintf(engine->model, sizeof(engine->model), "%s", model_id);
    *out_engine = engine;
    LAST_ERROR[0] = '\0';
    return DE_OK;
}

int de_engine_decide_json(
    de_engine_t *engine,
    const char *request_json,
    char **out_response_json
) {
    if (engine == NULL || request_json == NULL || out_response_json == NULL) {
        return fail(DE_INVALID_ARGUMENT, "engine, request, and output are required");
    }
    *out_response_json = NULL;
    if (strstr(request_json, "force_native_error") != NULL) {
        return fail(DE_ENGINE_ERROR, "forced native fixture failure");
    }
    const unsigned int hits = engine->calls == 0 ? 0u : 3u;
    const unsigned int misses = engine->calls == 0 ? 3u : 0u;
    engine->calls += 1;

    const char *format =
        "{\"id\":\"dec_fixture\",\"model\":\"%s\",\"created\":1,"
        "\"results\":{"
        "\"route\":{\"type\":\"choice\",\"selected\":\"billing\","
        "\"probabilities\":{\"billing\":0.9,\"technical\":0.05,\"sales\":0.05},"
        "\"confidence\":0.8},"
        "\"urgent\":{\"type\":\"noul\",\"value\":0.8,\"confidence\":0.6},"
        "\"severity\":{\"type\":\"score\",\"value\":1.6,"
        "\"distribution\":{\"0\":0.1,\"1\":0.2,\"2\":0.7},"
        "\"legend\":{\"0\":\"minor\",\"1\":\"blocked\",\"2\":\"critical\"},"
        "\"confidence\":0.5}},"
        "\"usage\":{\"input_characters\":42,\"questions\":3,\"candidates\":8,"
        "\"candidate_cache_hits\":%u,\"candidate_cache_misses\":%u}}";
    const int required = snprintf(NULL, 0, format, engine->model, hits, misses);
    if (required < 0) {
        return fail(DE_ENGINE_ERROR, "could not format response");
    }
    char *response = malloc((size_t)required + 1u);
    if (response == NULL) {
        return fail(DE_ENGINE_ERROR, "allocation failed");
    }
    snprintf(response, (size_t)required + 1u, format, engine->model, hits, misses);
    *out_response_json = response;
    LAST_ERROR[0] = '\0';
    return DE_OK;
}

void de_engine_destroy(de_engine_t *engine) { free(engine); }

void de_string_free(char *value) { free(value); }

const char *de_last_error(void) { return LAST_ERROR; }

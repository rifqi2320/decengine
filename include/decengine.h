#ifndef DECENGINE_H
#define DECENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DE_ABI_VERSION 1u

typedef struct de_engine de_engine_t;

typedef enum de_status {
    DE_OK = 0,
    DE_INVALID_ARGUMENT = 1,
    DE_MODEL_NOT_FOUND = 2,
    DE_UNSUPPORTED_MODEL = 3,
    DE_MODEL_LOAD_FAILED = 4,
    DE_CONTEXT_TOO_LONG = 5,
    DE_ENGINE_ERROR = 6,
    DE_OUT_OF_MEMORY = 7,
    DE_INVALID_MANIFEST = 8,
    DE_INTERNAL_ERROR = 255
} de_status_t;

uint32_t de_abi_version(void);
const char *de_version(void);

de_status_t de_engine_create(
    const char *model_id,
    const char *options_json,
    de_engine_t **out_engine
);

de_status_t de_engine_decide_json(
    de_engine_t *engine,
    const char *request_json,
    char **out_response_json
);

void de_engine_destroy(de_engine_t *engine);
void de_string_free(char *value);
const char *de_last_error(void);

#ifdef __cplusplus
}
#endif

#endif

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <limits.h>

typedef struct {
    uint32_t sequence_len;
    uint32_t vocab_size;
    uint32_t padded_vocab_size;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_kv_head;
    uint32_t n_embd;
    char *window_pattern;
    uint32_t *attention_left;  // [n_layer], UINT32_MAX means global attention
    uint32_t head_dim;
    uint32_t intermediate_size;
    float rope_local_theta;
    float rope_global_theta;
    float rms_norm_eps;
    float qk_multiplier;
    float logits_soft_cap;
} Config;

typedef struct {
    float *q;        // [n_head * head_dim, n_embd]
    float *k;        // [n_kv_head * head_dim, n_embd]
    float *v;        // [n_kv_head * head_dim, n_embd]
    float *proj;     // [n_embd, n_head * head_dim]
    float *q_scale;  // [head_dim], RMSNorm gain = 1 + scale
    float *k_scale;  // [head_dim], no bias
} AttentionWeights;

typedef struct {
    float *fc1;  // [intermediate_size, n_embd], GELU branch
    float *fc2;  // [intermediate_size, n_embd], linear branch
    float *fc3;  // [n_embd, intermediate_size]
} FeedForwardWeights;

/* Initialize with {0}. Each weight pointer owns a separate allocation. */
typedef struct {
    AttentionWeights attention;
    FeedForwardWeights ff;
    float *pre_attention_scale;     // [n_embd]
    float *post_attention_scale;    // [n_embd]
    float *pre_feedforward_scale;   // [n_embd]
    float *post_feedforward_scale;  // [n_embd]
} TransformerBlockWeights;

typedef struct {
    float *embedding;                 // [padded_vocab_size, n_embd]
    TransformerBlockWeights *blocks;  // [n_layer]
    float *final_scale;               // [n_embd]
    float *lm_head;                   // [padded_vocab_size, n_embd], untied
    float *resid_lambdas;             // [n_layer]
    float *x0_lambdas;                // [n_layer]
    float *smear_gate;                // [1, min(24, n_embd)]
    float *smear_lambda;              // [1]
} TransformerWeights;

void free_attention_weights(AttentionWeights *weights)
{
    if (weights == NULL) return;
    free(weights->q);
    free(weights->k);
    free(weights->v);
    free(weights->proj);
    free(weights->q_scale);
    free(weights->k_scale);
    *weights = (AttentionWeights){0};
}

void free_feedforward_weights(FeedForwardWeights *weights)
{
    if (weights == NULL) return;
    free(weights->fc1);
    free(weights->fc2);
    free(weights->fc3);
    *weights = (FeedForwardWeights){0};
}

void free_transformer_block_weights(TransformerBlockWeights *weights)
{
    if (weights == NULL) return;
    free_attention_weights(&weights->attention);
    free_feedforward_weights(&weights->ff);
    free(weights->pre_attention_scale);
    free(weights->post_attention_scale);
    free(weights->pre_feedforward_scale);
    free(weights->post_feedforward_scale);
    *weights = (TransformerBlockWeights){0};
}

/* n_layer must match the number of zero-initialized blocks allocated. */
void free_transformer_weights(TransformerWeights *weights, uint32_t n_layer)
{
    if (weights == NULL) return;
    if (weights->blocks != NULL)
        for (uint32_t i = 0; i < n_layer; ++i)
            free_transformer_block_weights(&weights->blocks[i]);
    free(weights->blocks);
    free(weights->embedding);
    free(weights->final_scale);
    free(weights->lm_head);
    free(weights->resid_lambdas);
    free(weights->x0_lambdas);
    free(weights->smear_gate);
    free(weights->smear_lambda);
    *weights = (TransformerWeights){0};
}


/* window_pattern and attention_left are owned heap allocations. */
void free_config(Config *config)
{
    if (config == NULL) return;
    free(config->window_pattern);
    free(config->attention_left);
    *config = (Config){0};
}

void linear(float *output, const float *input, const float *weight, const uint32_t *rows, const uint32_t *in_features, const uint32_t *out_features)
{
    #pragma omp parallel for collapse(2)
    for (size_t row = 0; row < *rows; ++row) {
        for (size_t out = 0; out < *out_features; ++out) {
            float sum = 0.0f;
            for (size_t in = 0; in < *in_features; ++in)
                sum += input[row * *in_features + in] * weight[out * *in_features + in];
            output[row * *out_features + out] = sum;
        }
    }
}

void gelu(float *output, const float *input, const uint32_t *count)
{
    #pragma omp parallel for
    for (size_t i = 0; i < *count; ++i) {
        float x = input[i];
        float argument = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        output[i] = (0.5f * x) * (1.0f + tanhf(argument));
    }
}

uint32_t read_uint32(FILE *file)
{
    uint32_t value;
    if (fread(&value, sizeof(uint32_t), 1, file) != 1)
        exit(1);
    return value;
}

float read_float32(FILE *file)
{
    float value;
    if (fread(&value, sizeof(float), 1, file) != 1)
        exit(1);
    return value;
}

float* read_tensor(FILE *file)
{
    char tensor_name[64];
    if (fread(tensor_name, sizeof(char), 64, file) != 64)
        exit(1);

    uint32_t dims[2];
    if (fread(dims, sizeof(uint32_t), 2, file) != 2)
        exit(1);

    size_t count = (size_t)dims[0] * dims[1];
    float *data = malloc(sizeof(float) * count);
    if (fread(data, sizeof(float), count, file) != count)
        exit(1);
    printf("%s read succesfully\n", tensor_name);

    return data;
}

void load_model(const char *path, Config *config, TransformerWeights *weights, uint32_t **windows)
{
    FILE *file = fopen(path, "rb");

    if (file == NULL) {
        perror("Cannot open file");
        exit(1);
    }

    // 1. Read config
    config->sequence_len = read_uint32(file);
    config->vocab_size = read_uint32(file);
    config->padded_vocab_size = read_uint32(file);

    config->n_layer = read_uint32(file);
    config->n_head = read_uint32(file);
    config->n_kv_head = read_uint32(file);
    config->n_embd = read_uint32(file);

    config->head_dim = read_uint32(file);
    config->intermediate_size = read_uint32(file);

    config->rope_local_theta = read_float32(file);
    config->rope_global_theta = read_float32(file);
    config->rms_norm_eps = read_float32(file);
    config->qk_multiplier = read_float32(file);
    config->logits_soft_cap = read_float32(file);

    config->window_pattern = malloc(32);

    if (config->window_pattern == NULL)
        exit(EXIT_FAILURE);

    if (fread(config->window_pattern, 1, 32, file) != 32)
        exit(EXIT_FAILURE);

    uint32_t tensor_count = read_uint32(file);

    // 2. Read attention windows
    *windows = malloc(config->n_layer * sizeof(uint32_t));

    if (*windows == NULL)
        exit(1);

    if (fread(*windows, sizeof(uint32_t), config->n_layer, file)
        != config->n_layer)
        exit(1);

    // 3. Embedding
    weights->embedding = read_tensor(file);

    // 4. Transformer blocks
    weights->blocks = calloc(config->n_layer, sizeof(TransformerBlockWeights));

    if (weights->blocks == NULL)
        exit(1);

    for (uint32_t i = 0; i < config->n_layer; i++) {

        TransformerBlockWeights *block = &weights->blocks[i];

        // Attention
        block->attention.q = read_tensor(file);
        block->attention.k = read_tensor(file);
        block->attention.v = read_tensor(file);
        block->attention.proj = read_tensor(file);

        block->attention.q_scale = read_tensor(file);
        block->attention.k_scale = read_tensor(file);

        // FeedForward
        block->ff.fc1 = read_tensor(file);
        block->ff.fc2 = read_tensor(file);
        block->ff.fc3 = read_tensor(file);

        // Norms
        block->pre_attention_scale = read_tensor(file);
        block->post_attention_scale = read_tensor(file);
        block->pre_feedforward_scale = read_tensor(file);
        block->post_feedforward_scale = read_tensor(file);
    }

    // 5. Final weights
    weights->final_scale = read_tensor(file);
    weights->lm_head = read_tensor(file);
    weights->resid_lambdas = read_tensor(file);
    weights->x0_lambdas = read_tensor(file);
    weights->smear_gate = read_tensor(file);
    weights->smear_lambda = read_tensor(file);

    fclose(file);

    printf("Model loaded successfully!\n");
    printf("Layers: %u\n", config->n_layer);
    printf("Tensors: %u\n", tensor_count);
}

int main(int argc, char **argv)
{   
    const char *path = argc > 1 ? argv[1] : "./models/model.bin";
    
    Config config = {0};
    TransformerWeights weights = {0};
    uint32_t *windows = NULL;

    load_model(path, &config, &weights, &windows);

    printf("Sequence length: %u\n", config.sequence_len);
    printf("Vocabulary size: %u\n", config.vocab_size);
    printf("Embedding dimension: %u\n", config.n_embd);

    printf("First embedding weight: %f\n", weights.embedding[0]);

    printf("First Q weight: %f\n", weights.blocks[0].attention.q[0]);

    free_transformer_weights(&weights, config.n_layer);
    free_config(&config);
    free(windows);

    return 0;
}

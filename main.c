#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdint.h>


void linear(float *output, const float *input, const float *weight, uint32_t rows, uint32_t in_features, uint32_t out_features)
{
    #pragma omp parallel for collapse(2)
    for (size_t row = 0; row < rows; ++row) {
        for (size_t out = 0; out < out_features; ++out) {
            float sum = 0.0f;
            for (size_t in = 0; in < in_features; ++in)
                sum += input[row * in_features + in] * weight[out * in_features + in];
            output[row * out_features + out] = sum;
        }
    }
}

int32_t main(void)
{   
    uint32_t rows = 512;
    uint32_t in_features = 1024;
    uint32_t out_features = 4096;
    
    float *x = (float *)malloc(sizeof(float) * rows * in_features);
    float *w = (float *)malloc(sizeof(float) * in_features * out_features);
    float *y = (float *)malloc(sizeof(float) * rows * out_features);
    
    for (uint32_t i = 0; i < rows * in_features; ++i) {
        *(x + i) = i;
    }
    for (uint32_t i = 0; i < in_features * out_features; ++i) {
        *(w + i) = i;
    }

    double s = omp_get_wtime();
    linear(y, x, w, rows, in_features, out_features);
    double e = omp_get_wtime();
    printf("time: %f", e - s);

    // for (uint32_t i = 0; i < rows; ++i) {
    //     printf("[");
    //     for (uint32_t j = 0; j < out_features; ++j) {
    //         printf("%f", *(y + j));
    //     }
    //     printf("]\n");
    // }

    return 0;
}
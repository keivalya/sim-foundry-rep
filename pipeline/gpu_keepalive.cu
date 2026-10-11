// Low-intensity GPU keep-alive: periodic small SGEMMs so a reserved GPU
// does not register as idle. Deliberately modest: ~200 MB, duty-cycled.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <time.h>

int main(int argc, char** argv) {
    int n      = (argc > 1) ? atoi(argv[1]) : 4096;   // matrix dim
    int burst  = (argc > 2) ? atoi(argv[2]) : 40;     // gemms per burst
    int nap_ms = (argc > 3) ? atoi(argv[3]) : 150;    // ms idle between bursts

    float *A, *B, *C;
    size_t bytes = (size_t)n * n * sizeof(float);
    if (cudaMalloc(&A, bytes) || cudaMalloc(&B, bytes) || cudaMalloc(&C, bytes)) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }
    cudaMemset(A, 1, bytes); cudaMemset(B, 1, bytes);

    cublasHandle_t h; cublasCreate(&h);
    const float alpha = 1.0f, beta = 0.0f;
    printf("keepalive: n=%d burst=%d nap=%dms  (%.0f MB)\n",
           n, burst, nap_ms, 3.0*bytes/1e6); fflush(stdout);

    long iter = 0;
    while (1) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < burst; ++i)
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                        &alpha, A, n, B, n, &beta, C, n);
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double s = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9;
        double tflops = 2.0*n*n*n*burst / s / 1e12;
        if (++iter % 12 == 1)
            printf("burst %ld: %.2f s  %.2f TFLOP/s fp32\n", iter, s, tflops), fflush(stdout);
        usleep(nap_ms * 1000);
    }
}

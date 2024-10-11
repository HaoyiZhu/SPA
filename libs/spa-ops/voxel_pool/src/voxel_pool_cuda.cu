#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void voxel_pool_kernel(int b, int z, int y, int x, int n, int c, int n_intervals,
                                  const float* feats,
                                  const int* coords,
                                  const int* interval_starts,
                                  const int* interval_lengths,
                                  float* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  const int* cur_coords = coords + interval_start * 4;
  const float* cur_feats = feats + interval_start * c + cur_c;
  // (b, c, z, y, x)
  float* cur_out = out + cur_coords[0] * c * z * y * x + cur_c * z * y * x +
    cur_coords[1] * y * x + cur_coords[2] * x + 
    cur_coords[3];
  float psum = 0;
  for(int i = 0; i < interval_length; i++){
    psum += cur_feats[i * c];
  }
  *cur_out = psum;
}


__global__ void voxel_pool_grad_kernel(int b, int z, int y, int x, int n, int c, int n_intervals,
                                  const float* out_grad,
                                  const int* coords,
                                  const int* interval_starts,
                                  const int* interval_lengths,
                                  float* feats_grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  const int* cur_coords = coords + interval_start * 4;
  float* cur_feats_grad = feats_grad + interval_start * c + cur_c;
  const float* cur_out_grad = out_grad + cur_coords[0] * c * z * y * x + 
    cur_c * z * y * x + cur_coords[1] * y * x + 
    cur_coords[2] * x + cur_coords[3];
  for(int i = 0; i < interval_length; i++){
    cur_feats_grad[i * c] = *cur_out_grad;
  }
}

void voxel_pool(int b, int z, int y, int x, int n, int c, int n_intervals, const float* feats,
  const int* coords, const int* interval_starts, const int* interval_lengths, float* out) {
  voxel_pool_kernel<<<DIVUP(n_intervals * c, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
    b, z, y, x, n, c, n_intervals, feats, coords, interval_starts, interval_lengths, out
  );
}

void voxel_pool_grad(int b, int z, int y, int x, int n, int c, int n_intervals, const float* out_grad,
  const int* coords, const int* interval_starts, const int* interval_lengths, float* feats_grad) {
  voxel_pool_grad_kernel<<<DIVUP(n_intervals * c, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
    b, z, y, x, n, c, n_intervals, out_grad, coords, interval_starts, interval_lengths, feats_grad
  );
}

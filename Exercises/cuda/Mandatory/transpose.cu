#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define TILE_SIZE 32
#define N_ROWS 8
#define N_TEST 10

__global__ void naive_transpose(float * A, float * B)
{
  int x = blockIdx.x * TILE_SIZE + threadIdx.x;
  int y = blockIdx.y * TILE_SIZE + threadIdx.y;
  int start = gridDim.x * TILE_SIZE;
  
  int j;
  for(j = 0; j < TILE_SIZE; j += N_ROWS)
    B[x * start + (y + j)] = A[(y + j)* start + x]; 
} 


__global__ void fast_transpose(float * A, float * B)
{
  __shared__ float tile[TILE_SIZE][TILE_SIZE];
  int x = blockIdx.x * TILE_SIZE + threadIdx.x;
  int y = blockIdx.y * TILE_SIZE + threadIdx.y;
  int start = gridDim.x * TILE_SIZE;

  int j;
  for(j = 0; j < TILE_SIZE; j+= N_ROWS)
    tile[threadIdx.y + j][threadIdx.x] = A[(y + j)*start + x];

  __syncthreads();

  x = blockIdx.y * TILE_SIZE + threadIdx.x;
  y = blockIdx.x * TILE_SIZE + threadIdx.y;

  for(j = 0; j < TILE_SIZE; j += N_ROWS)
    B[(y + j)*start + x] = tile[threadIdx.x][threadIdx.y + j];
}


void fill_host(float* host_a, const int nx, const int ny)
{
  int i, j;
  for(i = 0; i < ny; i++)
    for(j = 0; j < nx; j++)
      host_a[i * nx + j] = i * nx + j; 
}

void cpu_transpose(float * host_a, float * host_control, const int nx, const int ny)
{
  int i, j;
  for(i = 0; i < ny; i++)
    for(j = 0; j < nx; j++)
      host_control[i * nx + j] = host_a[j * nx + i];
}
int is_transpose(float * host_control, float * host_transpose, int size)
{
  int result, i;
  for(i = 0; i < size; i++){
    if(host_control[i] != host_transpose[i]){
      result = 0;
      return result;
    }
  }

  result = 1;
  return result;
}

int main(int argc, char * argv[])
{
  const int nx = 1024;
  const int ny = 1024;
  const int size = nx * ny * sizeof(float);

  dim3 dimGrid(nx/TILE_SIZE, ny/N_ROWS, 1);
  dim3 dimBlock(TILE_SIZE, N_ROWS, 1);

  float * host_a = (float*)malloc(size);
  float * host_naive = (float*)malloc(size);
  float * host_fast = (float*)malloc(size);
  float * host_control = (float*)malloc(size);

  float * device_a, * device_naive,* device_fast;
     
  cudaMalloc((void**)&device_a, size);
  cudaMalloc((void**)&device_naive, size);
  cudaMalloc((void**)&device_fast, size);

  fill_host(host_a, nx, ny);
  
  cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  float ms;

  printf("%25s", "naive_transpose");

  cudaEventRecord(begin);
  naive_transpose<<<dimGrid, dimBlock>>>(device_a, device_naive);
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&ms, begin, end);
  cudaMemcpy(host_naive, device_naive, size, cudaMemcpyDeviceToHost);
  
  int correctness = is_transpose(host_control, host_naive, nx * ny);
  
  if(correctness)
    printf("%20.5s\n", 2 * size * 1e-16 * (1/ms));

  return 0;
}

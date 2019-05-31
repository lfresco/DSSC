#include <stdio.h>
#include <math.h>
#include <stdlib.h>

const int TILE_SIZE = 8;
const int N_ROWS = 4;
#define N_TEST 1

__global__ void naive_transpose(const float * A, float * B, int size)
{ 
  /**
  int x = blockIdx.x * TILE_SIZE + threadIdx.x;
  int y = blockIdx.y * TILE_SIZE + threadIdx.y;
  int start = gridDim.x * TILE_SIZE;
  
  int j;
  for(j = 0; j < TILE_SIZE; j += N_ROWS)
    B[x * start + (y + j)] = A[(y + j)* start + x]; 
*/
 int row = blockIdx.x * blockDim.x + threadIdx.x;
 int col = blockIdx.y * blockDim.y + threadIdx.y;
 
 B[col * size + row] = A[row * size + col]; 
} 


__global__ void fast_transpose(const float * A, float * B, const int size)
{
  __shared__ float tile[TILE_SIZE][TILE_SIZE];
  /**int x = blockIdx.x * TILE_SIZE + threadIdx.x;
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
  */
 int col = blockIdx.x * blockDim.x + threadIdx.x;
 int row = blockIdx.y * blockDim.y + threadIdx.y;
 tile[threadIdx.x][threadIdx.y] = A[row * size + col];
 __syncthreads();
 B[col * size + row] = tile[threadIdx.x][threadIdx.y];

  
}


void fill_host(float* host_a, const int N)
{
  int i,j;
  for(j = 0; j < N; j++){
    for(i =  0; i < N; i++)
      host_a[j * N + i] = i;
  }
      
}

void cpu_transpose(float * host_a, float * host_control, const int N)
{
  int i, j;
  for(j = 0; j < N; j++)
    for(i = 0; i < N; i++)
      host_control[j * N + i] = host_a[i * N + j];
}
int is_transpose(float * host_control, float * host_transpose, const int N)
{
  int i,j;
  for(j = 0; j < N; j++){
    for(i = 0; i < N; i++){
      if(host_control[j*N + i] != host_transpose[j*N + i]){
          return 0;
      }
    }
 }
  
  return 1;
}

void print_matrix(const float * A, const int N)
{
  int i, j;
  for(i = 0; i < N; i++){
    for(j = 0; j < N; j++)
      printf("%.1f ", A[i*N + j]);

    printf("\n");
  }
  printf("\n");
}

int main(int argc, char * argv[])
{
  const int N = 16;
  
  const int size = N * N * sizeof(float);

  dim3 dimGrid(N/TILE_SIZE, N/N_ROWS, 1);
  dim3 dimBlock(TILE_SIZE, N_ROWS, 1);

  float * host_a = (float*)malloc(size);
  float * host_naive = (float*)malloc(size);
  float * host_fast = (float*)malloc(size);
  float * host_control = (float*)malloc(size);

  float * device_a, * device_naive,* device_fast;
     
  cudaMalloc((void**)&device_a, size);
  cudaMalloc((void**)&device_naive, size);
  cudaMalloc((void**)&device_fast, size);

  fill_host(host_a, N);
  cpu_transpose(host_a, host_control, N);  
    
  print_matrix(host_a, N);
  print_matrix(host_control, N);

  cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  float  ms;

  printf("%25s\n", "naive_transpose");

  cudaEventRecord(begin,0);
  naive_transpose<<< dimGrid, dimBlock >>>(device_a, device_naive, N);
  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&ms, begin, end);
  cudaMemcpy(host_naive, device_naive, size, cudaMemcpyDeviceToHost);
  
  print_matrix(host_naive, N);

  //int correctness = is_transpose(host_control, host_naive, nx , ny);
  
  //printf("Correctness Test : %d\n", is_transpose(host_control, host_naive, nx, ny));
  
  //printf("Required time : %f\n", ms);
  
  cudaEventRecord(begin, 0);
  fast_transpose<<< dimGrid, dimBlock >>>(device_a, device_fast, N);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&ms, begin, end);
  cudaMemcpy(host_fast, device_fast, size, cudaMemcpyDeviceToHost);
  
  print_matrix(host_fast, N);  
  

  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  cudaFree(device_a);
  cudaFree(device_naive);
  cudaFree(device_fast);
  free(host_a);
  free(host_control);
  free(host_naive);
  free(host_fast);
  return 0;
}

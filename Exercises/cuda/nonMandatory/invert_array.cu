#include <stdio.h>
#include <math.h>

__global__ void reverse(int * a, int* b){
  
  
}


void print_array(int* array, int size){
  int i = 0;
  for(i; i < size; i++){
    printf("%lf ",array[i]);
  }
}

#define N 512

int main(void){
 int size = 5*sizeof(int);
 int d_in[5] = {100, 110, 200, 220, 300};
 int * dev_in, dev_out;
 int d_out[5];

 printf("Prima dll'inversione");
 print_arrray(d_in, 5);

 //allocate device copy of d_in
 cudaMalloc( (void**)&dev_in, size );
 cudaMalloc( (void**)$dev_out, size);
 //copy input to device

 cudaMemcpy(dev_in, &d_in, size, cudaMemcpyHostToDevice );
 
 //launch reverse kernel
 reverse<<<1, 1>>>(dev_in, dev_out);

 //copy device result into host memory
 cudaMemcpy(dev_out, &d_out, size, cudaMemcpyDeviceToHost);
 cudaFree(dev_in);
 cudaFree(dev_out);
 
 printf("Dopo l'inversione");
 print_array(d_out);
 return 0; 
}

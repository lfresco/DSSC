#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define N 100

void init_vector(int * vec1, int * vec2,int * vec3, const int rank){
  int i;
  for(i = 0; i < N; i++){
    vec1[i] = rank;
    vec2[i] = 0;
    vec3[i] = 0;
  }

}

int main(int argv, char* argc[]){
  
  int rank = 0;
  int npes = 0;
  MPI_Request send_request;
  MPI_Status status;

  MPI_Init(&argv, &argc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  int * local_vec = (int*)malloc(N*sizeof(int)); // vector that contains local variable
  int * sum = (int*)malloc(N * sizeof(int));     // this is the vector that will contain the the final result
  int * recv = (int*)malloc(N*sizeof(int));      // vector that will contain the recieved message
  
  init_vector(local_vec, sum,recv,  rank);    // initialize the local vectors
  
  int i;
  for(i = 0; i < npes ; i++){
    if(rank != 0){
      MPI_Recv(recv, N, MPI_INT, (rank - 1 + npes)%npes, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //MPI_Send(sum, N, MPI_INT, (rank + 1)%npes, 101, MPI_COMM_WORLD);
      int j;
      for(j = 0;j < N; j++){
        sum[i] = recv[i] + local_vec[i];
      }
      MPI_Send(sum, N, MPI_INT, (rank + 1)%npes, 101, MPI_COMM_WORLD);
    } else {
      MPI_Send(sum, N, MPI_INT, (rank + 1)%npes, 101, MPI_COMM_WORLD);
      MPI_Recv(recv, N, MPI_INT, (rank - 1 + npes) % npes, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int j;
      for(j = 0; j < N; j++){
        sum[i] = recv[i];
      }
    }
  }

  fprintf(stderr, "Proccess %d, Riceve da %d, la somma è %d\n",rank, (rank-1+npes)%npes, sum[0]);
  MPI_Barrier(MPI_COMM_WORLD);
  free( local_vec);
  free( sum);
  free(recv);
  MPI_Finalize();
  return 0;
 
}

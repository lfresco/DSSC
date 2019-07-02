#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define N 100

void swap(int ** a, int ** b)
{
  int * tmp = *a;
  *a = *b;
  *b = tmp;
}

int main(int argc, char * argv[])
{
  int rank = 0;
  int npes = 1;

  MPI_Init(&argc, & argv);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank);
  MPI_Comm_size( MPI_COMM_WORLD, &npes);

  MPI_Request srequest, rrequest;
  MPI_Status status;

  unsigned int i, j;
  int final = npes - 1;

  int * message = (int*)malloc(sizeof(int)*N);
  int * recieve = (int*)malloc(sizeof(int)* N);
  int * sum = (int*)malloc(sizeof(int)*N);

  for(i = 0; i < N; i++){
    message[i] = rank;
    sum[i] = message[i];
  }
  for(i = 0; i < npes; i++)
  {

    MPI_Isend(message, N, MPI_INT, (rank + 1)%npes, 101, MPI_COMM_WORLD, &srequest);

    for(j = 0; j < N; ++j)
      sum[j] += message[j];

    MPI_Irecv(recieve, N, MPI_INT, (rank - 1 + npes)%npes, 101, MPI_COMM_WORLD, &rrequest);
    MPI_Wait(&srequest, &status);
    MPI_Wait(&rrequest, &status);
    swap(&recieve, &message);
  }

  fprintf(stderr, "I am process %d and my message is %d \n", rank, sum[0]);

  MPI_Barrier(MPI_COMM_WORLD);

  free( message );
  free( recieve );
  free( sum );

  MPI_Finalize();


  return 0;
}

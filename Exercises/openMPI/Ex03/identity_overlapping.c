#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define N 20

void swap(int** a, int** b)
{
  int * tmp = *a;
  *a = *b;
  *b = tmp;
}

void print_mat(int * mat, int n_block)
{
  unsigned int i, j;
  for(i = 0; i < n_block; i++){
    for(j = 0; j < N; j++)
      printf("%d ", mat[i * N + j]);
    printf("\n");
  }
}

void print_mat_file(int * mat, int n_block, FILE* outfile)
{
  unsigned int i, j;
  for(i = 0; i < n_block; i++){
    for(j = 0; j < N; j++)
      fprintf(outfile, "%d", mat[i * N + j]);
    fprintf(outfile, "\n");
  }
}

int main(int argc, char* argv[])
{
  int rank = 0;
  int npes = 1;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank);
  MPI_Comm_size( MPI_COMM_WORLD, &npes);

  int i, j, i_global;

  int * mat = (int *)malloc(sizeof(int)*N*N);

  int n_block = N/npes;
  int rest = N%npes;
  if(rest != 0 && rank < rest)
    n_block++;

  for(i = 0; i < n_block; i++){
    for(j = 0; j < N; j++){
      
      if(rank < rest) i_global = i + (rank*n_block);
      if(rank >= rest) i_global = i + (rank * n_block) + rest;

      if(i_global == i) mat[i *N + j] = 1;
      else mat[i*N + j] = 0;
    }
  }

  if(rank = 0)
  {
    MPI_Request request;
    MPI_STatus status;

    FILE*outfile;
    outfile = fopen("over_identity.txt", "w");

    int* write_buff = mat;
    int * recv_buf = (int*)malloc(N*n_block*sizeof(int));

    MPI_Irecv(recv_buf, n_block*N, MPI_INT, 1, 101, MPI_COMM_WORLD, &request);
    print_mat_file(write_buff, n_block, outfile);
    MPI_Wait(&request, &status);

    for(i = 1; i < npes - 1; i++){
      
      swap(&recv_buff, &write_buff);

      MPI_Irecv(&n_block, 1, MPI_INT, i + 1, 102, MPI_COMM_WORLD, &request);
      MPI_Irecv(recv_buff, n_block*N, MPI_INT, i + 1, 101, MPI_COMM_WORLD, &request);
      print_mat_file(write_buff, n_block, outfile);
      MPI_Wait(&request, &status);
    }

    if(rest == 0)
      print_mat_file(recv_buff, n_block, outfile);
    else
      print_mat_file(recv_buff, n_block-1, outfile);
  } else {
     MPI_Send(mat, n_block*N, MPI_INT, 0, 101, MPI_COMM_WORLD);
     MPI_Send(&n_block, 1, MPI_INT, 0, 102, MPI_COMM_WORLD);
  }

  free( mat );
  MPI_Finalize();:x

}

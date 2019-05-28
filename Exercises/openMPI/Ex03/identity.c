#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>



void print_matrix_stdout(int ** matrix, const int N_LOC, const int N){
  int i, j;
  for(i = 0; i < N_LOC; i++){
    for(j = 0; j < N - 1; j++){
      printf( "%d\t", matrix[i][j]);
    }
    printf( "%d\n", matrix[i][N - 1]);
  }
}

void print_matrix_file(int ** matrix, const int N_LOC, const int N, FILE * output){
  int i,j;
  for(i = 0; i < N_LOC; i++){
    for(j = 0; j < N - 1; j++){
      fprintf(output, "%d\t", matrix[i][j]);
    }
    fprintf(output,"%d\n", matrix[i][N - 1]);
  }
}

int main(int argc, char* argv[])
{
  const int N = 9;
  int npes = 1;
  int rank;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int N_LOC, start, REST;
  REST = N % npes;
  N_LOC = N/npes + ((int)rank < REST ? 1 : 0);
  start = N_LOC * rank + ((int)rank < REST ? 0 : REST);
  
  //Let's now allocate the local matrix

  int ** local_mat = (int**)malloc(N_LOC * sizeof(int*));
  int i,j;
  for(i = 0; i < N; i++)
    local_mat[i] = (int*)malloc(N * sizeof(int));
    
  // Now that we have the matrix we can initialize it with the needed numbers
  for(i = 0; i < N_LOC; i++){
    for(j = 0; j < N; j++){
      if(i + start == j)
        local_mat[i][j] = 1;
      else 
	local_mat[i][j] = 0;
    }
  }

  //Now we have to divide in two cases related to the dimension of our matrix
  //if N < 11 we must print the output on the stdout, otherwise we must use a file. 

  if(N < 11)
  {
    if(rank == 0){
      FILE* identity = fopen("identity.txt", "w"); 
      print_matrix_file(local_mat, N_LOC, N, identity);
    

     for(i = 1; i < npes; i++){
     // Since the number of lines will not be the same for every process I have to calculate the 
     // number of line the process 0 will recieve from the other processes

       int n_of_rows = N_LOC; // The root has 1 process more than the one without rest so 
        		     // we know have to remove one if the message is coming from a process
			     // whose rank is >= rest
       if(i >= REST && REST > 0)
         n_of_rows--;
      

       // now a for cicle in which we will recieve the messages coming from the other processes and 
       // print it
       for(j = 0; j < n_of_rows; j++)
         MPI_Recv(local_mat[j],  N, MPI_INT, i, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       
       print_matrix_file(local_mat, n_of_rows, N, identity);
     }
   }else {
     for(i = 0; i < N_LOC; i++){
       MPI_Send(local_mat[i], N, MPI_INT, 0, 101, MPI_COMM_WORLD);
       free(local_mat[i]);	   
     }
     free(local_mat);
   }
   
 } else {
     
     if(rank == 0){
       FILE * id_file = fopen("identity.txt", "w");
       print_matrix_file(local_mat, N_LOC, N, id_file);
       for(i = 1; i < npes; i++){
         int n_of_rows = N_LOC;
	 if(i >= REST && REST > 0)
           n_of_rows--;

	 for(j = 0; j < n_of_rows; j++)
           MPI_Recv(local_mat[j], N, MPI_INT, i, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         
	 print_matrix_file(local_mat, n_of_rows, N, id_file); 
       } 
     }else {
       for(i = 0; i < N_LOC; i++)
         MPI_Send(local_mat[i], N, MPI_INT, 0, 101, MPI_COMM_WORLD);   
     }

}

MPI_Finalize();

return 0;
}

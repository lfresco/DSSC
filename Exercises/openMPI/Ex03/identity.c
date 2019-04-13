#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


int main(int argc, char * argv[]){

 int rank = 0;
 int npes = 1;
 int N = 10;
 

 MPI_Init(&argc, &argv);
 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 MPI_Comm_size(MPI_COMM_WORLD, &npes);
 
 int N_LOC = N/npes;
 int **mat = (int**)malloc(sizeof(int*)* N_LOC);
 int i = 0;
 for(i; i < N_LOC; i++){
    mat[i] = (int*)malloc(sizeof(int)* 10);
 }
 int j; 
 for(i = 0; i < N_LOC; i++ ){
   for(j = 0; j < N; j++){
     if((i == 0 && j == 2*rank)|| (i == 1 && j == 2*rank +1)){
       mat[i][j]=1;
     } else {
       mat[i][j] = 0;
     }
   }
 }

 //for(i = 0; i < npes + 1; i++){
  if(rank == 0){
    int k;
    for(k = 0; k < N_LOC; k++){
      for(j = 0; j < N; j++){
        printf("%ld \t", mat[k][j]);
    }
     printf("\n");
  }
 }
//}
 MPI_Finalize();
 return 0;
}

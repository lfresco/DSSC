#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>
double f(double x){

	return 1.0/(1.0 + x*x);
}

double local_sum(double local_a, double local_b, int local_n, double h){

  unsigned int i;
  double local_result = 0.0;
  double x_i = 0.0;

  for(i = 0; i < local_n; ++i){
    
    x_i = local_a + i * h + h/2.0;
    local_result += f(x_i);
  }

  local_result = local_result * h * 4.0;
  return local_result;
}



int main(int argc, char * argv[]){
  
  clock_t start, end;
  double cpu_time_used;
  start = clock(); 
  int N = 2000000000;
  int rank  = 0; // stores the MPI identifier of the process
  int npes = 1;  // stores the number of processes
  double global_result;
  double a = 0.0;
  double b = 1.0;
  
  MPI_Init(&argc,&argv);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank);
  MPI_Comm_size( MPI_COMM_WORLD, &npes);
  
  double h = (b - a)*(1.0/N);
  int local_n = N/npes;
  double local_a = a + rank*local_n *h;
  double local_b = local_a + local_n * h;
  double local_result = local_sum(local_a, local_b, local_n, h);
  int final = npes - 1 ;
  
  MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, final, MPI_COMM_WORLD);
  end = clock();
  cpu_time_used = ((double)(end - start))/CLOCKS_PER_SEC;
 
  if(rank  == final)
    MPI_Send(&global_result, 1, MPI_DOUBLE, 0, 101,MPI_COMM_WORLD);
  if(rank == 0){
    
    MPI_Recv(&global_result, 1, MPI_DOUBLE, final, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    fprintf(stderr, "\n The approximation of pi is %lf \n in time: %lf", global_result, cpu_time_used);
  }
  
  MPI_Finalize();

  return 0;

}

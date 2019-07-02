/**
 * @author Lorenzo Fresco
 *
 * @brief This simple program implements a way to approximate the value of pi in a parallel way using openMP
 *
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

/**
 *@brief The function used in the approximation
 *
 *@param The value for which we want to compute the function value
 *
 */

double f(double x){

  double f_x = 1.0/(1.0 + x*x);
  return f_x;
}

/**
 *@brief This function is used inside the various threads to compute the local result as it is given 
 *       by a summatory
 *
 *@param local_a : is the beginning of the local interval
 *       local_b : the end point of the local interval
 *	 local_n : the number of subintervals in which we will divide the local interval
 *	 h : a costant equal to 1/N, where N is the number of points we want to use to compute the approximation
 *
 *
 */
double local_sum(double local_a, double local_b, int local_n, double h )
{
	unsigned int i;  // index of the for cicle
	double local_result = 0.0; // this variable will be used to keep track of the result of the summation during 
				   // the execution of the for cycle
	double x_i = 0.0;	
	
	for(i = 0; i < local_n; ++i){

	  x_i = local_a + i *h + h/2.0;  // This is the midpoint of each subinterval
	  local_result += f(x_i);        
	}
	
	local_result = local_result * h * 4;
	
	return local_result;
}


/**
 * @brief Function that performs the approximation using the critical keyword
 *
 * @param int N, problem size
 *
 *
 *
 */

double critical(int N)
{
  double pi_approx = 0.0;
  int b = 1;
  int a = 0;
  

  #pragma omp parallel 
  {
    double h = (b - a) * (1.0 / N);
    int thread_id = omp_get_thread_num();
    int n_threads = omp_get_num_threads();
    
    int local_n = N/n_threads;
    
    double local_a = a + thread_id * local_n * h;
    double local_b = local_a + local_n * h;
    
    double local_result = local_sum(local_a, local_b, local_n, h);
    
    #pragma omp critical
    pi_approx += local_result;

  }
  
  return pi_approx;
}

/**
 *@brief Computes parallel approximation of Pi using the reduction keyword
 *
 *@param int N, the problem size (number of sub intervals)
 *
 */

double reduction(int N)
{
  double pi_approx = 0.0;
  double b = 1.0;
  double a = 0.0;
  int n_threads;
#pragma omp parallel reduction(+:pi_approx)
{
  double h = (b - a) * (1.0 / N);
  int thread_id = omp_get_thread_num();
  n_threads = omp_get_num_threads();
  
  int local_n = N/n_threads;
  double local_a = a + thread_id * local_n * h;
  double local_b = local_a + local_n * h;

  double local_result = local_sum(local_a, local_b, local_n, h);
  
  pi_approx += local_result;

  
}
return pi_approx;
}


/**
 *@brief Computes parallel approximation of Pi by using the atomic keyword
 *
 *@param int N, the problem size (number of subintervals)
 *
 *
 */
double atomic(int N)
{
  double a = 0.0;
  double b = 1.0;
  int n_threads;
  double pi_approx = 0.0;

#pragma omp parallel
{
  double h = (b - a) * (1.0 / N);
  int thread_id = omp_get_thread_num();
  n_threads = omp_get_num_threads();
  int local_n = N / n_threads;
 
  double local_a = a + thread_id * local_n * h;
  double local_b = local_a + local_n * h;

  double local_result = local_sum(local_a, local_b, local_n, h);

#pragma omp atomic
  pi_approx += local_result;
}

return pi_approx;
}


int main(){

  int N = 1000000000;
  double pi_approx;
  
  double start = omp_get_wtime();
  pi_approx = critical(N);
  double end = omp_get_wtime();

  printf("Critical keyword\n");
  printf("Value of pi : %f \n", pi_approx);
  printf("Execution time : %f \n", end - start);
  printf("------------------------------\n");

  start = omp_get_wtime();
  pi_approx = atomic(N);
  end = omp_get_wtime();

  printf("Atomic ketyword\n");
  printf("Value of pi : %f \n", pi_approx);
  printf("Execution time : %f \n", end- start);
  printf("------------------------------\n");

  start = omp_get_wtime();
  pi_approx = reduction(N);
  end = omp_get_wtime();

  printf("Reduction keyword");
  printf("Value of Pi : %f \n", pi_approx);
  printf("Execution time : %f \n", end - start);
  printf("------------------------------\n");
  
  return 0;
}

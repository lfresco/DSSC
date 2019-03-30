
#!/bin/bash

cd myPARALLEL/DSSC/Exercises/openMP/APPROX/

module load intel

icc -qopenmp approximation.c

for threads in 1 2 4 8 16 20; do

 export OMP_NUM_THREADS=${threads}
 ./a.out >> approx_result.txt
done

exit


#!/bin/bash

cd ~/myPARALLEL/DSSC/Exercises/openMP/



gcc  -fopenmp loop_schedule.c -std=c99

export OMP_NUM_THREADS=10

./a.out

exit

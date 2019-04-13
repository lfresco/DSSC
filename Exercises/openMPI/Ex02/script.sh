
#!/bin/bash

cd myPARALLEL/DSSC/Lab/Day3/Day3/

module load openmpi

for i in 1 2 4 8 16 20 40; do
mpirun -np ${i} ./a.out >> result.txt
done 

exit

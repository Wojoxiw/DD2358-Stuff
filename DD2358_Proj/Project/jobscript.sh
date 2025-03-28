#!/bin/bash
# ### SBATCH # --qos=test  ## to run a quick, high-priority test (should be very quick). This is currently commented out (I hope)
#SBATCH -t 8:45:00 ## job is killed after this time - overestimate
#SBATCH -A lu2024-2-93 ##from projinfo command

#SBATCH -N 2 ##number of nodes that will be allocated - must use --ntasks-per-node or --cpus-per-task to use more than 1 core per node
#SBATCH --tasks-per-node=4 ##number of cores used per task? - up to 48 per node for COSMOS. Presumably this is what I want, using MPI

#SBATCH -o jobresults/%j.out ## result filename, %j becomes the job number
#SBATCH -e jobresults/%j.err ## errors filename - should be empty unless an error occurs
#SBATCH -J again ##puts a job name, to identify it

cat $0 ## unix command - outputs this script to the top of the job's output file
echo ## maybe makes a newline?
echo ## maybe makes a newline?
free -h ## print available memory, just to check
awk '/MemFree/ { printf "%.3f \n", $2/1024/1024 }' /proc/meminfo
echo ## maybe makes a newline?
echo "hello from $HOSTNAME:" $HOSTNAME ## some unix script. Not sure how to print newline...
echo "jobscript listed above, date listed below..."
date ## prints current date/time

## if not using node-local disk, just run it and hopefully this is fine
#time mpirun -n 1 python runScatt3D.py ## run the main process, and time it
time mpirun --bind-to core python runScatt3D.py
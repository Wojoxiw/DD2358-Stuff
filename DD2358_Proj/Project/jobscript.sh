#!/bin/bash
# ### SBATCH # --qos=test  ## to run a quick, high-priority test (should be very quick). This is currently commented out (I hope)
#SBATCH -t 30:45:00 ## job is killed after this time - overestimate
#SBATCH -A lu2024-2-93 ##from projinfo command

#SBATCH -N 2 ##number of nodes that will be allocated - must use --ntasks-per-node or --cpus-per-task to use more than 1 core per node
#SBATCH --tasks-per-node=12 ##number of cores used per task? - up to 48 per node for COSMOS. Presumably this is what I want, using MPI

#SBATCH -o jobresults/%j_result.out ## result filename, %j becomes the job number
#SBATCH -e jobresults/%j_result.err ## errors filename - should be empty unless an error occurs
#SBATCH -J MPI12-2nodes ##puts a job name, to identify it

cat $0 ## unix command - outputs this script to the top of the job's output file

echo "\n""\nhello from $HOSTNAME:" $HOSTNAME ## some unix script
echo "jobscript listed above, date listed below..."
date ## prints current date/time

: '
## NODE LOCAL DISK
## if using node-local disk, move files over to it before running, then back after
srun -n $SLURM_NNODES -N $SLURM_NNODES cp -p runScatt3D.py $SNIC_TMP ## reads this file into the node-local disks/execution directories. I first update it with git pull origin master
srun -n $SLURM_NNODES -N $SLURM_NNODES cp -p meshMaker.py $SNIC_TMP
srun -n $SLURM_NNODES -N $SLURM_NNODES cp -p memTimeEstimation.py $SNIC_TMP
srun -n $SLURM_NNODES -N $SLURM_NNODES cp -p scatteringProblem.py $SNIC_TMP
cd $SNIC_TMP ## go to that directory to make the data 3D folder so I can move my input file(s) there... then go back to move stuff.
mkdir data3D ## Presumably (hopefully) the rank 0 process is here, so all files will be saved in this directory
cd $SLURM_SUBMIT_DIR
cp -p data3D/prevRuns.npz $SNIC_TMP"/data3D"
cd $SNIC_TMP ## go to that directory to run the script

#time python Scatt3D.py ### then run it... and time it
export MPInum=12 ## number of MPI processes
time mpirun -n $MPInum python runScatt3D.py ## run the main process, and time it
#mpirun --bind-to core python runScatt3D.py

# cp -p prevRuns.info $SLURM_SUBMIT_DIR ## copies this file out to whatever directory you were in when using sbatch jobscript.sh
cp -rp data3D $SLURM_SUBMIT_DIR/ ## copy the data folder over also
## NODE LOCAL DISK
'

## if not using node-local disk, just run it and hopefully this does not slow things down much
time mpirun -n $MPInum python runScatt3D.py ## run the main process, and time it
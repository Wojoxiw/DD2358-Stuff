#!/bin/bash
# ### SBATCH # --qos=test  ## to run a quick, high-priority test (should be very quick)
#SBATCH -t 30:45:00 ## job is killed after this time - overestimate
#SBATCH -A lu2024-2-93 ##from projinfo command

#SBATCH -N 1 ##number of nodes that will be allocated - must use --ntasks-per-node or --cpus-per-task to use more than 1 core per node
#SBATCH --tasks-per-node=24 ##number of cores used per task? - up to 48 per node for COSMOS. Presumably this is what I want, using MPI

#SBATCH -o jobresults/%j_result.out ## result filename, %j becomes the job number
#SBATCH -e jobresults/%j_result.err ## errors filename - should be empty unless an error occurs
#SBATCH -J MPI1 ##puts a job name, to identify it

cat $0 ## unix command - outputs this script to the top of the job's output file

echo \n"hello from $HOSTNAME:" $HOSTNAME ## some unix script
echo "jobscript listed above, date listed below..."
date ## prints current date/time

cp -p runScatt3D.py $SNIC_TMP ## reads this file into the node-local disk/execution directory. I first update it with git pull origin master
cp -p meshMaker.py $SNIC_TMP
cp -p memTimeEstimation.py $SNIC_TMP
cp -p scatteringProblem.py $SNIC_TMP
cd $SNIC_TMP ## go to that directory to make the data 3D folder so I can move stuff there... then go back to move stuff
mkdir data3D
cd $SLURM_SUBMIT_DIR
cp -p data3D/prevRuns.prev $SNIC_TMP"/data3D"
cd $SNIC_TMP ## go to that directory to run the script

#time python Scatt3D.py ### then run it... and time it
export MPInum=1 ## number of MPI processes
time mpirun -n $MPInum python runScatt3D.py ## run the main process, and time it

# cp -p prevRuns.info $SLURM_SUBMIT_DIR ## copies this file out to whatever directory you were in when using sbatch jobscript.sh
cp -rp data3D $SLURM_SUBMIT_DIR/ ## copy the data folder over also
#!/bin/bash
#SBATCH --qos=test  ##since this is just testing if the Python script can even run
#SBATCH -t 00:05:00 ## job is killed after this time - overestimate
#SBATCH -A lu2024-2-93 ##from projinfo command

#SBATCH -N 1 ##number of nodes that will be allocated - must use --ntasks-per-node or --cpus-per-task to use more than 1 core per node
#SBATCH --tasks-per-node=1 ##number of cores used per task? - up to 48 per node for COSMOS. Presumably this is what I want, using MPI

#SBATCH -o result_filename_%j.out ## result filename, %j becomes the job number
#SBATCH -e result_filename_%j.err ## errors filename - should be empty unless an error occurs
#SBATCH -J TEST_01 ##puts a job name, to identify it

cat $0 ## unix command - outputs this script to the top of the job's output file

date ## prints current date/time

echo "hello from $HOSTNAME:" $HOSTNAME ## some unix script
cp -p Scatt3D.py $SNIC_TMP ## reads this file into the node-local disk/execution directory
cd $SNIC_TMP ## go to that directory
		 
time python Scatt3D.py ### then run it... and time it
cp -p result.dat $SLURM_SUBMIT_DIR ## copies this file out to the submission directory-whatever directory you were in when using sbatch jobscript.sh

## save this in .sh script, maybe make it using nano run_scipt.sh
#### find job with squeue -u alepal, or squeue --me
#read something? with more slurm-job#.out
##cancel job with scancel job#
## queue jobs sequentially with sbatch -d afterok:firstjobid run_script.sh
## to run one, sbatch run_scipt.sh

## COSMOS node local disk has ~1.6 TB SSD, default 5300MB RAM per core. can do #SBATCH --mem-per-cpu=10600

## variable SNIC_TMP addresses the node-local disk
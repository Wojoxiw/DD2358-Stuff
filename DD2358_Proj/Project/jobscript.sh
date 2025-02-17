#!/bin/bash
# ###SBATCH --qos=test  ## to run a quick, high-priority test (should be very quick)
#SBATCH -t 00:05:00 ## job is killed after this time - overestimate
#SBATCH -A lu2024-2-93 ##from projinfo command

#SBATCH -N 1 ##number of nodes that will be allocated - must use --ntasks-per-node or --cpus-per-task to use more than 1 core per node
#SBATCH --tasks-per-node=1 ##number of cores used per task? - up to 48 per node for COSMOS. Presumably this is what I want, using MPI

#SBATCH -o %j_result.out ## result filename, %j becomes the job number
#SBATCH -e %j_result.err ## errors filename - should be empty unless an error occurs
#SBATCH -J scatt-test ##puts a job name, to identify it

cat $0 ## unix command - outputs this script to the top of the job's output file

echo "  hello from $HOSTNAME:" $HOSTNAME ## some unix script
echo "jobscript listed above, date listed below..."
date ## prints current date/time
							
cp -p Scatt3D.py $SNIC_TMP ## reads this file into the node-local disk/execution directory. I first update it with git pull origin master
cp -p prevRuns.info $SNIC_TMP
cd $SNIC_TMP ## go to that directory

time python Scatt3D.py ### then run it... and time it

cp -p result.dat $SLURM_SUBMIT_DIR ## copies this file out to the submission directory-whatever directory you were in when using sbatch jobscript.sh
cp -p prevRuns.info $SLURM_SUBMIT_DIR ## should now be updated with whatever the runtime was
## then sync output folder with my local output folder, so I have the files locally... by running the following script locally, afterward
#rsync -a alepal@cosmos.lunarc.lu.se:/home/alepal/Alexandros/PreTestingStuff/FileTransferTest/ "/mnt/d/Microwave Imaging/LUNARC_file_transferring_folder"
#rsync -a alepal@cosmos.lunarc.lu.se:/home/alepal/Alexandros/PreTestingStuff/FileTransferTest/prevRuns.info "/mnt/d/Microwave Imaging/repository/DD2358 Proj/Project/prevRuns.info"

## login with ssh cosmos.lunarc.lu.se -l alepal

#### find job with squeue -u alepal, or squeue --me
#read something? with more slurm-job#.out
##cancel job with scancel job#
## queue jobs sequentially with sbatch -d afterok:firstjobid run_script.sh

## COSMOS node local disk has ~1.6 TB SSD, default 5300MB RAM per core. can do #SBATCH --mem-per-cpu=10600
## variable SNIC_TMP addresses the node-local disk

## to run one, sbatch jobscript.sh
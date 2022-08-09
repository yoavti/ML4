#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 3-10:30:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name ml4			### name of the job
#SBATCH --output /cs_storage/yoavti/log/ml4_%A_%a.out		### output log for running job - %J for job number
#SBATCH --gpus=1				### number of GPUs, allocating more than 1 requires IT team's permission

# Note: the following 4 lines are commented out
#SBATCH --mail-user=yoavti@post.bgu.ac.il	### user's email for sending job status messages
#SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
##SBATCH --mem=32G				### ammount of RAM memory, allocating more than 60G requires IT team's permission
##SBATCH --cpus-per-task=6			### number of CPU cores, allocating more than 10G requires IT team's permission

#SBATCH -a 1-19

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

LIST=($(<missing.csv))
ROW=${LIST[${SLURM_ARRAY_TASK_ID}]}
DATASET=$(echo "${ROW}" | cut -d"," -f1)
K=$(echo "${ROW}" | cut -d"," -f2)

echo "$DATASET"
echo "$K"

### Start your code below ####
#module load anaconda				### load anaconda module (must be present when working with conda environments)
#source activate ML4
#python run_experiments.py $DATASET -k 10
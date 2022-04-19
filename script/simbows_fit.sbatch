#!/bin/bash

#SBATCH --partition=broadwl
#SBATCH --ntasks-per-node=8
#SBATCH --mem=16G
#SBATCH --time=36:00:00

module load R/3.6.1

#source activate base

DAT=${1}
METHOD=${2}
K=${3}
OUTFILE=${4}

echo dat=${DAT}
echo method=${METHOD}
echo k=${K}
echo outfile=${OUTFILE}


Rscript simbows_fit.R ${DAT} ${METHOD} ${K} > ${OUTFILE} 

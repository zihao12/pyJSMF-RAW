#!/bin/bash

#SBATCH --partition=broadwl
#SBATCH --ntasks-per-node=8
#SBATCH --mem=16G
#SBATCH --time=36:00:00


export PATH="/home/wangzh/miniconda3/bin:$PATH"

DAT=${1}
K=${2}

echo dat=${DAT}
echo k=${K}


python3 simulate_bows.py ${DAT} ${K}  

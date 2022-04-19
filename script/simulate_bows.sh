#!/bin/bash

SCRIPT_FIT=simulate_bows.sbatch
dataname=nips

for k in 5 10 15 20 25 30 50 70 100
do
  sbatch ${SCRIPT_FIT} ${dataname} ${k} 
done


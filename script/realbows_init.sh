#!/bin/bash

SCRIPT_FIT=realbows_init.py
dataname=nips

for method in pn rec1_pn rec2_pn rec12_pn rec vanila nndsvd random
do
  for k in 5 10 15 20 25 30 50 70 100
  do
    output=realbows_init_${dataname}_${method}_${k}.out
    sbatch realbows_init.sbatch ${dataname} ${method} ${k} ${output}
  done
done

#!/bin/bash

SCRIPT_FIT=simbows_fit.sbatch
dataname=nips

for method in pn rec1_pn rec2_pn rec12_pn rec vanila nndsvd random
do
  for k in 5 10 15 20 25 30 50 70 100
  do
    output=out/simbows_fit_${dataname}_${method}_${k}.out
    sbatch ${SCRIPT_FIT} ${dataname} ${method} ${k} ${output}
  done
done

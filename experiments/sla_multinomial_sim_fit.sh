#!/bin/bash
SCRIPT_FIT=sla_multinomial_sim_fit.py
dataname=sla
k=6
inputname=../../ebpmf_data_analysis/output/fastTopics_fit/fit_${dataname}_fastTopics_k${k}.Rds

for rate in 3 1 0.5
do 
	outputname=output/fit_sim_${dataname}_fastTopics_k${k}_rate${rate}.pkl
	python ${SCRIPT_FIT} -d ${inputname} -o ${outputname} -r ${rate} > ${outputname}.out
done

# dataname=droplet
# k=6
# inputname=../../ebpmf_data_analysis/output/fastTopics_fit/fit_${dataname}_fastTopics_k${k}.Rds


# for rate in 3 1 0.5
# do 
# 	outputname=output/fit_sim_${dataname}_fastTopics_k${k}_rate${rate}.pkl
# 	python ${SCRIPT_FIT} -d ${inputname} -o ${outputname} -r ${rate} > ${outputname}.out
# done
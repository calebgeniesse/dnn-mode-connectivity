#!/bin/bash

JOB_ID=$1
echo ${JOB_ID}

# cat logs/*${JOB_ID}*.out | grep "_curve.npz" | awk '{print $2}' | xargs -n1 dirname | xargs -n1 basename > mc_${JOB_ID}_files.txt
cat logs/*${JOB_ID}*.out | grep "_curve.npz" | sed s'|checkpoints\/|\t|g' | awk '{print $3}' > mc_${JOB_ID}_files.txt

# cat logs/*${JOB_ID}*.out | grep "Mode Connectivity" | awk '{print $3}' > mc_${JOB_ID}.txt
# cat logs/*${JOB_ID}*.out | grep "Mode Connectivity" | awk '{print $3}' | xargs -n1 printf "% 3.6f\n" > mc_${JOB_ID}.txt
cat logs/*${JOB_ID}*.out | grep "Mode Connectivity" | awk '{print $3}' | xargs -n1 printf "% 3.3f\n" > mc_${JOB_ID}.txt



paste mc_${JOB_ID}.txt mc_${JOB_ID}_files.txt

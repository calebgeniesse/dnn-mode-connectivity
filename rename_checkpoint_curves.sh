#!/bin/bash


ls checkpoints/*/checkpoint-*_curve.npz | while read F1 ; do
        
    D1=$(basename $(dirname $F1))
    F2=checkpoints/mode_connectivity/${D1}_$(basename $F1)
    
    # echo ${F2}
    cp ${F1} ${F2}
    echo "[+]" ${F2}
  
done
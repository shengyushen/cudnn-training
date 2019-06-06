#!/bin/bash
if [ $# -ne 7 ] ; 
  then echo "Usage : prof.sh <network type> <repeat time> <batch size> <resolution> <> <> <>"
else 
  rm -rf ssy_$1_$2_$3_$4_$5_$6_$7*
  mkdir ssy_$1_$2_$3_$4_$5_$6_$7.out
  nvprof -f --quiet --profile-api-trace none --print-gpu-trace --print-nvlink-topology --csv --log-file ssy_$1_$2_$3_$4_$5_$6_$7.csv -o ssy_$1_$2_$3_$4_$5_$6_$7.nvvp  ./cudnnModelParallel.exe "$@" 
fi



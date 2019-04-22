nvprof --print-gpu-trace --print-nvlink-topology -o %h-%p.nvvp --csv --log-file %h-%p.log "$@"

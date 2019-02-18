#!/usr/bin/gnuplot -p -e
set logscale y
set xlabel "Resolution"
set ylabel "Run time degrad ratio"
set key top left
set yrange [1:15]
plot "sss4" u 4:($2==16 ?$6:1/0) with linesp title "4 GPUs bs=16 ",\
         "" u 4:($2==32 ?$6:1/0) with linesp title "4 GPUs bs=32 ",\
         "" u 4:($2==64 ?$6:1/0) with linesp title "4 GPUs bs=64 ",\
         "" u 4:($2==128?$6:1/0) with linesp title "4 GPUs bs=128",\
     "sss8" u 4:($2==16 ?$6:1/0) with linesp title "8 GPUs bs=16 ",\
         "" u 4:($2==32 ?$6:1/0) with linesp title "8 GPUs bs=32 ",\
         "" u 4:($2==64 ?$6:1/0) with linesp title "8 GPUs bs=64 ",\
         "" u 4:($2==128?$6:1/0) with linesp title "8 GPUs bs=128"


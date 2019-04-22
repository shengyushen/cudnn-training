#!/usr/bin/gnuplot -p -e
set key top left
set logscale y
set xlabel "Resolution"
set ylabel "Average bandwidth(GBps)"
plot "sss4" u 4:($2==16 ?$8:1/0) with linesp title "4 GPUs bs=16 ",\
         "" u 4:($2==32 ?$8:1/0) with linesp title "4 GPUs bs=32 ",\
         "" u 4:($2==64 ?$8:1/0) with linesp title "4 GPUs bs=64 ",\
         "" u 4:($2==128?$8:1/0) with linesp title "4 GPUs bs=128",\
     "sss8" u 4:($2==16 ?$8:1/0) with linesp title "8 GPUs bs=16 ",\
         "" u 4:($2==32 ?$8:1/0) with linesp title "8 GPUs bs=32 ",\
         "" u 4:($2==64 ?$8:1/0) with linesp title "8 GPUs bs=64 ",\
         "" u 4:($2==128?$8:1/0) with linesp title "8 GPUs bs=128",\
"sss16_dgx2" u 4:($2==16 ?$8:1/0) with linesp title "16 GPUs bs=16 ",\
          "" u 4:($2==32 ?$8:1/0) with linesp title "16 GPUs bs=32 ",\
          "" u 4:($2==64 ?$8:1/0) with linesp title "16 GPUs bs=64 ",\
          "" u 4:($2==128?$8:1/0) with linesp title "16 GPUs bs=128"

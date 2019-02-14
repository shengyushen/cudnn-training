#!/usr/bin/gnuplot -p -e
set logscale x
set logscale y
set xlabel "fraction of feature synchronization"
set ylabel "Run time per iteration(ms)"
set title "Run time per iteration(ms)"
plot "res" u ($6==0.000000?0.00001:$6):($4==50 ?$8:1/0) with linesp title "4 GPUs 50 pixels",\
        "" u ($6==0.000000?0.00001:$6):($4==100?$8:1/0) with linesp title "4 GPUs 100 pixels",\
        "" u ($6==0.000000?0.00001:$6):($4==200?$8:1/0) with linesp title "4 GPUs 200 pixels",\
        "" u ($6==0.000000?0.00001:$6):($4==400?$8:1/0) with linesp title "4 GPUs 400 pixels",\
        "" u ($6==0.000000?0.00001:$6):($4==800?$8:1/0) with linesp title "4 GPUs 800 pixels",\
        "" u ($6==0.000000?0.00001:$6):($4==50 ?$8:1/0):(sprintf("%f",$8)) with labels notitle,\
        "" u ($6==0.000000?0.00001:$6):($4==100?$8:1/0):(sprintf("%f",$8)) with labels notitle,\
        "" u ($6==0.000000?0.00001:$6):($4==200?$8:1/0):(sprintf("%f",$8)) with labels notitle,\
        "" u ($6==0.000000?0.00001:$6):($4==400?$8:1/0):(sprintf("%f",$8)) with labels notitle,\
        "" u ($6==0.000000?0.00001:$6):($4==800?$8:1/0):(sprintf("%f",$8)) with labels notitle,\



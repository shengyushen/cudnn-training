#grep Ite lenet.txt > res
export DISPLAY=:0.0
gnuplot -p -e 'set xlabel "percentage";set ylabel "run time(ms)";set key bottom right;set logscale x;set logscale y;plot "res" u 6:($4==50?$8:1/0) w linesp title "width 50", "" u 6:($4==100?$8:1/0) w linesp  title "width 100",  "" u 6:($4==200?$8:1/0) w linesp title "width 200", "" u 6:($4==400?$8:1/0) w linesp title "width 400", "" u 6:($4==800?$8:1/0) w linesp title "width 800"'

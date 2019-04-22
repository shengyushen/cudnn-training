#!/usr/bin/gnuplot -p -e
set xlabel "resolution"
set ylabel "Performance degrad ratio"
plot "memSwap2.txt"	u 3:($1==1 && $2==16 ?$6:1/0) w linesp title "GPU 1 batch 16" ,\
								""	u 3:($1==1 && $2==32 ?$6:1/0) w linesp title "GPU 1 batch 32" ,\
								""	u 3:($1==1 && $2==64 ?$6:1/0) w linesp title "GPU 1 batch 64" ,\
								""	u 3:($1==1 && $2==128?$6:1/0) w linesp title "GPU 1 batch 128" ,\
								""	u 3:($1==2 && $2==16 ?$6:1/0) w linesp title "GPU 2 batch 16" ,\
								""	u 3:($1==2 && $2==32 ?$6:1/0) w linesp title "GPU 2 batch 32" ,\
								""	u 3:($1==2 && $2==64 ?$6:1/0) w linesp title "GPU 2 batch 64" ,\
								""	u 3:($1==2 && $2==128?$6:1/0) w linesp title "GPU 2 batch 128" ,\
								""	u 3:($1==4 && $2==16 ?$6:1/0) w linesp title "GPU 4 batch 16" ,\
								""	u 3:($1==4 && $2==32 ?$6:1/0) w linesp title "GPU 4 batch 32" ,\
								""	u 3:($1==4 && $2==64 ?$6:1/0) w linesp title "GPU 4 batch 64" ,\
								""	u 3:($1==4 && $2==128?$6:1/0) w linesp title "GPU 4 batch 128" ,\

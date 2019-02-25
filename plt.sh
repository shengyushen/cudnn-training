#grep Ite lenet.txt > res
#export DISPLAY=localhost:10.0
#export DISPLAY=:0.0
#gnuplot -p -e 'set xlabel "percentage";set ylabel "run time(ms)";set key bottom right;set logscale x;set logscale y;plot "res" u 6:($4==50?$8:1/0) w linesp title "width 50", "" u 6:($4==100?$8:1/0) w linesp  title "width 100",  "" u 6:($4==200?$8:1/0) w linesp title "width 200", "" u 6:($4==400?$8:1/0) w linesp title "width 400", "" u 6:($4==800?$8:1/0) w linesp title "width 800"'
#rm -f res
#touch res


#grep Ite lenet.txt_hwfast4 |awk '{if($6==0.000000) {cur=$4;v=$8} else if($6 ==1.000000) {if($4==cur) {print "4 " $4 " " $8/v}}}' >> res
#gnuplot -p -e 'set title "Performance downgrade ratio due to feature synchronization";set yrange [0:2] ;set xlabel "resolution";set ylabel "Downgrade ratio";plot "res" u 2:($1==4?$3:1/0) with linesp title "4 GPUs"'



grep Ite resnet_huawei_4gpu > ss4
cat ss4|awk '{if($8==0) {bs=$4;w=$6;frat=$8;time=$12} else if($8==1) {if(bs==$4 && w==$6) {print "bs " $4 " w " $6 " ratio " $12/time " bandwidth " $10*1000/(($12-time)*1024*1024*1024) " GB"}}}'> sss4

grep Ite resnet_huawei_4gpu_nohint > ss4_nohint
cat ss4_nohint|awk '{if($8==0) {bs=$4;w=$6;frat=$8;time=$12} else if($8==1) {if(bs==$4 && w==$6) {print "bs " $4 " w " $6 " ratio " $12/time " bandwidth " $10*1000/(($12-time)*1024*1024*1024) " GB"}}}'> sss4_nohint

grep Ite resnet_huawei_8gpu > ss8
cat ss8|awk '{if($8==0) {bs=$4;w=$6;frat=$8;time=$12} else if($8==1) {if(bs==$4 && w==$6) {print "bs " $4 " w " $6 " ratio " $12/time " bandwidth " $10*1000/(($12-time)*1024*1024*1024) " GB"}}}'> sss8
./pltrt.plt
./pltbw.plt



#!/usr/bin/gnuplot -p -ea
set logscale x

set ytics 
set ylabel 'number of packet' tc lt 1
set autoscale y

set y2tics 
set y2label 'total packet size(GB)' tc lt 2
set autoscale y2

plot "tt.sznum" u 1:2 w linesp linetype 1 axes x1y1 title "packet number" , "" u 1:($3/1000/1000/1000)  linetype 2 axes x1y2 w linesp title "Packet size(GB)", "" u 1:($3/$4/1000/1000/1000) axes x1y2 w linesp title "Bandwidth(GBps)"

cat memSwap.txt |grep Ite|grep -v "advise 0"|sort -k4,4 -k6,6 -n -k8,8 |awk '{if($4==last4 && $6==last6 && $8==last8 && $10==1 && last10==0 ) {print $4 " " $6 " " $8 " " $12*1000/($16-last16)/1024/1024/1024 " GBps  " $16/last16};last4=$4;last6=$6;last8=$8;last10=$10;last16=$16 }'> memSwap2.txt
./memSwap.plt


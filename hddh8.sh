grep "PtoP" $1 > ss3

awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(0\).*Tesla V100-SXM2-16GB \(0\).*Tesla V100-SXM2-16GB \(0\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_00
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(0\).*Tesla V100-SXM2-16GB \(0\).*Tesla V100-SXM2-16GB \(1\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_01
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(0\).*Tesla V100-SXM2-16GB \(0\).*Tesla V100-SXM2-16GB \(2\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_02
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(0\).*Tesla V100-SXM2-16GB \(0\).*Tesla V100-SXM2-16GB \(3\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_03
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(0\).*Tesla V100-SXM2-16GB \(0\).*Tesla V100-SXM2-16GB \(4\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_04
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(0\).*Tesla V100-SXM2-16GB \(0\).*Tesla V100-SXM2-16GB \(5\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_05
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(0\).*Tesla V100-SXM2-16GB \(0\).*Tesla V100-SXM2-16GB \(6\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_06
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(0\).*Tesla V100-SXM2-16GB \(0\).*Tesla V100-SXM2-16GB \(7\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_07
                                           
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(1\).*Tesla V100-SXM2-16GB \(1\).*Tesla V100-SXM2-16GB \(0\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_10
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(1\).*Tesla V100-SXM2-16GB \(1\).*Tesla V100-SXM2-16GB \(1\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_11
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(1\).*Tesla V100-SXM2-16GB \(1\).*Tesla V100-SXM2-16GB \(2\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_12
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(1\).*Tesla V100-SXM2-16GB \(1\).*Tesla V100-SXM2-16GB \(3\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_13
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(1\).*Tesla V100-SXM2-16GB \(1\).*Tesla V100-SXM2-16GB \(4\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_14
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(1\).*Tesla V100-SXM2-16GB \(1\).*Tesla V100-SXM2-16GB \(5\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_15
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(1\).*Tesla V100-SXM2-16GB \(1\).*Tesla V100-SXM2-16GB \(6\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_16
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(1\).*Tesla V100-SXM2-16GB \(1\).*Tesla V100-SXM2-16GB \(7\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_17
                                           
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(2\).*Tesla V100-SXM2-16GB \(2\).*Tesla V100-SXM2-16GB \(0\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_20
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(2\).*Tesla V100-SXM2-16GB \(2\).*Tesla V100-SXM2-16GB \(1\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_21
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(2\).*Tesla V100-SXM2-16GB \(2\).*Tesla V100-SXM2-16GB \(2\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_22
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(2\).*Tesla V100-SXM2-16GB \(2\).*Tesla V100-SXM2-16GB \(3\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_23
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(2\).*Tesla V100-SXM2-16GB \(2\).*Tesla V100-SXM2-16GB \(4\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_24
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(2\).*Tesla V100-SXM2-16GB \(2\).*Tesla V100-SXM2-16GB \(5\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_25
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(2\).*Tesla V100-SXM2-16GB \(2\).*Tesla V100-SXM2-16GB \(6\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_26
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(2\).*Tesla V100-SXM2-16GB \(2\).*Tesla V100-SXM2-16GB \(7\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_27
                                           
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(3\).*Tesla V100-SXM2-16GB \(3\).*Tesla V100-SXM2-16GB \(0\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_30
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(3\).*Tesla V100-SXM2-16GB \(3\).*Tesla V100-SXM2-16GB \(1\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_31
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(3\).*Tesla V100-SXM2-16GB \(3\).*Tesla V100-SXM2-16GB \(2\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_32
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(3\).*Tesla V100-SXM2-16GB \(3\).*Tesla V100-SXM2-16GB \(3\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_33
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(3\).*Tesla V100-SXM2-16GB \(3\).*Tesla V100-SXM2-16GB \(4\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_34
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(3\).*Tesla V100-SXM2-16GB \(3\).*Tesla V100-SXM2-16GB \(5\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_35
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(3\).*Tesla V100-SXM2-16GB \(3\).*Tesla V100-SXM2-16GB \(6\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_36
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(3\).*Tesla V100-SXM2-16GB \(3\).*Tesla V100-SXM2-16GB \(7\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_37
                                                                       
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(4\).*Tesla V100-SXM2-16GB \(4\).*Tesla V100-SXM2-16GB \(0\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_40
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(4\).*Tesla V100-SXM2-16GB \(4\).*Tesla V100-SXM2-16GB \(1\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_41
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(4\).*Tesla V100-SXM2-16GB \(4\).*Tesla V100-SXM2-16GB \(2\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_42
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(4\).*Tesla V100-SXM2-16GB \(4\).*Tesla V100-SXM2-16GB \(3\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_43
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(4\).*Tesla V100-SXM2-16GB \(4\).*Tesla V100-SXM2-16GB \(4\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_44
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(4\).*Tesla V100-SXM2-16GB \(4\).*Tesla V100-SXM2-16GB \(5\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_45
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(4\).*Tesla V100-SXM2-16GB \(4\).*Tesla V100-SXM2-16GB \(6\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_46
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(4\).*Tesla V100-SXM2-16GB \(4\).*Tesla V100-SXM2-16GB \(7\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_47
                                                                       
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(5\).*Tesla V100-SXM2-16GB \(5\).*Tesla V100-SXM2-16GB \(0\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_50
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(5\).*Tesla V100-SXM2-16GB \(5\).*Tesla V100-SXM2-16GB \(1\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_51
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(5\).*Tesla V100-SXM2-16GB \(5\).*Tesla V100-SXM2-16GB \(2\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_52
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(5\).*Tesla V100-SXM2-16GB \(5\).*Tesla V100-SXM2-16GB \(3\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_53
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(5\).*Tesla V100-SXM2-16GB \(5\).*Tesla V100-SXM2-16GB \(4\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_54
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(5\).*Tesla V100-SXM2-16GB \(5\).*Tesla V100-SXM2-16GB \(5\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_55
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(5\).*Tesla V100-SXM2-16GB \(5\).*Tesla V100-SXM2-16GB \(6\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_56
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(5\).*Tesla V100-SXM2-16GB \(5\).*Tesla V100-SXM2-16GB \(7\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_57
                                                                       
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(6\).*Tesla V100-SXM2-16GB \(6\).*Tesla V100-SXM2-16GB \(0\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_60
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(6\).*Tesla V100-SXM2-16GB \(6\).*Tesla V100-SXM2-16GB \(1\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_61
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(6\).*Tesla V100-SXM2-16GB \(6\).*Tesla V100-SXM2-16GB \(2\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_62
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(6\).*Tesla V100-SXM2-16GB \(6\).*Tesla V100-SXM2-16GB \(3\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_63
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(6\).*Tesla V100-SXM2-16GB \(6\).*Tesla V100-SXM2-16GB \(4\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_64
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(6\).*Tesla V100-SXM2-16GB \(6\).*Tesla V100-SXM2-16GB \(5\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_65
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(6\).*Tesla V100-SXM2-16GB \(6\).*Tesla V100-SXM2-16GB \(6\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_66
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(6\).*Tesla V100-SXM2-16GB \(6\).*Tesla V100-SXM2-16GB \(7\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_67
                                                                       
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(7\).*Tesla V100-SXM2-16GB \(7\).*Tesla V100-SXM2-16GB \(0\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_70
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(7\).*Tesla V100-SXM2-16GB \(7\).*Tesla V100-SXM2-16GB \(1\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_71
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(7\).*Tesla V100-SXM2-16GB \(7\).*Tesla V100-SXM2-16GB \(2\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_72
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(7\).*Tesla V100-SXM2-16GB \(7\).*Tesla V100-SXM2-16GB \(3\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_73
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(7\).*Tesla V100-SXM2-16GB \(7\).*Tesla V100-SXM2-16GB \(4\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_74
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(7\).*Tesla V100-SXM2-16GB \(7\).*Tesla V100-SXM2-16GB \(5\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_75
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(7\).*Tesla V100-SXM2-16GB \(7\).*Tesla V100-SXM2-16GB \(6\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_76
awk -F, '{if($0 ~ /Tesla V100-SXM2-16GB \(7\).*Tesla V100-SXM2-16GB \(7\).*Tesla V100-SXM2-16GB \(7\)/) print $1 " " $2 " " $12 " " $13}'  ss3 > ss3_77

$(dirname "$0")/hddh8_part2.sh |awk '{if(NF==3) {print $1 " " $3 " 0"} else if(NF==0) {print ""} else {print $1 " " $3 " " $4}}' > hddh8
gnuplot -p -e 'set view 60,150,1,1;set pm3d at b;splot "hddh8" u 1:2:3 with lines '


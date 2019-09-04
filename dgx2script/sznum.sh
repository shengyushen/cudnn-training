rm -f $1.sznum
touch $1.sznum
awk -v start=0         -v end=4096       '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "4096       " num " " sz " " time}' $1 >> $1.sznum
awk -v start=4096      -v end=8192       '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "8192       " num " " sz " " time}' $1 >> $1.sznum
awk -v start=8192      -v end=16384      '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "16384      " num " " sz " " time}' $1 >> $1.sznum
awk -v start=16384     -v end=32768      '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "32768      " num " " sz " " time}' $1 >> $1.sznum
awk -v start=32768     -v end=65536      '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "65536      " num " " sz " " time}' $1 >> $1.sznum
awk -v start=65536     -v end=131072     '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "131072     " num " " sz " " time}' $1 >> $1.sznum
awk -v start=131072    -v end=262144     '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "262144     " num " " sz " " time}' $1 >> $1.sznum
awk -v start=262144    -v end=524288     '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "524288     " num " " sz " " time}' $1 >> $1.sznum
awk -v start=524288    -v end=1048576    '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "1048576    " num " " sz " " time}' $1 >> $1.sznum
awk -v start=1048576   -v end=2097152    '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "2097152    " num " " sz " " time}' $1 >> $1.sznum
awk -v start=2097152   -v end=4194304    '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "4194304    " num " " sz " " time}' $1 >> $1.sznum
awk -v start=4194304   -v end=8388608    '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "8388608    " num " " sz " " time}' $1 >> $1.sznum
awk -v start=8388608   -v end=16777216   '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "16777216   " num " " sz " " time}' $1 >> $1.sznum
awk -v start=16777216  -v end=33554432   '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "33554432   " num " " sz " " time}' $1 >> $1.sznum
awk -v start=33554432  -v end=67108864   '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "67108864   " num " " sz " " time}' $1 >> $1.sznum
awk -v start=67108864  -v end=134217728  '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "134217728  " num " " sz " " time}' $1 >> $1.sznum
awk -v start=134217728 -v end=268435456  '{if($4>=start && $4<end) {num=num+1;sz=sz+$4;time=time+$6;}} END{ print "268435456  " num " " sz " " time}' $1 >> $1.sznum

awk '{if(NF>1) print}' $1.sznum


#cat $@ |grep op_generic_tensor_kernel | awk -F, '{x16=substr($16,2,length($16)-2);split(x16,arr16," ");y16=substr(arr16[3],2,length(arr16[3])-2);print $1 " 1 " y16 " " }' > xx
cat $@ |awk -F, '{x16=substr($16,2,length($16)-2);split(x16,arr16," ");y16=substr(arr16[3],2,length(arr16[3])-2);print $1 " 1 " y16 " " }' > xx

grep DtoD  $1|awk -F, '{print $1 " " substr($16,24,length($16)-24) " " substr($17,24,length($17)-25) " " $20 " " $21 " " $2}' 

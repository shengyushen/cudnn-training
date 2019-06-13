 awk '{arr[$2][$3]=arr[$2][$3]+$4} END{for(x in arr) {for(y in arr[x]) {print x " " y " " arr[x][y]}}}'

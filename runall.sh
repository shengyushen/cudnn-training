./cudnnModelParallel.exe  50  100 16 256 0 1
./cudnnModelParallel.exe  50  100 16 256 1 0.0001
./cudnnModelParallel.exe  50  100 16 256 1 0.001
./cudnnModelParallel.exe  50  100 16 256 1 0.01
./cudnnModelParallel.exe  50  100 16 256 1 0.03
./cudnnModelParallel.exe  50  100 16 256 1 0.1
./cudnnModelParallel.exe  50  100 16 256 1 0.3
./cudnnModelParallel.exe  50  100 16 256 1 1

./cudnnModelParallel.exe  100 100 16 256 0 1
./cudnnModelParallel.exe  100 100 16 256 1 0.0001
./cudnnModelParallel.exe  100 100 16 256 1 0.001
./cudnnModelParallel.exe  100 100 16 256 1 0.01
./cudnnModelParallel.exe  100 100 16 256 1 0.03
./cudnnModelParallel.exe  100 100 16 256 1 0.1
./cudnnModelParallel.exe  100 100 16 256 1 0.3
./cudnnModelParallel.exe  100 100 16 256 1 1

./cudnnModelParallel.exe  200 100 16 256 0 1
./cudnnModelParallel.exe  200 100 16 256 1 0.0001
./cudnnModelParallel.exe  200 100 16 256 1 0.001
./cudnnModelParallel.exe  200 100 16 256 1 0.01
./cudnnModelParallel.exe  200 100 16 256 1 0.03
./cudnnModelParallel.exe  200 100 16 256 1 0.1
./cudnnModelParallel.exe  200 100 16 256 1 0.3
./cudnnModelParallel.exe  200 100 16 256 1 1

./cudnnModelParallel.exe  400 100 16 256 0 1
./cudnnModelParallel.exe  400 100 16 256 1 0.0001
./cudnnModelParallel.exe  400 100 16 256 1 0.001
./cudnnModelParallel.exe  400 100 16 256 1 0.01
./cudnnModelParallel.exe  400 100 16 256 1 0.03
./cudnnModelParallel.exe  400 100 16 256 1 0.1
./cudnnModelParallel.exe  400 100 16 256 1 0.3
./cudnnModelParallel.exe  400 100 16 256 1 1

# cudnn dont allow tensor>2GB, while 800 exceed that
./cudnnModelParallel.exe  700 100 16 256 0 1
./cudnnModelParallel.exe  700 100 16 256 1 0.0001
./cudnnModelParallel.exe  700 100 16 256 1 0.001
./cudnnModelParallel.exe  700 100 16 256 1 0.01
./cudnnModelParallel.exe  700 100 16 256 1 0.03
./cudnnModelParallel.exe  700 100 16 256 1 0.1
./cudnnModelParallel.exe  700 100 16 256 1 0.3
./cudnnModelParallel.exe  700 100 16 256 1 1



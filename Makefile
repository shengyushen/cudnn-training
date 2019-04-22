CC = nvcc
ARCH = sm_70
EXE = cudnnModelParallel.exe memSwap.exe
LDFLAG = -lcudnn -lcublas -g -G
.SUFFIXES: .cu .exe

all : $(EXE)

.cu.exe:
		$(CC) --std c++11 $(LDFLAG) -arch $(ARCH) $^ -o $@

clean:
	rm -f *.o *.exe


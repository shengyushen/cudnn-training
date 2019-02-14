CC = nvcc
ARCH = sm_70
EXE = cudnnModelParallel.exe
SRC = cudnnModelParallel.cu
OBJ = $(SRC:.cu=.o)
LDFLAG = -lcudnn -lcublas

all : $(EXE)

$(EXE) : $(SRC)
		$(CC) $(LDFLAG) -arch $(ARCH) $^ -o $@

clean:
	rm -f *.o *.exe


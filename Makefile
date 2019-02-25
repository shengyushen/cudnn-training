CC = nvcc
ARCH = sm_70
EXE = cudnnModelParallel.exe
SRC = cudnnModelParallel.cu
OBJ = $(SRC:.cu=.o)
LDFLAG = -lcudnn -lcublas -g -G

all : $(EXE)

$(EXE) : $(SRC)
		$(CC) --std c++11 $(LDFLAG) -arch $(ARCH) $^ -o $@

clean:
	rm -f *.o *.exe


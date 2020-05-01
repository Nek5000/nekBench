ifndef OCCA_DIR
ERROR:
	@echo "Error, environment variable [OCCA_DIR] is not set"
endif

include ${OCCA_DIR}/scripts/Makefile

# define variables
OGSDIR  = ./include
GSDIR  = ../../3rdParty/gslib

# set options for this machine
# specify which compilers to use for c, fortran and linking

# compiler flags to be used (set to compile with debugging on)
CFLAGS += -I. -I$(GSDIR)/src -DUSE_NEW_PINNED_MALLOC=1
CXXFLAGS += -I. -I$(GSDIR)/src -DUSE_NEW_PINNED_MALLOC=1

# libraries to be linked in
LIBS	=   -L$(OCCA_DIR)/lib  $(links) -L$(GSDIR)/lib  -lgs 

#-llapack -lblas

INCLUDES = ogs.hpp ogsKernels.hpp ogsInterface.h
DEPS = $(INCLUDES) \
$(HDRDIR)/types.h 

# types of files we are going to construct rules for
.SUFFIXES: .c .cpp

# rule for .c files
.c.o: $(DEPS)
	$(CC) $(CFLAGS) -o $*.o -c $*.c $(paths)

.cpp.o: $(DEPS)
	$(CXX) $(CXXFLAGS) -o $*.o -c $*.cpp $(paths)

# list of objects to be compiled
OBJS    = \
./src/ogsGatherScatter.o \
./src/ogsGatherScatterVec.o \
./src/ogsGatherScatterMany.o \
./src/ogsGather.o \
./src/ogsGatherVec.o \
./src/ogsGatherMany.o \
./src/ogsScatter.o \
./src/ogsScatterVec.o \
./src/ogsScatterMany.o \
./src/ogsSetup.o \
./src/ogsKernels.o 

COBJS = \
./src/ogsHostGatherScatter.o \
./src/ogsHostGatherScatterVec.o \
./src/ogsHostGatherScatterMany.o \
./src/ogsHostGather.o \
./src/ogsHostGatherVec.o \
./src/ogsHostGatherMany.o \
./src/ogsHostScatter.o \
./src/ogsHostScatterVec.o \
./src/ogsHostScatterMany.o \
./src/ogsHostSetup.o


lib:$(OBJS) gslibInterface
	ar -cr libogs.a $(OBJS) $(COBJS)
	mkdir -p $(PREFIX)
	mkdir -p $(PREFIX)/lib
	cp -a okl $(PREFIX)
	cp libogs.a $(PREFIX)/lib

gslibInterface:
	CC=$(CC) $(MAKE) -C ../../3rdParty/gslib -j lib install 
	$(CC) $(CFLAGS) -c -o ./src/ogsHostGatherScatter.o     ./src/ogsHostGatherScatter.c     $(paths)
	$(CC) $(CFLAGS) -c -o ./src/ogsHostGatherScatterVec.o  ./src/ogsHostGatherScatterVec.c  $(paths)
	$(CC) $(CFLAGS) -c -o ./src/ogsHostGatherScatterMany.o ./src/ogsHostGatherScatterMany.c $(paths)
	$(CC) $(CFLAGS) -c -o ./src/ogsHostGather.o        ./src/ogsHostGather.c        $(paths)
	$(CC) $(CFLAGS) -c -o ./src/ogsHostGatherVec.o     ./src/ogsHostGatherVec.c     $(paths)
	$(CC) $(CFLAGS) -c -o ./src/ogsHostGatherMany.o    ./src/ogsHostGatherMany.c    $(paths)
	$(CC) $(CFLAGS) -c -o ./src/ogsHostScatter.o       ./src/ogsHostScatter.c       $(paths)
	$(CC) $(CFLAGS) -c -o ./src/ogsHostScatterVec.o    ./src/ogsHostScatterVec.c    $(paths)
	$(CC) $(CFLAGS) -c -o ./src/ogsHostScatterMany.o   ./src/ogsHostScatterMany.c   $(paths)
	$(CC) $(CFLAGS) -c -o ./src/ogsHostSetup.o         ./src/ogsHostSetup.c         $(paths)

all: lib

clean:
	rm -rf src/*.o libogs.a

realclean:
	cd ../../3rdParty/gslib; make clean; cd ../../3rdParty/ogs
	rm -rf src/*.o libogs.a


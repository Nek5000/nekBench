ifndef OCCA_DIR
ERROR:
	@echo "Error, environment variable [OCCA_DIR] is not set"
endif

PROJ_DIR=$(CURDIR)
include ${OCCA_DIR}/scripts/Makefile

# define variables
OGSDIR  = $(CURDIR)
GSDIR  ?= ../../3rdParty/gslib

# set options for this machine
# specify which compilers to use for c, fortran and linking

# compiler flags to be used (set to compile with debugging on)

paths += -I$(OGSDIR)
paths += -I$(OGSDIR)/include
paths += -I$(GSDIR)/src
paths += -L$(OCCA_DIR)/lib
paths += -L$(GSDIR)/lib

cCompilerFlags += $(DEFS)
cCompilerFlags += -DDOGS='"${CURDIR}"'
cCompilerFlags += -DUSE_NEW_PINNED_MALLOC=1

compilerFlags += $(DEFS)
compilerFlags += -DDOGS='"${CURDIR}"' 
compilerFlags += -DUSE_NEW_PINNED_MALLOC=1

# libraries to be linked in
linkerFlags	+= -lgs 

#-llapack -lblas

INCLUDES = ogs.hpp ogsKernels.hpp ogsInterface.h
DEPS = $(INCLUDES) \
$(HDRDIR)/types.h 

# types of files we are going to construct rules for
.SUFFIXES: .c .cpp

# rule for .c files
.c.o: $(DEPS)
	$(cCompiler) -c $(cCompilerFlags) $(paths) $*.c -o $*.o

.cpp.o: $(DEPS)
	$(compiler) -c $(compilerFlags) $(paths) $*.cpp -o $*.o

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


lib:$(OBJS) gslibInterface $(COBJS)
	ar -cr libogs.a $(OBJS) $(COBJS)

gslibInterface:
	$(MAKE) -C ../../3rdParty/gslib -j install

all: lib

clean:
	rm -f src/*.o libogs.a

realclean:
	cd ../../3rdParty/gslib; make clean; cd ../../3rdParty/ogs
	rm -f src/*.o libogs.a


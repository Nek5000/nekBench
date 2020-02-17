NALIGN ?= 64

ifndef OCCA_DIR
ERROR:
	@echo "Error, environment variable [OCCA_DIR] is not set"
endif

include ${OCCA_DIR}/scripts/Makefile

# define variables
HDRDIR = ../core
GSDIR  = ../3rdParty/gslib/
OGSDIR = ../3rdParty/ogs/
BLASLAPACK_DIR = ../3rdParty/BlasLapack

# set options for this machine
# specify which compilers to use for c, fortran and linking
CC = mpicc
FC = mpif77  
CC = mpic++
LD = mpic++

compilerFlags += -Ddfloat=double
compilerFlags += -Ddlong=int
compilerFlags += -DdfloatString='"double"'
compilerFlags += -DdlongString='"int"'
compilerFlags += -DUSE_OCCA_MEM_BYTE_ALIGN=$(NALIGN)
compilerFlags += -g

CFLAGS = -I. -DOCCA_VERSION_1_0 $(compilerFlags) $(flags) -I$(HDRDIR) -I$(OGSDIR)  -D DBP='"./"' $(LIBP_OPT_FLAGS) -I$(OGSDIR)/include 

# link flags to be used
LDFLAGS	= -DOCCA_VERSION_1_0 $(compilerFlags) $(LIBP_OPT_FLAGS)

# libraries to be linked in
LIBS	=  -L$(OGSDIR) -logs -L$(GSDIR)/lib -lgs -L$(OCCA_DIR)/lib -locca $(linkerFlags) $(BLASLAPACK_DIR)/libBlasLapack.a  $(LIBP_EXT_LIBS) -lgfortran -fopenmp

INCLUDES = BP.h BPPrecon.h
DEPS = $(INCLUDES) \
$(HDRDIR)/mesh.h \
$(HDRDIR)/mesh2D.h \
$(HDRDIR)/mesh3D.h \
$(OGSDIR)/ogs.hpp \


# types of files we are going to construct rules for
.SUFFIXES: .c .f .cu .cpp

# rule for .c files
.c.o: $(DEPS)
	$(CC) $(CFLAGS) -o $*.o -c $*.c $(paths)

.cpp.o: $(DEPS)
	$(CC) $(CFLAGS) -o $*.o -c $*.cpp $(paths)

.f.o: $(DEPS)
	$(FC) $(CFLAGS) -o $*.o -c $*.f $(paths)

# list of objects to be compiled
AOBJS    = \
./main.o \
./setup.o \
./solve.o\

# library objects
LOBJS = \
../core/meshSetup.o \
../core/meshBasis.o \
../core/setupAide.o

.PHONY: all install clean realclean

all: BP install 

install:
	@mkdir -p $(PREFIX)
	@cp setup.ini $(PREFIX)
	@cp -a kernel $(PREFIX) 
	@cd ../axhelm/kernel; find . -type f -name "*" -exec install -D {} $(PREFIX)/kernel/{} \;
	@cd $(CURDIR)	 
	@mv nekBone $(PREFIX) 


BP: $(AOBJS) $(LOBJS) libogs libblas
	$(LD)  $(LDFLAGS)  -o nekBone $(COBJS) $(AOBJS) $(LOBJS) $(paths) $(LIBS)

libogs:
	$(MAKE) -C $(OGSDIR) -j lib

libblas:
	$(MAKE) -C $(BLASLAPACK_DIR) -j lib

clean:
	@rm -rf *.o ../core/*.o ./build

realclean: clean
	@$(MAKE) -C $(OGSDIR) realclean
	@CC=$(CC) make -C $(BLASLAPACK_DIR) clean

export PREFIX ?= $(CURDIR)/build
export NALIGN ?= 64

export CC	= mpicc
export FC	= mpif77
export CXX= mpic++
export LD	= mpic++

export CFLAGS		= -O3 -fopenmp
export CXXFLAGS = -O3 -fopenmp
export LDFLAGS 	=  -fopenmp

# Bypass MPI compiler wrappers and handle linking ourselves
#export USE_MPI_WRAPPER = no
#export MPI_DIR = $(MPICH_DIR)
#export MPI_LIBFLAG = -lmpich

# Use an external BLAS/LAPACK library
#export USE_SYSTEM_BLASLAPACK = yes
#export BLASLAPACK_LIBFLAG = -mkl

export OCCA_DIR = $(CURDIR)/3rdParty/occa
export OCCA_INCLUDE_PATH 	?=
export OCCA_LIBRARY_PATH 	?=
export OCCA_CUDA_ENABLED	?= 0
export OCCA_HIP_ENABLED 	?= 0
export OCCA_OPENCL_ENABLED?= 0
export OCCA_METAL_ENABLED ?= 0

DEFS = -Ddfloat=double
DEFS += -DdfloatString='"double"'
DEFS += -DMPI_DFLOAT='MPI_DOUBLE'
DEFS += -DdfloatFormat='"%lf"'
DEFS += -Ddlong=int
DEFS += -DdlongString='"int"'
DEFS += -DMPI_DLONG='MPI_INT'
DEFS += -DdlongFormat='"%d"'
DEFS += -Dhlong='long long int'
DEFS += -DhlongString='"long long int"'
DEFS += -DMPI_HLONG='MPI_LONG_LONG_INT'
DEFS += -DhlongFormat='"%lld"'
DEFS += -DUSE_OCCA_MEM_BYTE_ALIGN=$(NALIGN)
DEFS += -DOCCA_VERSION_1_0 
DEFS += -D DBP='"./"'
export DEFS

export HDRDIR = ${CURDIR}/core
export GSDIR  = ${CURDIR}/3rdParty/gslib
export OGSDIR = ${CURDIR}/3rdParty/ogs
export BLASLAPACK_DIR := $(CURDIR)/3rdParty/BlasLapack

NEKBONEDIR = ./nekBone
NEKBONEDEPS= libogs

AXHELMDIR  = ./axhelm 
AXHELMDEPS = 

ifeq ($(USE_SYSTEM_BLASLAPACK),yes)
export LDFLAGS += $(BLASLAPACK_LIBFLAG)	
else
NEKBONEDEPS += libblas
AXHELMDEPS += libblas
export LDFLAGS += -lgfortran 
endif

ifeq ($(USE_MPI_WRAPPER),no)
export LDFLAGS += $(MPI_LIBFLAG)
export OCCA_INCLUDE_PATH += $(MPI_DIR)/include
export OCCA_LIBRARY_PATH += $(MPI_DIR)/lib
endif

.PHONY: install axhelm nekBone libogs libblas all clean realclean

all: occa axhelm nekBone install
	@if test -f ${PREFIX}/axhelm && test -f ${PREFIX}/nekBone; then \
	echo ""; \
	echo "compilation successful!"; \
	echo "install dir: ${PREFIX}"; \
	fi

install:
	@rm -rf $(PREFIX)/libgs.a

axhelm: $(AXHELMDEPS)
	$(MAKE) -C $(AXHELMDIR) 

nekBone: $(NEKBONEDEPS)
	$(MAKE) -C $(NEKBONEDIR)

occa:
	$(MAKE) -j8 -C $(OCCA_DIR)
	@rm -rf ${PREFIX}/lib
	@rm -rf ${PREFIX}/bin
	@rm -rf ${PREFIX}/include

libogs:
	$(MAKE) -C $(OGSDIR) -j4 lib

libblas:
	$(MAKE) -C $(BLASLAPACK_DIR) -j4 lib

clean:
	@$(MAKE) -C $(NEKBONEDIR) clean
	@$(MAKE) -C $(AXHELMDIR) realclean
	@$(MAKE) -C $(OGSDIR) realclean
	@CC=$(CC) $(MAKE) -C $(BLASLAPACK_DIR) clean

realclean: clean
	@$(MAKE) -C $(OCCA_DIR) clean
	@rm -rf ${PREFIX}

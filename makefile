export CC = mpicc
export FC = mpif77
export CXX = mpic++
export LD = mpic++

export NALIGN ?= 64
export PREFIX ?= $(CURDIR)/build

export OCCA_DIR = $(CURDIR)/3rdParty/occa
export OCCA_CUDA_ENABLED=1
export OCCA_HIP_ENABLED=1
export OCCA_OPENCL_ENABLED=1
export OCCA_METAL_ENABLED=0

#export OCCA_INCLUDE_PATH="/usr/local/cuda/include"
#export OCCA_LIBRARY_PATH="/usr/local/cuda/lib"

include ${OCCA_DIR}/scripts/Makefile

flags += -g
flags += -Ddfloat=double
flags += -DdfloatString='"double"'
flags += -DMPI_DFLOAT='MPI_DOUBLE'
flags += -DdfloatFormat='"%lf"'
flags += -Ddlong=int
flags += -DdlongString='"int"'
flags += -DMPI_DLONG='MPI_INT'
flags += -DdlongFormat='"%d"'
flags += -Dhlong='long long int'
flags += -DhlongString='"long long int"'
flags += -DMPI_HLONG='MPI_LONG_LONG_INT'
flags += -DhlongFormat='"%lld"'
flags += -DUSE_OCCA_MEM_BYTE_ALIGN=$(NALIGN)

export HDRDIR = $(CURDIR)/core
export GSDIR  = $(CURDIR)/3rdParty/gslib/
export OGSDIR = $(CURDIR)/3rdParty/ogs/
export BLASLAPACK_DIR = $(CURDIR)/3rdParty/BlasLapack

NEKBONEDIR = ./nekBone 
AXHELMDIR  = ./axhelm 
BWDIR  = ./bw 

export CFLAGS = -I. -DOCCA_VERSION_1_0 $(cCompilerFlags) $(flags) -I$(HDRDIR) -I$(OGSDIR) -I$(OGSDIR)/include  -D DBP='"./"' $(paths)

export CXXFLAGS = -I. -DOCCA_VERSION_1_0 $(compilerFlags) $(flags) -I$(HDRDIR) -I$(OGSDIR) -I$(OGSDIR)/include  -D DBP='"./"' $(paths)

LDFLAGS = $(BLASLAPACK_DIR)/libBlasLapack.a -lgfortran -fopenmp
LDFLAGS_OCCA = -L$(OCCA_DIR)/lib -locca
LDFLAGS_GS = -L$(OGSDIR) -logs -L$(GSDIR)/lib -lgs 

.PHONY: install bw axhelm nekBone all clean realclean

all: occa bw axhelm nekBone install
	@if test -f ${PREFIX}/axhelm && test -f ${PREFIX}/nekBone; then \
	echo ""; \
	echo "compilation successful!"; \
	echo "install dir: ${PREFIX}"; \
	fi

install:
	@rm -rf $(PREFIX)/libgs.a

bw:
	LDFLAGS="$(LDFLAGS_OCCA) $(LDFLAGS)" $(MAKE) -C $(BWDIR) 

axhelm:
	LDFLAGS="$(LDFLAGS_OCCA) $(LDFLAGS)" $(MAKE) -C $(AXHELMDIR) 

nekBone:
	LDFLAGS="$(LDFLAGS_OCCA) $(LDFLAGS_GS) $(LDFLAGS)" $(MAKE) -C $(NEKBONEDIR)

occa:
	$(MAKE) -j8 -C $(OCCA_DIR)
	@rm -rf ${PREFIX}/lib
	@rm -rf ${PREFIX}/bin
	@rm -rf ${PREFIX}/include

clean:
	@$(MAKE) -C $(NEKBONEDIR) clean
	@$(MAKE) -C $(AXHELMDIR) clean

realclean:
	@$(MAKE) -C $(OCCA_DIR) clean
	@$(MAKE) -C $(NEKBONEDIR) realclean
	@$(MAKE) -C $(AXHELMDIR) realclean

#########################################################
# USER SETTINGS
#########################################################
export CC = mpicc
export FC = mpif77
export CXX = mpic++
export LD = mpic++

export NALIGN ?= 64

export OCCA_CUDA_ENABLED=1
export OCCA_HIP_ENABLED=0
export OCCA_OPENCL_ENABLED=0
export OCCA_METAL_ENABLED=0
#export OCCA_INCLUDE_PATH="/usr/local/cuda/include"
#export OCCA_LIBRARY_PATH="/usr/local/cuda/lib"

#########################################################
ifneq (,$(strip $(INSTALLDIR)))
PREFIX = $(abspath $(INSTALLDIR))
else
PREFIX = $(CURDIR)/build
endif
export PREFIX

export OCCA_DIR = $(CURDIR)/3rdParty/occa
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
export GSDIR = $(CURDIR)/3rdParty/gslib
export OGSDIR = $(CURDIR)/3rdParty/ogs
export BLASLAPACK_DIR = $(CURDIR)/3rdParty/BlasLapack
export NEKBONEDIR = $(CURDIR)/nekBone 
export AXHELMDIR  = $(CURDIR)/axhelm 
export BWDIR  = $(CURDIR)/bw 

export CFLAGS = -I. -DOCCA_VERSION_1_0 $(cCompilerFlags) $(flags) -I$(HDRDIR) -I$(OGSDIR) -I$(OGSDIR)/include -DDOGS='"$(PREFIX)/gs/"' -D DBP='"$(PREFIX)/"' $(paths)

export CXXFLAGS = -I. -DOCCA_VERSION_1_0 $(compilerFlags) $(flags) -I$(HDRDIR) -I$(OGSDIR) -I$(OGSDIR)/include -DDOGS='"$(PREFIX)/gs/"' -D DBP='"$(PREFIX)/"' $(paths)

LDFLAGS = $(PREFIX)/blasLapack/lib/libBlasLapack.a -lgfortran -fopenmp
LDFLAGS_OCCA = -L$(PREFIX)/occa/lib -locca
LDFLAGS_GS = -L$(PREFIX)/gs/lib -logs -L$(PREFIX)/gs/lib -lgs 

.PHONY: install bw axhelm nekBone all clean realclean libblas libogs

all: occa nekBone axhelm bw install
	@if test -f ${PREFIX}/axhelm && test -f ${PREFIX}/nekBone; then \
	echo ""; \
	echo "install dir: ${PREFIX}"; \
	echo "please set the following env-vars:"; \
	echo "  export OCCA_DIR=${PREFIX}/occa"; \
	echo "  export LD_LIBRARY_PATH=\$$LD_LIBRARY_PATH:\$$OCCA_DIR/lib"; \
	echo ""; \
	echo "compilation successful!"; \
	echo ""; \
	fi

install:

bw:
	LDFLAGS="$(LDFLAGS_OCCA) $(LDFLAGS)" $(MAKE) -C $(BWDIR) 

axhelm: libblas
	LDFLAGS="$(LDFLAGS_OCCA) $(LDFLAGS)" $(MAKE) -C $(AXHELMDIR) 

nekBone: libogs libblas
	LDFLAGS="$(LDFLAGS_OCCA) $(LDFLAGS_GS) $(LDFLAGS)" $(MAKE) -C $(NEKBONEDIR)

occa:
	@PREFIX=$(PREFIX)/occa $(MAKE) -j8 -C $(OCCA_DIR)

libogs:
	@PREFIX=$(PREFIX)/gs $(MAKE) -C $(OGSDIR) -j4 lib

libblas:
	@PREFIX=$(PREFIX)/blasLapack $(MAKE) -C $(BLASLAPACK_DIR) -j4 lib

clean:
	@$(MAKE) -C $(NEKBONEDIR) clean
	@$(MAKE) -C $(AXHELMDIR) clean
	@$(MAKE) -C $(BWDIR) clean

realclean: clean
	@rm -rf core/*.o
	@$(MAKE) -C $(OCCA_DIR) clean
	@$(MAKE) -C $(OGSDIR) clean
	@$(MAKE) -C $(GSDIR) clean
	@$(MAKE) -C $(BLASLAPACK_DIR) clean

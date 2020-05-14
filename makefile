#########################################################
# USER SETTINGS
#########################################################
export CC = mpicc
export FC = mpif77
export CXX = mpic++
export LD = mpic++

export NALIGN ?= 64

export OCCA_CUDA_ENABLED=0
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
export LIBGSDIR = $(CURDIR)/3rdParty/gslib
export OLIBGSDIR = $(CURDIR)/3rdParty/ogs
export BLASLAPACK_DIR = $(CURDIR)/3rdParty/BlasLapack
export NEKBONEDIR = $(CURDIR)/nekBone 
export AXHELMDIR  = $(CURDIR)/axhelm 
export BWDIR  = $(CURDIR)/bw 
export ADVDIR  = $(CURDIR)/adv 
export DOTDIR  = $(CURDIR)/dot 
export GSDIR  = $(CURDIR)/gs 

export CFLAGS = -I. -DOCCA_VERSION_1_0 $(cCompilerFlags) $(flags) -I$(HDRDIR) -I$(OLIBGSDIR) -I$(OLIBGSDIR)/include -DDOGS='"$(PREFIX)/libgs/"' -D DBP='"$(PREFIX)/"' $(paths)

export CXXFLAGS = -I. -DOCCA_VERSION_1_0 $(compilerFlags) $(flags) -I$(HDRDIR) -I$(OLIBGSDIR) -I$(OLIBGSDIR)/include -DDOGS='"$(PREFIX)/libgs/"' -D DBP='"$(PREFIX)/"' $(paths)

LDFLAGS = -lgfortran -fopenmp
LDFLAGS_BLAS = $(PREFIX)/blasLapack/lib/libBlasLapack.a
LDFLAGS_OCCA = -L$(PREFIX)/occa/lib -locca
LDFLAGS_GS = -L$(PREFIX)/libgs/lib -logs -L$(PREFIX)/libgs/lib -lgs 

.PHONY: install bw dot axhelm adv gs nekBone all clean realclean libblas libogs

all: occa nekBone axhelm bw dot adv gs install
	@rm -rf $(PREFIX)/blasLapack \
	echo ""; \
	echo "install dir: ${PREFIX}"; \
	echo "please set the following env-vars:"; \
	echo "  export OCCA_DIR=${PREFIX}/occa"; \
	echo "  export LD_LIBRARY_PATH=\$$LD_LIBRARY_PATH:\$$OCCA_DIR/lib"; \
	echo ""; \
	echo "compilation successful!"; \
	echo "";

install:

bw: occa
	LDFLAGS="$(LDFLAGS_OCCA) $(LDFLAGS)" $(MAKE) -C $(BWDIR) 

dot: occa
	LDFLAGS="$(LDFLAGS_OCCA) $(LDFLAGS)" $(MAKE) -C $(DOTDIR) 

adv: occa
	LDFLAGS="$(LDFLAGS_OCCA) $(LDFLAGS)" $(MAKE) -C $(ADVDIR) 

axhelm: occa libblas
	LDFLAGS="$(LDFLAGS_OCCA) $(LDFLAGS_BLAS) $(LDFLAGS)" $(MAKE) -C $(AXHELMDIR) 

nekBone: occa libogs libblas
	LDFLAGS="$(LDFLAGS_OCCA) $(LDFLAGS_GS) $(LDFLAGS_BLAS) $(LDFLAGS)" $(MAKE) -C $(NEKBONEDIR)

gs: occa libogs libblas
	LDFLAGS="$(LDFLAGS_OCCA) $(LDFLAGS_GS) $(LDFLAGS_BLAS) $(LDFLAGS)" $(MAKE) -C $(GSDIR)

occa:
	@PREFIX=$(PREFIX)/occa $(MAKE) -j8 -C $(OCCA_DIR)

libogs:
	@PREFIX=$(PREFIX)/libgs $(MAKE) -C $(OLIBGSDIR) -j4 lib

libblas:
	@PREFIX=$(PREFIX)/blasLapack $(MAKE) -C $(BLASLAPACK_DIR) -j4 lib

clean:
	@$(MAKE) -C $(NEKBONEDIR) clean
	@$(MAKE) -C $(AXHELMDIR) clean
	@$(MAKE) -C $(BWDIR) clean
	@$(MAKE) -C $(DOTDIR) clean
	@$(MAKE) -C $(GSDIR) clean

realclean: clean
	@rm -rf core/*.o
	@$(MAKE) -C $(OCCA_DIR) clean
	@$(MAKE) -C $(OLIBGSDIR) clean
	@$(MAKE) -C $(LIBGSDIR) clean
	@$(MAKE) -C $(BLASLAPACK_DIR) clean

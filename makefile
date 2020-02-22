ifndef OCCA_DIR
ERROR:
	@echo "ERROR:  Environment variable OCCA_DIR is not set."
endif
include ${OCCA_DIR}/scripts/Makefile

export CC = mpicc
export FC = mpif77
export CXX = mpic++
export LD = mpic++

export NALIGN ?= 64
export PREFIX ?= "$(CURDIR)/build"

export HDRDIR = $(CURDIR)/core
export GSDIR  = $(CURDIR)/3rdParty/gslib/
export OGSDIR = $(CURDIR)/3rdParty/ogs/
export BLASLAPACK_DIR = $(CURDIR)/3rdParty/BlasLapack

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

export CFLAGS = -I. -DOCCA_VERSION_1_0 $(cCompilerFlags) $(flags) -I$(HDRDIR) -I$(OGSDIR)  -D DBP='"./"' $(LIBP_OPT_FLAGS) -I$(OGSDIR)/include $(paths)

export CXXFLAGS = -I. -DOCCA_VERSION_1_0 $(compilerFlags) $(flags) -I$(HDRDIR) -I$(OGSDIR)  -D DBP='"./"' $(LIBP_OPT_FLAGS) -I$(OGSDIR)/include $(paths)

NEKBONEDIR = ./nekBone 
AXHELMDIR  = ./axhelm 

.PHONY: install axhelm nekBone all clean realclean

all: axhelm nekBone install
	@if test -f ${PREFIX}/axhelm && test -f ${PREFIX}/nekBone; then \
	echo ""; \
	echo "compilation successful!"; \
	echo "install dir: ${PREFIX}"; \
	fi

install:
	@rm -rf $(PREFIX)/libgs.a

axhelm:
	$(MAKE) -C $(AXHELMDIR) 

nekBone:
	$(MAKE) -C $(NEKBONEDIR)

clean:
	@$(MAKE) -C $(NEKBONEDIR) realclean
	@$(MAKE) -C $(NEKBONEDIR) realclean

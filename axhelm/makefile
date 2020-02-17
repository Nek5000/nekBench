WS ?= 8
MPI ?= 1
NALIGN ?= 64
PREFIX ?= "$(CURDIR)/build"

PROJ_DIR=./
CORE_DIR=../core

ifndef OCCA_DIR
ERROR:
	@echo "ERROR:  Environment variable OCCA_DIR is not set."
endif

include ${OCCA_DIR}/scripts/Makefile

CC  = mpicc
CXX = mpicxx

ifeq (1,$(MPI))
compilerFlags += -DUSEMPI
endif
compilerFlags += -Ddlong=int
compilerFlags += -DdlongString='"int"'
ifeq (4,$(WS))
compilerFlags += -Ddfloat=float
compilerFlags += -DdfloatString='"float"'
else
compilerFlags += -Ddfloat=double
compilerFlags += -DdfloatString='"double"'
endif
compilerFlags += -DUSE_OCCA_MEM_BYTE_ALIGN=$(NALIGN)

BLASLAPACK_DIR = ../3rdParty/BlasLapack

.PHONY: all install clean realclean

all: axhelm install

axhelm: ${PROJ_DIR}/main.cpp makefile ${CORE_DIR}/meshBasis.cpp libblas
	$(CXX) -DOCCA_VERSION_1_0 $(compilerFlags) -o axhelm $(linkerFlags) ${PROJ_DIR}/main.cpp -I${CORE_DIR} ${CORE_DIR}/meshBasis.cpp $(paths) $(BLASLAPACK_DIR)/libBlasLapack.a -lgfortran -L${OCCA_DIR}/lib $(flags) 
#$(linkerFlags)


$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(CXX) -DOCCA_VERSION_1_0 -g $(compilerFlags) -o $@ $(flags) -c $(paths) $<

install:
	@mkdir -p $(PREFIX)/kernel
	@cd kernel && find . -type f -name "*" -exec install -D {} $(PREFIX)/kernel/{} \; && cd ..
	@mv axhelm $(PREFIX) 

libblas:
	@cd ../3rdParty/BlasLapack; $(MAKE) -j4 lib

clean:
	@rm -f ${PROJ_DIR}/axhelm

realclean: clean
	@$(MAKE) -C $(BLASLAPACK_DIR) clean

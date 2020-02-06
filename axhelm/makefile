
PROJ_DIR=./
CORE_DIR=../core

SP ?= 0

ifndef OCCA_DIR
ERROR:
	@echo "ERROR:  Environment variable OCCA_DIR is not set."
endif

include ${OCCA_DIR}/scripts/Makefile

compiler = mpic++
compilerFlags += -Ddlong=int
compilerFlags += -DdlongString='"int"'
ifeq (1,$(SP))
compilerFlags += -Ddfloat=float
compilerFlags += -DdfloatString='"float"'
else
compilerFlags += -Ddfloat=double
compilerFlags += -DdfloatString='"double"'
endif
compilerFlags += -DUSE_OCCA_MEM_BYTE_ALIGN=64

BLASLAPACK_DIR = ../3rdParty/BlasLapack

${PROJ_DIR}/BK6: ${PROJ_DIR}/main.cpp makefile ${CORE_DIR}/meshBasis.cpp libblas
	$(compiler) -DOCCA_VERSION_1_0 $(compilerFlags) -o axhelm $(linkerFlags) ${PROJ_DIR}/main.cpp -I${CORE_DIR} ${CORE_DIR}/meshBasis.cpp $(paths) $(BLASLAPACK_DIR)/libBlasLapack.a -lgfortran -L${OCCA_DIR}/lib $(flags) 
#$(linkerFlags)


$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(compiler) -DOCCA_VERSION_1_0 -g $(compilerFlags) -o $@ $(flags) -c $(paths) $<

libblas:
	cd ../3rdParty/BlasLapack; make -j lib

clean:
	rm -f ${PROJ_DIR}/axhelm
	make -C $(BLASLAPACK_DIR) clean

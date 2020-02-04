
PROJ_DIR=./
CORE_DIR=../core

ifndef OCCA_DIR
ERROR:
	@echo "ERROR:  Environment variable OCCA_DIR is not set."
endif

include ${OCCA_DIR}/scripts/Makefile

compiler = mpic++
compilerFlags += -Ddlong=int
compilerFlags += -DdlongString='"int"'
compilerFlags += -Ddfloat=double
compilerFlags += -DdfloatString='"double"'
compilerFlags += -DUSE_OCCA_MEM_BYTE_ALIGN=32

${PROJ_DIR}/BK6: ${PROJ_DIR}/main.cpp makefile ${CORE_DIR}/meshBasis.cpp
	$(compiler) -DOCCA_VERSION_1_0 $(compilerFlags) -o matvec $(linkerFlags) ${PROJ_DIR}/main.cpp -I${CORE_DIR} ${CORE_DIR}/meshBasis.cpp $(paths) -L${OCCA_DIR}/lib $(flags) -locca -llapack -lblas
#$(linkerFlags)


$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(compiler) -DOCCA_VERSION_1_0 $(compilerFlags) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f ${PROJ_DIR}/matvec

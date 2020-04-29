WS ?= 8

PROJ_DIR=./
CORE_DIR=$(HDRDIR)

ifeq (0,$(MAKELEVEL))
ERROR:
	@echo "ERROR:  Please use toplevel makefile"
endif

.PHONY: all install clean realclean

all: axhelm install

axhelm: main.cpp kernelHelper.cpp ${CORE_DIR}/meshBasis.cpp libblas
	$(CXX) $(CXXFLAGS) -o axhelm main.cpp ${CORE_DIR}/meshBasis.cpp $(linkerFlags) $(LDFLAGS)

$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(CXX) $(CXXFLAGS) -o $@ -c $<

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

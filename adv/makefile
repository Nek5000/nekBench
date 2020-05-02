WS ?= 8

PROJ_DIR=./
CORE_DIR=$(HDRDIR)

ifeq (0,$(MAKELEVEL))
ERROR:
	@echo "ERROR:  Please use toplevel makefile"
endif

.PHONY: all install clean realclean

all: adv install

adv: main.cpp kernelHelper.cpp ${CORE_DIR}/setCompilerFlags.hpp
	$(CXX) $(CXXFLAGS) -o adv main.cpp $(linkerFlags) $(LDFLAGS)

$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(CXX) $(CXXFLAGS) -o $@ -c $<

install:
	@mkdir -p $(PREFIX)/kernel
	@cd kernel && find . -type f -name "*" -exec install -D {} $(PREFIX)/kernel/{} \; && cd ..
	@mv adv $(PREFIX) 

clean:
	@rm -rf *.o

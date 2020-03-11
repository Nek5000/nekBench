WS ?= 8

PROJ_DIR=$(CURDIR)
include ${OCCA_DIR}/scripts/Makefile

ifeq (0,$(MAKELEVEL))
ERROR:
	@echo "ERROR:  Please use toplevel makefile"
endif

paths += -I$(HDRDIR)
paths += -L$(OCCA_DIR)/lib

compilerFlags += $(DEFS)

ifeq ($(USE_SYSTEM_BLASLAPACK),no)
	linkerFlags += $(BLASLAPACK_DIR)/libBlasLapack.a
endif

.PHONY: all install clean realclean

all: axhelm install

axhelm: main.cpp $(HDRDIR)/meshBasis.cpp
	$(compiler) $(compilerFlags) $(paths) $^ -o $(@) $(linkerFlags)

$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(compiler) -c $(compilerFlags) $(paths) $< -o $(@)

install:
	@mkdir -p $(PREFIX)/kernel
	@cd kernel && find . -type f -name "*" -exec install -D {} $(PREFIX)/kernel/{} \; && cd ..
	@mv axhelm $(PREFIX) 

clean:
	@rm -f ${PROJ_DIR}/axhelm

realclean: clean
	@$(MAKE) -C $(BLASLAPACK_DIR) clean

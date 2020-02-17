# set options for this machine
# specify which compilers to use for c, fortran and linking
NALIGN ?= 64
PREFIX ?= "$(CURDIR)/build"

export NALIGN
export PREFIX

ifndef OCCA_DIR
ERROR:
	@echo "Error, environment variable [OCCA_DIR] is not set"
endif

NEKBONEDIR = ./nekBone 
AXHELMDIR  = ./axhelm 

.PHONY: install axhelm nekBone all clean realclean

all: axhelm nekBone install 

install:
	@rm -rf $(PREFIX)/libgs.a

axhelm:
	$(MAKE) -C $(AXHELMDIR) 

nekBone:
	$(MAKE) -C $(NEKBONEDIR)

clean:
	@$(MAKE) -C $(NEKBONEDIR) realclean
	@$(MAKE) -C $(NEKBONEDIR) realclean

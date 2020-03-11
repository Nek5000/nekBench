ifeq (0,$(MAKELEVEL))
ERROR:
	@echo "ERROR:  Please use toplevel makefile"
endif

PROJ_DIR=$(CURDIR)
include $(OCCA_DIR)/scripts/Makefile

paths += -I$(HDRDIR)
paths += -I$(GSDIR)/src
paths += -I$(OGSDIR)
paths += -L$(OCCA_DIR)/lib
paths += -L$(GSDIR)/lib
paths += -L$(OGSDIR)

compilerFlags += $(DEFS)

linkerFlags += -logs -lgs

ifeq ($(USE_SYSTEM_BLASLAPACK),no)
	linkerFlags += $(BLASLAPACK_DIR)/libBlasLapack.a
endif

INCLUDES = BP.h BPPrecon.h
DEPS = $(INCLUDES) \
$(HDRDIR)/timer.hpp \
$(HDRDIR)/mesh.h \
$(HDRDIR)/mesh2D.h \
$(HDRDIR)/mesh3D.h \
$(OGSDIR)/ogs.hpp \


# types of files we are going to construct rules for
.SUFFIXES: .c .f .cpp

# rule for .c files
.c.o: $(DEPS)
	$(cCompiler) $(cCompilerFlags) $(paths) -o $*.o -c $*.c 

.cpp.o: $(DEPS)
	$(compiler) $(compilerFlags) $(paths) -o $*.o -c $*.cpp

.f.o: $(DEPS)
	$(FC) $(cCompilerFlags) $(paths) -o $*.o -c $*.f 

# list of objects to be compiled
AOBJS    = \
./main.o \
./setup.o \
./solve.o\

# library objects
LOBJS = \
../core/timer.o \
../core/meshSetup.o \
../core/meshBasis.o \
../core/setupAide.o

.PHONY: all install clean realclean

all: BP install 

install:
	@mkdir -p $(PREFIX)
	@cp setup.ini $(PREFIX)/nekBone.ini
	@cp -a kernel $(PREFIX) 
	@cd ../axhelm/kernel; find . -type f -name "*" -exec install -D {} $(PREFIX)/kernel/{} \;
	@cd $(CURDIR)	 
	@mv nekBone $(PREFIX) 

BP: $(AOBJS) $(LOBJS)
	$(compiler) $(compilerFlags) $(paths) $^ -o nekBone $(linkerFlags)

clean:
	@rm -rf *.o ../core/*.o ./build

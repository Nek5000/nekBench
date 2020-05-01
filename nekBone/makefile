ifeq (0,$(MAKELEVEL))
ERROR:
	@echo "ERROR:  Please use toplevel makefile"
endif

INCLUDES = BP.h BPPrecon.h kernelHelper.cpp
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
	$(CC) $(CFLAGS) -o $*.o -c $*.c 

.cpp.o: $(DEPS)
	$(CXX) $(CXXFLAGS) -o $*.o -c $*.cpp

.f.o: $(DEPS)
	$(FC) $(CFLAGS) -o $*.o -c $*.f 

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


BP: $(AOBJS) $(LOBJS) libogs libblas
	$(LD)  -o nekBone $(COBJS) $(AOBJS) $(LOBJS) $(LDFLAGS)

libogs:
	@PREFIX=$(PREFIX)/gs $(MAKE) -C $(OGSDIR) -j4 lib 

libblas:
	@PREFIX=$(PREFIX)/blasLapack $(MAKE) -C $(BLASLAPACK_DIR) -j4 lib

clean:
	@rm -rf *.o ../core/*.o ./build

realclean: clean
	@$(MAKE) -C $(OGSDIR) realclean
	@CC=$(CC) make -C $(BLASLAPACK_DIR) clean

ifeq (0,$(MAKELEVEL))
ERROR:
	@echo "ERROR:  Please use toplevel makefile"
endif

# libraries to be linked in
LIBS	=  -L$(OGSDIR) -logs -L$(GSDIR)/lib -lgs -L$(OCCA_DIR)/lib -locca $(linkerFlags) $(BLASLAPACK_DIR)/libBlasLapack.a  $(LIBP_EXT_LIBS) -lgfortran -fopenmp

INCLUDES = BP.h BPPrecon.h
DEPS = $(INCLUDES) \
$(HDRDIR)/mesh.h \
$(HDRDIR)/mesh2D.h \
$(HDRDIR)/mesh3D.h \
$(OGSDIR)/ogs.hpp \


# types of files we are going to construct rules for
.SUFFIXES: .c .f .cu .cpp

# rule for .c files
.c.o: $(DEPS)
	$(CC) $(CFLAGS) -o $*.o -c $*.c $(paths)

.cpp.o: $(DEPS)
	$(CXX) $(CXXFLAGS) -o $*.o -c $*.cpp $(paths)

.f.o: $(DEPS)
	$(FC) $(CFLAGS) -o $*.o -c $*.f $(paths)

# list of objects to be compiled
AOBJS    = \
./main.o \
./setup.o \
./solve.o\

# library objects
LOBJS = \
../core/meshSetup.o \
../core/meshBasis.o \
../core/setupAide.o

.PHONY: all install clean realclean

all: BP install 

install:
	@mkdir -p $(PREFIX)
	@cp setup.ini $(PREFIX)
	@cp -a kernel $(PREFIX) 
	@cd ../axhelm/kernel; find . -type f -name "*" -exec install -D {} $(PREFIX)/kernel/{} \;
	@cd $(CURDIR)	 
	@mv nekBone $(PREFIX) 


BP: $(AOBJS) $(LOBJS) libogs libblas
	$(LD)  $(LDFLAGS)  -o nekBone $(COBJS) $(AOBJS) $(LOBJS) $(paths) $(LIBS)

libogs:
	$(MAKE) -C $(OGSDIR) -j lib

libblas:
	$(MAKE) -C $(BLASLAPACK_DIR) -j lib

clean:
	@rm -rf *.o ../core/*.o ./build

realclean: clean
	@$(MAKE) -C $(OGSDIR) realclean
	@CC=$(CC) make -C $(BLASLAPACK_DIR) clean

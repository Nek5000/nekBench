ifeq (0,$(MAKELEVEL))
ERROR:
	@echo "ERROR:  Please use toplevel makefile"
endif

INCLUDES =  
DEPS = $(INCLUDES) \
$(HDRDIR)/timer.hpp \
$(HDRDIR)/mesh.h \
$(HDRDIR)/mesh2D.h \
$(HDRDIR)/mesh3D.h \
$(OGSDIR)/ogs.hpp

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

# library objects
LOBJS = \
../core/timer.o \
../core/meshSetup.o \
../core/meshBasis.o \
../core/setupAide.o

.PHONY: all install clean realclean

all: gs install 

install:
	@mkdir -p $(PREFIX)
	@cd $(CURDIR)	 
	@mv gs $(PREFIX) 


gs: $(AOBJS) $(LOBJS)
	$(LD)  -o gs $(COBJS) $(AOBJS) $(LOBJS) $(LDFLAGS)

clean:
	@rm -rf *.o

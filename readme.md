# nekRSbench 

nekRSbench is a benchmark suite representing key kernels of nekRS.
It serves as a lightweight tool for performance analysis on high performance computing architectures. 

### Available benchmarks
* axhelm (local Helmholtz matrix-vector product)
* gs (sparse communcation - regular nearest neighbor exchange and irregular AMG)
* nekBone (solving homogeneous Helmholtz equation using PCG+Jacobi)

# Installation

First you need to install OCCA :
```
git clone https://github.com/libocca/occa
cd occa
make -j
export OCCA_DIR=$HOME/occa
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OCCA_DIR/lib
```

Then, just run 
```
make PREFIX=<install dir>
```
to build the benchmarks. 

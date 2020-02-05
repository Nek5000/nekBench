# Setup

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
make
```

to build the benchmark. 

# Examples
Here a few examples how to run the benchmark for 8192 elements with a polynomial degree 7

### Single GPU with OCCA kernel tuned for NVidia's lastest GPUs (kernel version 0)
```
./axhelm 7 8192 CUDA 0 0
```

### Single GPU with native CUDA kernel
```
./axhelm 7 8192 NATIVE+CUDA 0 0
```

### MPI with native CPU kernel
```
OCCA_CXX='g++' OCCA_CXXFLAGS='-O3 -march=native -mtune=native' mpirun -np 32 -bind-to core ./axhelm 7 256 MPI+NATIVE+SERIAL 0
```

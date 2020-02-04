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

### Single GPU with OCCA kernel tuned for NVidia's lastest GPUs
```
OCCA_CUDA_COMPILER_FLAGS='--use_fast_math' ./axhelm 8 8192 CUDA 0
```
### MPI with native CPU kernel
```
>OCCA_CXXFLAGS='-O3 -march=native -mtune=native' mpirun -np 32 -bind-to core ./axhelm 8 256 MPI+NATIVE+SERIAL 0
```

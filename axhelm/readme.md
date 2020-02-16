This benchmark computes the Helmholtz operator 

```
H = lambda_0*A + lambda_1*B
```
where A is the Laplace operator. All elements are assumed to be deformed and lambda is variable.  

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
make MPI=1 WS=8
```

to build the benchmark. 

# Usage

```
./axhelm polynomialDegree Ndim numElements [NATIVE|OKL]+SERIAL|CUDA|OPENCL arch [kernelVersion] [deviceID] [platformID]
```
Tuned kernels for the following architectures are available:
* Nvidia VOLTA
* Generic CPU 	

# Examples
Here a few examples how to run the benchmark for 8192 elements with a polynomial degree 7

### Single GPU with OCCA kernel tuned for NVidia's lastest GPUs
```
./axhelm 7 1 8192 OCCA+CUDA VOLTA
```

### Single GPU with native CUDA kernel
```
./axhelm 7 1 8192 NATIVE+CUDA VOLTA
```

### MPI with native CPU kernel
```
OCCA_CXX='g++' OCCA_CXXFLAGS='-O3 -march=native -mtune=native' mpirun -np 32 -bind-to core ./axhelm 7 1 256 NATIVE+SERIAL CPU
```

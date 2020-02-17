This benchmark computes the Helmholtz matrix-vector product 

```
AU = helm1*[A]u + helm2*[B]u
```
where A is the Laplace operator. All elements are assumed to be deformed and coefficiencts are variable.  

# Usage

```
./axhelm polynomialDegree Ndim numElements [NATIVE|OKL]+SERIAL|CUDA|OPENCL arch [kernelVersion] [deviceID] [platformID]
```
Tuned kernels for the following architectures are available:
* VOLTA (NVidia Pascal, Volta or Turing)
* CPU (generic)	

# Examples
Here a few examples how to run the benchmark for 8000 elements with a polynomial degree 7

### Single GPU with OCCA kernel tuned for NVidia's lastest GPUs
```
./axhelm 7 1 8000 OCCA+CUDA VOLTA
```

### Single GPU with native CUDA kernel
```
./axhelm 7 1 8000 NATIVE+CUDA VOLTA
```

### Pure MPI with native CPU kernel
```
OCCA_CXX='g++' OCCA_CXXFLAGS='-O3 -march=native -mtune=native' mpirun -np 48 -bind-to core ./axhelm 7 1 166 NATIVE+SERIAL CPU
```

### Hybrid MPI+openMP with native CPU kernel
```
OMP_PLACES=cores OMP_PROC_BIND=close OMP_NUM_THREADS=24 OCCA_CXX='g++' OCCA_CXXFLAGS='-O3 -march=native -mtune=native -fopenmp' mpirun -np 2 -bind-to socket ./axhelm 7 1 4000 NATIVE+OPENMP CPU
```

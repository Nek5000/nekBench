This benchmark computes the Helmholtz matrix-vector product on deformed hexhedral spectral elements 

```
AU = [A]u + lambda[B]u
```
where A is the Laplace operator.

# Usage

```
./axhelm polynomialDegree Ndim numElements [NATIVE|OKL]+SERIAL|CUDA|OPENCL arch [kernelVersion] [deviceID] [platformID]
```
Tuned kernels for the following architectures are available:
* VOLTA (NVidia Pascal, Volta or Turing)
* CPU (generic)	

# Examples
Here a few examples how to run the benchmark for 8000 elements with a polynomial degree 7

### Single GPU with OCCA kernel
```
>./axhelm 7 1 8000 OCCA+CUDA VOLTA
MPItasks=1 OMPthreads=1 Ndim=1 N=7 Nelements=8000 elapsed time=0.000393228 GDOF/s=6.97814 GB/s=749.977 GFLOPS/s=1187.46
```

### Single GPU with native CUDA kernel
```
>./axhelm 7 1 8000 NATIVE+CUDA VOLTA
MPItasks=1 OMPthreads=1 Ndim=1 N=7 Nelements=8000 elapsed time=0.000395148 GDOF/s=6.94424 GB/s=746.334 GFLOPS/s=1181.7
```

### Hybrid MPI+OpenMP with native CPU kernel
```
>OMP_PLACES=cores OMP_PROC_BIND=close OMP_NUM_THREADS=24 OCCA_CXX='g++' OCCA_CXXFLAGS='-O3 -march=native -mtune=native -fopenmp' mpirun -np 2 -bind-to socket ./axhelm 7 1 4000 NATIVE+OPENMP CPU
MPItasks=2 OMPthreads=24 Ndim=1 N=7 Nelements=8000 elapsed time=0.00120629 GDOF/s=2.27475 GB/s=244.479 GFLOPS/s=387.092
```

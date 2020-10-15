This benchmark computes the Helmholtz matrix-vector product
```
AU = lambda0*[A]u + lambda1*[B]u
```
or in BK mode
```
AU = [A]u
```
on deformed hexhedral spectral elements where A is the Laplace operator.

# Usage

```
./axhelm N Ndim numElements [NATIVE|OKL]+SERIAL|CUDA|HIP|OPENCL CPU|VOLTA [BKmode] [nRepetitions] [kernelVersion]
```
Tuned kernels for the following architectures are available:
* VOLTA (NVidia Pascal, Volta or Turing)
* CPU (generic)	

# Examples
Here a few examples how to run the benchmark for 2000 elements with a polynomial degree 7

### Single Nvidia V100
```
>./axhelm 7 1 8000 OCCA+CUDA VOLTA

word size: 8 bytes
active occa mode: CUDA
BK mode enabled
Correctness check: maxError = 9.09495e-13
MPItasks=1 OMPthreads=96 NRepetitions=100 Ndim=1 N=7 Nelements=4000 elapsed time=0.00018497 GDOF/s=7.41742 GB/s=797.189 GFLOPS/s=1229 
```

### Hybrid MPI+OpenMP with native CPU kernel on two socket Intel Xeon Gold 6252
```
>OMP_PLACES=cores OMP_PROC_BIND=close OMP_NUM_THREADS=24 OCCA_CXX='g++' OCCA_CXXFLAGS='-O3 -march=native -mtune=native -fopenmp' mpirun -np 2 -bind-to socket ./axhelm 7 1 2000 NATIVE+OPENMP CPU

word size: 8 bytes
active occa mode: OpenMP
BK mode enabled
Correctness check: maxError = 1.36424e-12
MPItasks=2 OMPthreads=24 NRepetitions=100 Ndim=1 N=7 Nelements=4000 elapsed time=0.000351181 GDOF/s=3.90682 GB/s=419.886 GFLOPS/s=647.324
```

This benchmark solves a 3D inhomogenous Helmholtz equation 
```
lambda0*[A]u + lambda1*[B]u = f
```
on a deformed hexahedral spectral element mesh using conjuate gradients.

# Usage

```
./nekBone <setup ini file>
```
Tuned kernels for the following architectures are available:
* VOLTA (Nvidia Pascal and Volta)
* CPU (generic)	

# Examples

### MPI+serial with native CPU kernel

Just run
```
>OCCA_CXX='g++' OCCA_CXXFLAGS='-O3 -march=native -mtune=native' mpirun -np 48 -bind-to core ./nekBone nekBone.ini

active occa mode: Serial
Compiling GatherScatter Kernels...done.
gs_setup: 267913 unique labels shared
   pairwise times (avg, min, max): 7.46851e-05 7.09057e-05 8.31842e-05
   crystal router                : 0.00019652 0.000191188 0.000199413
   used all_to_all method: pairwise
   handle bytes (avg, min, max): 752036 731316 776516
   buffer bytes (avg, min, max): 204179 182928 229440
setup done: bytes allocated = 40309856

correctness check: globalMaxError = 1.252e-08

N, Nfields, Nelements, elapsed, iterations, GDOF/s/iter, BW GB/s, kernel Id
7, 1, 8000, 4.75473, 1001, 0.577687, 145.341, 0
```

### MPI+OpenMP with native CPU kernels

Set in setup.ini
```
[THREAD MODEL]
NATIVE+OPENMP
```

Now run
```
>OMP_PLACES=cores OMP_PROC_BIND=close OMP_NUM_THREADS=24 OCCA_CXX='g++' OCCA_CXXFLAGS='-O3 -march=native -mtune=native -fopenmp' mpirun -np 2 -bind-to socket ./nekBone nekBone.ini

active occa mode: OpenMP
Compiling GatherScatter Kernels...done.
gs_setup: 39200 unique labels shared
   pairwise times (avg, min, max): 0.000137198 0.000136185 0.000138211
   crystal router                : 0.000142539 0.000140381 0.000144696
   all reduce                    : 0.00024184 0.000241494 0.000242186
   used all_to_all method: pairwise
   handle bytes (avg, min, max): 1.4173e+07 14173044 14173044
   buffer bytes (avg, min, max): 627200 627200 627200
setup done: bytes allocated = 804735224

correctness check: globalMaxError = 6.21725e-15

N, Nfields, Nelements, elapsed, iterations, GDOF/s/iter, BW GB/s, kernel Id
7, 1, 8000, 5.19151, 1001, 0.529084, 133.113, 0
```

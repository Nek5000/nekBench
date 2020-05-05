This benchmark solves a 3D inhomogenous Helmholtz equation 
```
lambda0*[A]u + lambda1*[B]u = f
```
on a deformed hexahedral spectral element mesh using Jacobi preconditioned conjuate gradients.
Note, this benchmark is different from the old nekBone or CEED BP5 due to diagonal term [B] and the variable coefficients.

# Usage

```
./nekBone <setup ini file>
```
Tuned kernels for the following architectures are available:
* VOLTA (Nvidia Pascal and Volta)
* CPU (generic)	

# Examples

### Single GPU with OCCA kernel
```
>mpirun -np 1 -bind-to core ./nekBone nekBone.ini

Compiling GatherScatter Kernels...done.
gs_setup: 0 unique labels shared
   handle bytes (avg, min, max): 2.70722e+07 27072220 27072220
   buffer bytes (avg, min, max): 0 0 0
setup done: bytes allocated = 1663553624
correctness check: maxError = 0.916949

running solver ... done

summary
  MPItasks     : 1
  polyN        : 7
  Nelements    : 8000
  iterations   : 5
  Nrepetitions : 1000
  elapsed time : 9.32177 s
  throughput   : 1.47182 GDOF/s/iter
  bandwidth    : 370.127 GB/s
``` 

### MPI+OpenMP with native CPU kernels

Set in nekBone.ini
```
[ARCH]
CPU

[THREAD MODEL]
NATIVE+OPENMP
```

Now run
```
>OMP_PLACES=cores OMP_PROC_BIND=close OMP_NUM_THREADS=24 OCCA_CXX='g++' OCCA_CXXFLAGS='-O3 -march=native -mtune=native -fopenmp' mpirun -np 2 -bind-to socket ./nekBone nekBone.ini

active occa mode: OpenMP
Compiling GatherScatter Kernels...done.
gs_setup: 39200 unique labels shared
   pairwise times (avg, min, max): 0.000143051 0.000141215 0.000144887
   crystal router                : 0.000136805 0.000135899 0.000137711
   all reduce                    : 0.000231695 0.000231385 0.000232005
   used all_to_all method: crystal router
   handle bytes (avg, min, max): 1.60548e+07 16054804 16054804
   buffer bytes (avg, min, max): 1.2544e+06 1254400 1254400
setup done: bytes allocated = 873655144
correctness check: maxError = 0.916949

running solver ... done

summary
  MPItasks     : 2
  OMPthreads   : 24
  polyN        : 7
  Nelements    : 8000
  iterations   : 5
  Nrepetitions : 1000
  elapsed time : 38.5381 s
  throughput   : 0.356012 GDOF/s/iter
  bandwidth    : 89.5282 GB/s
```

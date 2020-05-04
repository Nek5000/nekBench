This benchmark solves a 3D inhomogenous Helmholtz equation 
```
lambda0*[A]u + lambda1*[B]u = f
```
on a deformed hexahedral spectral element mesh using Jacobi preconditioned conjuate gradients.
Note, this benchmark is different fomr the old nekBone or CEED BP5 due to diagnoal term [B], the variable coefficients.

# Usage

```
./nekBone <setup ini file>
```
Tuned kernels for the following architectures are available:
* VOLTA (Nvidia Pascal and Volta)
* CPU (generic)	

# Examples

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

gs_setup: 39200 unique labels shared
   pairwise times (avg, min, max): 0.000127995 0.000125408 0.000130582
   crystal router                : 0.000141501 0.00014081 0.000142193
   all reduce                    : 0.000233245 0.000232983 0.000233507
   used all_to_all method: pairwise
   handle bytes (avg, min, max): 1.4173e+07 14173044 14173044
   buffer bytes (avg, min, max): 627200 627200 627200
setup done: bytes allocated = 844163944

running solver ... done

correctness check: maxError = 6.32827e-15

summary
  MPItasks   : 2
  OMPthreads : 24
  polyN      : 7
  Nelements  : 8000
  iterations : 5000
  walltime   : 28.1554 s
  throughput : 0.487295 GDOF/s/iter
  bandwidth  : 122.6 GB/s

timings
  Ax         : 9.51866 s
  gs         : 7.09125 s
  updatePCG  : 4.01666 s
  dotp       : 4.67891 s
```

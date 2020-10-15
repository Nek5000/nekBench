This benchmark solves a 3D inhomogenous Helmholtz equation 
```
lambda0*[A]u + lambda1*[B]u = f
```
or in BP mode
```
[A]u = f
```
on a deformed hexahedral spectral element mesh using preconditioned conjuate gradients.

# Usage

```
./nekBone <setup ini file>
```
Tuned kernels for the following architectures are available:
* VOLTA (Nvidia Pascal and Volta)
* CPU (generic)	

Note, set env-var `OGS_MPI_SUPPORT=1` to enable GPU aware MPI support.  

# Examples

### Single GPU run on Nvidia V100
```
>mpirun -np 1 -bind-to core ./nekBone nekBone.ini

active occa mode: CUDA
BP mode enabled
overlap disabled
Compiling GatherScatter Kernels...done.
gs_setup: 0 unique labels shared
   handle bytes (avg, min, max): 2.77219e+07 27721948 27721948
   buffer bytes (avg, min, max): 0 0 0
gs_setup: 0 unique labels shared
   handle bytes (avg, min, max): 2.77219e+07 27721948 27721948
   buffer bytes (avg, min, max): 0 0 0
setup done: bytes allocated = 1653409376
correctness check: maxError = 1.02763e-07 in 78 iterations

running solver ... done

summary
  MPItasks     : 1
  polyN        : 7
  Nelements    : 8192
  Nfields      : 1
  iterations   : 5000
  Nrepetitions : 1
  elapsed time : 6.71864 s
  throughput   : 2.09109 GDOF/s/iter
  bandwidth    : 599.308 GB/s
  GFLOPS/s     : 393.296
```

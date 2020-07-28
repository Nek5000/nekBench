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

BP mode enabled
overlap disabled
Compiling GatherScatter Kernels...done.
gs_setup: 0 unique labels shared
   handle bytes (avg, min, max): 2.70722e+07 27072220 27072220
   buffer bytes (avg, min, max): 0 0 0
gs_setup: 0 unique labels shared
   handle bytes (avg, min, max): 2.70722e+07 27072220 27072220
   buffer bytes (avg, min, max): 0 0 0
setup done: bytes allocated = 1614529640
correctness check: maxError = 2.99848e-08 in 48 iterations

running solver ... done

summary
  MPItasks     : 1
  polyN        : 7
  Nelements    : 8000
  Nfields      : 1
  iterations   : 5000
  Nrepetitions : 1
  elapsed time : 6.8114 s
  throughput   : 2.01427 GDOF/s/iter
  bandwidth    : 577.291 GB/s
  GFLOPS/s     : 378.847
```

This benchmark solves a 3D inhomogenous Helmholtz equation 
on a deformed hexahedral spectral element mesh using Jacobi preconditioned conjuate gradients.

# Usage

```
./nekBone <setup ini file>
```
Tuned kernels for the following architectures are available:
* VOLTA (Nvidia Pascal and Volta)
* CPU (generic)	

# Examples

### Pure MPI with native CPU kernel

Set in setup.ini
```
[ARCH]
CPU

[THREAD MODEL]
NATIVE+SERIAL
```

Then simply run
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

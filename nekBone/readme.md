This benchmark solves a 3D inhomogenous Helmholtz equation 
on a deformed hexahedral spectral element mesh using Jacobi preconditioned conjuate gradients.
It exposes the principal computational kernel to reveal the essential elements of the algorithmic- architectural coupling that is pertinent to nek5000/nekRS.

# Usage

```
./nekBone <setup ini file>
```
Tuned kernels for the following architectures are available:
* VOLTA (Nvidia Pascal and Volta)
* CPU (generic)	

# Examples
Here a few examples how to run the benchmark

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
OCCA_CXX='g++' OCCA_CXXFLAGS='-O3 -march=native -mtune=native' mpirun -np 48 -bind-to core ./nekBone setup.ini
```

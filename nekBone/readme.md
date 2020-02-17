This benchmark solves a screened 3D Possion equation on a spectral element mesh 
using preconditioned conjuate gradients.

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

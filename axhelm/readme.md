Fist we need to setup some env vars:

>export OCCA_DIR=$HOME/occa
>export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OCCA_DIR/lib

Then simply run

>make

To run the kernel with N=7 and 8192 elements

>OCCA_CUDA_COMPILER_FLAGS='--use_fast_math' ./axhelm 8 8192 CUDA 0
>OCCA_OPENCL_COMPILER_FLAGS='-cl-mad-enable -cl-finite-math-only -cl-fast-relaxed-math' ./axhelm 8 8192 OPENCL 0
>OCCA_CXXFLAGS='-O3 -march=native -mtune=native' mpirun -np 32 -bind-to core ./axhelm 8 256 MPI+NATIVE+SERIAL 0

To run the kernel with N=7 and 8192 elements

>OCCA_CUDA_COMPILER_FLAGS='--use_fast_math' ./matvec 8 8192 CUDA 0
>OCCA_OPENCL_COMPILER_FLAGS='-cl-mad-enable -cl-finite-math-only -cl-fast-relaxed-math' ./matvec 8 8192 OPENCL 0
>OCCA_CXXFLAGS='-O3 -march=native -mtune=native' OMP_NUM_THREADS=32 ./matvec 8 8192 OPENMP 0
>OCCA_CXXFLAGS='-O3 -march=native -mtune=native' mpirun -np 32 ./matvec 8 8192 MPI 0

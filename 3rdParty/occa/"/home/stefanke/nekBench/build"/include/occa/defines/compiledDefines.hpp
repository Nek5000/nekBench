#ifndef OCCA_DEFINES_COMPILEDDEFINES_HEADER
#define OCCA_DEFINES_COMPILEDDEFINES_HEADER

#ifndef OCCA_LINUX_OS
#  define OCCA_LINUX_OS 1
#endif

#ifndef OCCA_MACOS_OS
#  define OCCA_MACOS_OS 2
#endif

#ifndef OCCA_WINDOWS_OS
#  define OCCA_WINDOWS_OS 4
#endif

#ifndef OCCA_WINUX_OS
#  define OCCA_WINUX_OS (OCCA_LINUX_OS | OCCA_WINDOWS_OS)
#endif

#define OCCA_OS             OCCA_LINUX_OS
#define OCCA_USING_VS       0
#define OCCA_UNSAFE         0

#define OCCA_MPI_ENABLED    1
#define OCCA_OPENMP_ENABLED 1
#define OCCA_CUDA_ENABLED   0
#define OCCA_HIP_ENABLED    0
#define OCCA_OPENCL_ENABLED 0
#define OCCA_METAL_ENABLED  0

#define OCCA_BUILD_DIR     "/home/stefanke/nekBench/build"

#endif

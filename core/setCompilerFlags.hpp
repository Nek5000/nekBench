static void setCompilerFlags(occa::device device, occa::properties &kernelInfo)
{
  if(device.mode()=="CUDA"){ // add backend compiler optimization for CUDA
    kernelInfo["compiler_flags"] += " --ftz=true ";
    kernelInfo["compiler_flags"] += " --prec-div=false ";
    kernelInfo["compiler_flags"] += " --prec-sqrt=false ";
    kernelInfo["compiler_flags"] += " --use_fast_math ";
    kernelInfo["compiler_flags"] += " --fmad=false ";
  }

  if(device.mode()=="OpenCL"){ // add backend compiler optimization for OPENCL
    kernelInfo["compiler_flags"] += " -cl-std=CL2.0 ";
    kernelInfo["compiler_flags"] += " -cl-strict-aliasing ";
    kernelInfo["compiler_flags"] += " -cl-mad-enable ";
    kernelInfo["compiler_flags"] += " -cl-no-signed-zeros ";
    kernelInfo["compiler_flags"] += " -cl-unsafe-math-optimizations ";
    kernelInfo["compiler_flags"] += " -cl-fast-relaxed-math ";
  }

  if(device.mode()=="HIP"){ // add backend compiler optimization for HIP
    kernelInfo["compiler_flags"] += " -O3 ";
    kernelInfo["compiler_flags"] += " -ffp-contract=fast ";
    kernelInfo["compiler_flags"] += " -funsafe-math-optimizations ";
    kernelInfo["compiler_flags"] += " -ffast-math ";
  }
}    

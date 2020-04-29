#include "setCompilerFlags.hpp"

static occa::kernel loadAxKernel(occa::device device, const std::string threadModel,
                                 const std::string arch, std::string kernelName,
                                 int N, dlong Nelements){

  int rank = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int Nq = N + 1;
  const int Np = Nq*Nq*Nq;
 
  occa::properties props;
  props["defines"].asObject();

  setCompilerFlags(device, props);
  
  props["defines/p_Nalign"] = USE_OCCA_MEM_BYTE_ALIGN;

  props["defines/p_Nq"] = Nq;
  props["defines/p_Np"] = Np;

  props["defines/p_Nggeo"] = p_Nggeo;
  props["defines/p_G00ID"] = p_G00ID;
  props["defines/p_G01ID"] = p_G01ID;
  props["defines/p_G02ID"] = p_G02ID;
  props["defines/p_G11ID"] = p_G11ID;
  props["defines/p_G12ID"] = p_G12ID;
  props["defines/p_G22ID"] = p_G22ID;
  props["defines/p_GWJID"] = p_GWJID;

  props["defines/dfloat"] = dfloatString;
  props["defines/dlong"]  = dlongString;

  occa::kernel axKernel;

  std::string filename = "kernel/" + arch + "/axhelm";
  for (int r=0;r<2;r++){
    if ((r==0 && rank==0) || (r==1 && rank>0)) {
      if(strstr(threadModel.c_str(), "NATIVE+CUDA")){
        props["okl/enabled"] = false;
        axKernel = device.buildKernel(filename + ".cu", kernelName, props);
        axKernel.setRunDims(Nelements, Nq*Nq);
      } else if(strstr(threadModel.c_str(), "NATIVE+SERIAL") || 
                strstr(threadModel.c_str(), "NATIVE+OPENMP")){
        props["defines/USE_OCCA_MEM_BYTE_ALIGN"] = USE_OCCA_MEM_BYTE_ALIGN;
        props["okl/enabled"] = false;
        axKernel = device.buildKernel(filename + ".c", kernelName, props);
      } else { // fallback is okl
        //std::cout << props << std::endl;
        axKernel = device.buildKernel(filename + ".okl", kernelName, props);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return axKernel;
}

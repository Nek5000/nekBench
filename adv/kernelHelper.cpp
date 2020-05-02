#include "setCompilerFlags.hpp"

static occa::kernel loadKernel(occa::device device, const std::string threadModel,
                               const std::string arch, std::string kernelName,
                               int N, int cubN, dlong Nelements){

  int rank = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int Nq = N + 1;
  const int Np = Nq*Nq*Nq;

  const int cubNq = cubN + 1;
  const int cubNp = cubNq*cubNq*cubNq;
 
  occa::properties props;
  props["defines"].asObject();

  setCompilerFlags(device, props);
  
  props["defines/p_Nalign"] = USE_OCCA_MEM_BYTE_ALIGN;

  props["defines/p_Nq"] = Nq;
  props["defines/p_cubNq"] = cubNq;
  props["defines/p_Np"] = Np;
  props["defines/p_cubNp"] = cubNp;

  props["defines/p_Nvgeo"] = p_Nvgeo;

  props["defines/" "p_RXID"]= RXID;
  props["defines/" "p_SXID"]= SXID;
  props["defines/" "p_TXID"]= TXID;

  props["defines/" "p_RYID"]= RYID;
  props["defines/" "p_SYID"]= SYID;
  props["defines/" "p_TYID"]= TYID;

  props["defines/" "p_RZID"]= RZID;
  props["defines/" "p_SZID"]= SZID;
  props["defines/" "p_TZID"]= TZID;

  props["defines/" "p_JID"]= JID;
  props["defines/" "p_JWID"]= JWID;
  props["defines/" "p_IJWID"]= IJWID;

  props["defines/dfloat"] = dfloatString;
  props["defines/dlong"]  = dlongString;

  occa::kernel kernel;
  
  std::string root(DBP);
  std::string filename = root + "kernel/" + arch + "/adv";
  for (int r=0;r<2;r++){
    if ((r==0 && rank==0) || (r==1 && rank>0)) {
      if(strstr(threadModel.c_str(), "NATIVE+CUDA")){
        props["okl/enabled"] = false;
        kernel = device.buildKernel(filename + ".cu", kernelName, props);
        kernel.setRunDims(Nelements, Nq*Nq);
      } else if(strstr(threadModel.c_str(), "NATIVE+SERIAL") || 
                strstr(threadModel.c_str(), "NATIVE+OPENMP")){
        props["defines/USE_OCCA_MEM_BYTE_ALIGN"] = USE_OCCA_MEM_BYTE_ALIGN;
        props["okl/enabled"] = false;
        kernel = device.buildKernel(filename + ".c", kernelName, props);
      } else { // fallback is okl
        //std::cout << props << std::endl;
        kernel = device.buildKernel(filename + ".okl", kernelName, props);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return kernel;
}

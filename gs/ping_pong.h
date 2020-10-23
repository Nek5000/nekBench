extern "C" {

int pingPongSinglePair(bool dumptofile, int useDevice, occa::device device, MPI_Comm comm);
int multiPairExchange(bool dumptofile, int nmessages, int useDevice, occa::device device, MPI_Comm comm);

}

extern "C" {

int pingPongSinglePair(int useDevice, occa::device device, MPI_Comm comm);
int multiPairExchange(int nmessages, int useDevice, occa::device device, MPI_Comm comm);

}

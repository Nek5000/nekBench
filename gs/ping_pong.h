extern "C" {

int pingPongMulti(int pairs, int useDevice, occa::device device, MPI_Comm comm);
int pingPongSingle(int nmessages, int useDevice, occa::device device, MPI_Comm comm);

}

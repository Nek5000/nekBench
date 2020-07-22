extern "C" {

int pingPongMulti(int pairs, int useDevice, occa::device device, MPI_Comm comm);
int pingPongSingle(int useDevice, occa::device device, MPI_Comm comm);

}

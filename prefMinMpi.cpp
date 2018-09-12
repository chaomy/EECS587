#include <math.h>
#include <mpi.h>
#include <iostream>

using std::cout;
using std::endl;
using std::min;

/*
MPI_Send(
    void* data,
    int count,
    MPI_Datatype datatype,
    int destination,
    int tag,
    MPI_Comm communicator)

MPI_Recv(
    void* data,
    int count,
    MPI_Datatype datatype,
    int source,
    int tag,
    MPI_Comm communicator,
    MPI_Status* status)

MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
*/

void calprefixSum() {
  MPI_Init(NULL, NULL);
  int id, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  cout << "My rank is " << id << " " << p << endl;

  int v = (id - 4) * (id - 4) + 4;
  int x = v;
  int w = v;
  int sp = sqrt(p);

  if (id % sp == 0) {
    // horizontal forward scan */
    w = v;
    MPI_Send(&w, 1, MPI_INT, id + 1, 0, MPI_COMM_WORLD);

    // horizontal backward scan */
    MPI_Recv(&x, 1, MPI_INT, id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    w = min(w, x);
  } else if (id % sp == (sp - 1)) {
    // horizontal forward scan
    MPI_Recv(&w, 1, MPI_INT, id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    w = min(w, v);

    // vertical scan
    if (id == (p - 1)) {
      MPI_Recv(&x, 1, MPI_INT, id - sp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      w = min(w, x);
    } else if (id == sp - 1) {
      x = w;
      MPI_Send(&x, 1, MPI_INT, id + sp, 0, MPI_COMM_WORLD);
    } else {
      MPI_Recv(&x, 1, MPI_INT, id - sp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      w = min(w, x);
      int t = x;
      x = w;
      MPI_Send(&x, 1, MPI_INT, id + sp, 0, MPI_COMM_WORLD);
      x = t;
    }
    MPI_Send(&x, 1, MPI_INT, id - 1, 0, MPI_COMM_WORLD);
  } else {
    /* horizontal forward scan */
    MPI_Recv(&w, 1, MPI_INT, id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    w = min(w, v);
    MPI_Send(&w, 1, MPI_INT, id + 1, 0, MPI_COMM_WORLD);

    /* horizontal backward scan */
    MPI_Recv(&x, 1, MPI_INT, id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    w = min(w, x);
    MPI_Send(&x, 1, MPI_INT, id - 1, 0, MPI_COMM_WORLD);
  }

  cout << "My rank is " << id << " " << w << endl;
}

int main() {
  calprefixSum();
  return 0;
}
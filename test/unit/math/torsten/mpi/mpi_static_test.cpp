#ifdef TORSTEN_MPI

#include <stan/math/torsten/mpi/envionment.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <vector>

using torsten::mpi::Communicator;
using torsten::mpi::Session;

TEST(mpi_test, inter_intra_comms_3) {
  const Communicator world_comm(MPI_COMM_STAN);
  const Communicator& inter_comm(Session::inter_chain_comm(3));
  const Communicator& intra_comm(Session::intra_chain_comm(3));
  if (world_comm.size() == 3) {
    switch (world_comm.rank()) {
    case 0:
      EXPECT_EQ(inter_comm.rank(), 0);
      EXPECT_EQ(intra_comm.rank(), 0);
      break;
    case 1:
      EXPECT_EQ(inter_comm.rank(), 1);
      EXPECT_EQ(intra_comm.rank(), 0);
      break;
    case 2:
      EXPECT_EQ(inter_comm.rank(), 2);
      EXPECT_EQ(intra_comm.rank(), 0);
      break;
    }
  } else if (world_comm.size() == 4) {
    switch (world_comm.rank()) {
    case 0:
      EXPECT_EQ(inter_comm.rank(), 0);
      EXPECT_EQ(intra_comm.rank(), 0);
      break;
    case 1:
      EXPECT_EQ(inter_comm.rank(), -1);
      EXPECT_EQ(intra_comm.rank(), 1);
      break;
    case 2:
      EXPECT_EQ(inter_comm.rank(), 1);
      EXPECT_EQ(intra_comm.rank(), 0);
      break;
    case 3:
      EXPECT_EQ(inter_comm.rank(), 2);
      EXPECT_EQ(intra_comm.rank(), 0);
      break;
    }
  } else if (world_comm.size() == 5) {
    switch (world_comm.rank()) {
    case 0:
      EXPECT_EQ(inter_comm.rank(), 0);
      EXPECT_EQ(intra_comm.rank(), 0);
      break;
    case 1:
      EXPECT_EQ(inter_comm.rank(), -1);
      EXPECT_EQ(intra_comm.rank(), 1);
      break;
    case 2:
      EXPECT_EQ(inter_comm.rank(), 1);
      EXPECT_EQ(intra_comm.rank(), 0);
      break;
    case 3:
      EXPECT_EQ(inter_comm.rank(), -1);
      EXPECT_EQ(intra_comm.rank(), 1);
      break;
    case 4:
      EXPECT_EQ(inter_comm.rank(), 2);
      EXPECT_EQ(intra_comm.rank(), 0);
      break;
    }
  } else if (world_comm.size() == 6) {
    switch (world_comm.rank()) {
    case 0:
      EXPECT_EQ(inter_comm.rank(), 0);
      EXPECT_EQ(intra_comm.rank(), 0);
      break;
    case 1:
      EXPECT_EQ(inter_comm.rank(), -1);
      EXPECT_EQ(intra_comm.rank(), 1);
      break;
    case 2:
      EXPECT_EQ(inter_comm.rank(), 1);
      EXPECT_EQ(intra_comm.rank(), 0);
      break;
    case 3:
      EXPECT_EQ(inter_comm.rank(), -1);
      EXPECT_EQ(intra_comm.rank(), 1);
      break;
    case 4:
      EXPECT_EQ(inter_comm.rank(), 2);
      EXPECT_EQ(intra_comm.rank(), 0);
      break;
    case 5:
      EXPECT_EQ(inter_comm.rank(), -1);
      EXPECT_EQ(intra_comm.rank(), 1);
      break;
    }
  } else if (world_comm.size() == 7) {
    switch (world_comm.rank()) {
    case 0:
      EXPECT_EQ(inter_comm.rank(), 0);
      EXPECT_EQ(intra_comm.rank(), 0);
      break;
    case 1:
      EXPECT_EQ(inter_comm.rank(), -1);
      EXPECT_EQ(intra_comm.rank(), 1);
      break;
    case 2:
      EXPECT_EQ(inter_comm.rank(), -1);
      EXPECT_EQ(intra_comm.rank(), 2);
      break;
    case 3:
      EXPECT_EQ(inter_comm.rank(), 1);
      EXPECT_EQ(intra_comm.rank(), 0);
      break;
    case 4:
      EXPECT_EQ(inter_comm.rank(), -1);
      EXPECT_EQ(intra_comm.rank(), 1);
      break;
    case 5:
      EXPECT_EQ(inter_comm.rank(), 2);
      EXPECT_EQ(intra_comm.rank(), 0);
      break;
    case 6:
      EXPECT_EQ(inter_comm.rank(), -1);
      EXPECT_EQ(intra_comm.rank(), 1);
      break;
    }
  }

  const Communicator& inter_comm_1(Session::inter_chain_comm(3));
  const Communicator& intra_comm_1(Session::intra_chain_comm(3));
}


#endif

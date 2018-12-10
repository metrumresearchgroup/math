#ifdef TORSTEN_MPI

#include <gtest/gtest.h>
#include <stan/math/torsten/mpi/is_mine.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <boost/mpi.hpp>

TEST(torsten_mpi_test, is_mine) {

  MPI_Init(NULL, NULL);
  boost::mpi::communicator world;

  bool mine;
  int np;
  
  // only run with mpirun -np 3
  if (world.size() == 3) {

    // np > world size
    np = 10;
    // rank 0 has individual 0, 1, 2, 3
    for (int i = 0; i < 4; ++i) {
      mine = torsten::mpi::is_mine(world, i, np);
      if(world.rank() == 0) {
        EXPECT_EQ(mine, true);
      } else {
        EXPECT_EQ(mine, false);
      }    
    }

    // rank 1 has individual 4, 5, 6
    for (int i = 4; i < 7; ++i) {
      mine = torsten::mpi::is_mine(world, i, np);
      if(world.rank() == 1) {
        EXPECT_EQ(mine, true);
      } else {
        EXPECT_EQ(mine, false);
      }    
    }

    // rank 2 has individual 7, 8, 9
    for (int i = 7; i < 10; ++i) {
      mine = torsten::mpi::is_mine(world, i, np);
      if(world.rank() == 2) {
        EXPECT_EQ(mine, true);
      } else {
        EXPECT_EQ(mine, false);
      }
    }


    // np < world size
    np = 2;
    mine = torsten::mpi::is_mine(world, 0, np);
    if(world.rank() == 0) {
      EXPECT_EQ(mine, true);
    } else {
      EXPECT_EQ(mine, false);
    }
    mine = torsten::mpi::is_mine(world, 1, np);
    if(world.rank() == 1) {
      EXPECT_EQ(mine, true);
    } else {
      EXPECT_EQ(mine, false);
    }
    mine = torsten::mpi::is_mine(world, 2, np);
    EXPECT_EQ(mine, false);

    // np = world size
    np = 3;
    for (int i = 0; i < np; ++i) {
      mine = torsten::mpi::is_mine(world, i, np);      
      if(world.rank() == i) {
        EXPECT_EQ(mine, true); 
      } else {
        EXPECT_EQ(mine, false); 
      }
    }
  }

  MPI_Finalize();
}

#endif

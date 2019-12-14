#ifndef STAN_MATH_TORSTEN_MPI_SLAVE_AND_CLEANER_HPP
#define STAN_MATH_TORSTEN_MPI_SLAVE_AND_CLEANER_HPP

#include <stan/math/torsten/mpi/dynamic_load.hpp>
#include <exception>

namespace torsten {
  namespace mpi {

    struct slave_dismiss {
      const Communicator& pmx_comm;

      slave_dismiss(const Communicator& comm_in) :
        pmx_comm(comm_in)
      {}

      virtual const char* what() const throw() {
        return "Torsten: MPI slave dismissed.";
      }
    };

#ifdef TORSTEN_MPI_DYN
    /*
     * manages session's dynamic load balance by starting
     * slaves in constructor and kill slaves in destructor.
     */
    struct slave_and_cleaner {
      const Communicator& pmx_comm;

      /*
       * constructor traps slaves in loop until released by
       * cleaner in the destructor. After release, slaves
       * should exit by throwing a special exception, caught at
       * initialization stage of stan::lang.
       */
      slave_and_cleaner(const Communicator& comm_in) : pmx_comm(comm_in) {
        if (comm_in.rank() > 0) {
          PMXDynamicLoad<TORSTEN_MPI_DYN_SLAVE> load(comm_in);
          std::cout << "Torsten: " << "MPI slave " << comm_in.rank() <<  " initialized" << "\n";
          load.slave();          
          throw slave_dismiss(pmx_comm);
        }
      }

      /*
       * destructor send kill tag to salves through the
       * destructor of the cleaner.
       */
      ~slave_and_cleaner() {
        PMXDynamicLoad<TORSTEN_MPI_DYN_CLEANER> cleaner(pmx_comm);
      }
    };
#endif
  }
}

#endif

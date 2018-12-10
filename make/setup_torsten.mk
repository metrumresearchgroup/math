ifdef TORSTEN_MPI
  LIBMPI = $(BOOST_LIB)/libboost_serialization$(DLL) $(BOOST_LIB)/libboost_mpi$(DLL) $(MATH)bin/math/prim/arr/functor/mpi_cluster_inst.o
  CXXFLAGS_MPI = -DTORSTEN_MPI
  LDFLAGS_MPI ?= -Wl,-lboost_mpi -Wl,-lboost_serialization -Wl,-L,"$(BOOST_LIB_ABS)" -Wl,-rpath,"$(BOOST_LIB_ABS)"
endif


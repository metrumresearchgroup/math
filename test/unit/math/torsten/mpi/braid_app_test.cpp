#include <stan/math/rev/mat.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/torsten/pmx_ode_test_fixture.hpp>
#include <gtest/gtest.h>
#include <stan/math/torsten/mpi/cvodes_braid.hpp>
#include <iostream>
#include <vector>
#include <memory>

using torsten::dsolve::PMXCvodesFwdSystem;
using torsten::dsolve::PMXCvodesIntegrator;
using torsten::dsolve::PMXCvodesService;
using torsten::PMXCvodesSensMethod;
using torsten::mpi::CVBraidVec;
using torsten::mpi::CVBraidApp;
using stan::math::var;

TEST_F(TorstenOdeTest_sho, braid_app) {
  int rank, size;
  MPI_Init(NULL, NULL);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  PMXCvodesIntegrator solver(rtol, atol, 1000);

  using Ode = PMXCvodesFwdSystem<F, double, double, double, CV_BDF, torsten::CSDA>;

  PMXCvodesService<typename Ode::Ode> s1(2, 1);
  Ode ode{s1, f, t0, ts, y0, theta, x_r, x_i, msgs};

  auto mem       = ode.mem();
  auto y         = ode.nv_y();
  auto ys        = ode.nv_ys();
  const size_t n = ode.n();
  const size_t ns= ode.ns();

  CHECK_SUNDIALS_CALL(CVodeReInit(mem, ode.t0(), y));
  CHECK_SUNDIALS_CALL(CVodeSStolerances(mem, rtol, atol));
  CHECK_SUNDIALS_CALL(CVodeSetUserData(mem, ode.to_user_data()));
  CHECK_SUNDIALS_CALL(CVodeSetMaxNumSteps(mem, max_num_steps));
  CHECK_SUNDIALS_CALL(CVodeSetMaxErrTestFails(mem, 20));
  CHECK_SUNDIALS_CALL(CVodeSetMaxConvFails(mem, 30));
  CHECK_SUNDIALS_CALL(CVDlsSetJacFn(mem, torsten::dsolve::cvodes_jac<Ode>()));

  // CHECK_SUNDIALS_CALL(CVode(mem, ts[1], y, &t0, CV_NORMAL));
  CVBraidApp app(mem, ode.to_user_data(), ode.nv_y(), MPI_COMM_WORLD, 0.0, 1000.0, 1000);

  BraidUtil util;

  // Run Braid simulation.
  BraidCore core(comm, &app);
  core.SetMaxLevels(3);

  // opts.SetBraidCoreOptions(core);
  core.Drive();

  MPI_Finalize();
}

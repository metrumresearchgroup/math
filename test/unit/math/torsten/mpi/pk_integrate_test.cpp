#include <stan/math.hpp>
#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/torsten/dsolve/pk_cvodes_fwd_system.hpp>
#include <stan/math/torsten/dsolve/pk_cvodes_integrator.hpp>
#include <stan/math/torsten/dsolve/pk_integrate_ode_adams.hpp>
#include <stan/math/torsten/dsolve/pk_integrate_ode_bdf.hpp>
#include <test/unit/math/torsten/pk_ode_test_fixture.hpp>
#include <test/unit/math/prim/arr/functor/harmonic_oscillator.hpp>
#include <stan/math/rev/mat/functor/integrate_ode_bdf.hpp>
#include <nvector/nvector_serial.h>
#include <boost/mpi.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <vector>
#include <chrono>
#include <ctime>


TEST_F(TorstenOdeTest_sho, cvodes_ivp_system_mpi) {

  using torsten::dsolve::PKCvodesFwdSystem;
  using torsten::dsolve::PKCvodesIntegrator;
  using torsten::dsolve::PKCvodesService;
  using torsten::PkCvodesSensMethod;
  using stan::math::integrate_ode_bdf;
  using torsten::dsolve::pk_integrate_ode_bdf;
  using std::vector;

  vector<vector<double> > ts_m {ts, ts};
  vector<vector<double> > y0_m {y0, y0};
  vector<vector<double> > theta_m {theta, theta};
  vector<vector<double> > x_r_m {x_r, x_r};
  vector<vector<int> > x_i_m {x_i, x_i};

  vector<vector<double> > y = integrate_ode_bdf(f, y0, t0, ts, theta , x_r, x_i); // NOLINT
  vector<vector<vector<double> > > y_m = pk_integrate_ode_bdf(f, y0_m, t0, ts_m, theta_m , x_r_m, x_i_m);
  EXPECT_EQ(y_m.size(), theta_m.size());
  for (size_t i = 0; i < y_m.size(); ++i) {
    EXPECT_EQ(y_m[i].size(), ts.size());
    for (size_t j = 0; j < ts.size(); ++j) {
      EXPECT_EQ(y_m[i][j].size(), y0.size());
      for (size_t k = 0; k < y0.size(); ++k) {
        EXPECT_FLOAT_EQ(y_m[i][j][k], y[j][k]);
      }
    }
  }
}

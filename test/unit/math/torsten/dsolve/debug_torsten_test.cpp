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
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <chrono>
#include <ctime>

TEST_F(TorstenOdeTest_lorenz, fwd_sensitivity_theta_AD) {
  using torsten::dsolve::PKCvodesSystem;
  using torsten::dsolve::PKCvodesFwdSystem;
  using torsten::dsolve::PKCvodesIntegrator;
  using torsten::dsolve::PKCvodesService;
  using torsten::PkCvodesSensMethod;

  using stan::math::var;

  PKCvodesIntegrator solver(rtol, atol, 1e8);
  std::vector<var> theta_var = stan::math::to_var(theta);
  std::vector<double> ts0(ts);
  ts0.push_back(400);

  // y1 = stan::math::integrate_ode_adams(f, y0, t0, ts, theta_var, x_r, x_i);

  auto y_a = torsten::dsolve::pk_integrate_ode_bdf(f, y0, t0, ts0, theta_var, x_r, x_i);
}

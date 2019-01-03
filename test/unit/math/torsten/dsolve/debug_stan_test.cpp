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

TEST_F(TorstenOdeTest_neutropenia, fwd_sens_theta_performance_bdf) {
  using stan::math::var;

  std::vector<var> theta_var = stan::math::to_var(theta);
  std::vector<std::vector<var> > y1, y2;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> y1_elapsed, y2_elapsed;
 
  std::vector<double> ts0(ts);
  ts0.push_back(400);

  start = std::chrono::system_clock::now();
  y2 = stan::math::integrate_ode_bdf(f, y0, t0, ts0, theta_var, x_r, x_i);
  end = std::chrono::system_clock::now();
  y2_elapsed = end - start;
  std::cout << "stan    solver elapsed time: " << y2_elapsed.count() << "s\n";
}

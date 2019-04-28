#include <stan/math/rev/arr.hpp>
#include <gtest/gtest.h>
#include <stan/math/torsten/dsolve/pmx_integrate_ode_rk45.hpp>
#include <stan/math.hpp>
#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/torsten/dsolve/pmx_cvodes_fwd_system.hpp>
#include <stan/math/torsten/dsolve/pmx_cvodes_integrator.hpp>
#include <stan/math/torsten/dsolve/pmx_integrate_ode_adams.hpp>
#include <stan/math/torsten/dsolve/pmx_integrate_ode_bdf.hpp>
#include <test/unit/math/torsten/pmx_ode_test_fixture.hpp>
#include <test/unit/math/prim/arr/functor/harmonic_oscillator.hpp>
#include <stan/math/rev/mat/functor/integrate_ode_bdf.hpp>
#include <nvector/nvector_serial.h>
#include <test/unit/util.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <chrono>
#include <ctime>

using stan::math::integrate_ode_rk45;
using torsten::dsolve::pmx_integrate_ode_rk45;
using stan::math::var;

TEST_F(TorstenOdeTest_sho, odeint_rk45_ivp_system) {
  std::vector<std::vector<double> > y1(integrate_ode_rk45(f, y0, t0, ts, theta , x_r, x_i));
  std::vector<std::vector<double> > y2(pmx_integrate_ode_rk45(f, y0, t0, ts, theta , x_r, x_i));
  torsten::test::test_val(y1, y2);
}

TEST_F(TorstenOdeTest_lorenz, odeint_rk45_ivp_system) {
  std::vector<std::vector<double> > y1(integrate_ode_rk45(f, y0, t0, ts, theta , x_r, x_i));
  std::vector<std::vector<double> > y2(pmx_integrate_ode_rk45(f, y0, t0, ts, theta , x_r, x_i));
  torsten::test::test_val(y1, y2);
}

TEST_F(TorstenOdeTest_chem, odeint_rk45_ivp_system) {
  std::vector<std::vector<double> > y1(integrate_ode_rk45(f, y0, t0, ts, theta , x_r, x_i));
  std::vector<std::vector<double> > y2(pmx_integrate_ode_rk45(f, y0, t0, ts, theta , x_r, x_i));
  torsten::test::test_val(y1, y2);
}

TEST_F(TorstenOdeTest_chem, odeint_rk45_fwd_sensitivity_theta) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> theta_var2 = stan::math::to_var(theta);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.E-8, 1.E-5);
}

TEST_F(TorstenOdeTest_lorenz, odeint_rk45_fwd_sensitivity_theta) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> theta_var2 = stan::math::to_var(theta);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.E-8, 1.E-5);
}

TEST_F(TorstenOdeTest_sho, odeint_rk45_fwd_sensitivity_theta) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> theta_var2 = stan::math::to_var(theta);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.E-8, 1.E-5);
}

TEST_F(TorstenOdeTest_chem, odeint_rk45_fwd_sensitivity_y0) {
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0_var2, t0, ts, theta, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 1.E-8, 1.E-5);
}

TEST_F(TorstenOdeTest_lorenz, odeint_rk45_fwd_sensitivity_y0) {
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0_var2, t0, ts, theta, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 1.E-8, 1.E-5);
}

TEST_F(TorstenOdeTest_sho, odeint_rk45_fwd_sensitivity_y0) {
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0_var2, t0, ts, theta, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 1.E-8, 1.E-5);
}

TEST_F(TorstenOdeTest_sho, fwd_sensitivity_theta_y0) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> theta_var2 = stan::math::to_var(theta);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0_var2, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 1.E-8, 1.E-5);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.E-8, 1.E-5);
}

TEST_F(TorstenOdeTest_lorenz, fwd_sensitivity_theta_y0) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> theta_var2 = stan::math::to_var(theta);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0_var2, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 1.E-8, 1.E-5);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.E-8, 1.E-5);
}

TEST_F(TorstenOdeTest_chem, fwd_sensitivity_theta_y0) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> theta_var2 = stan::math::to_var(theta);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0_var2, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 1.E-8, 1.E-5);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.E-8, 1.E-5);
}

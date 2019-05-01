#include <stan/math/rev/arr.hpp>
#include <gtest/gtest.h>
#include <stan/math/torsten/dsolve/pmx_integrate_ode_rk45.hpp>
#include <stan/math.hpp>
#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
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

using stan::math::integrate_ode_rk45;
using torsten::pmx_integrate_ode_rk45;
using stan::math::var;

TEST_F(TorstenOdeTest_sho, odeint_rk45_ivp_system) {
  std::vector<std::vector<double> > y1(integrate_ode_rk45(f, y0, t0, ts, theta , x_r, x_i));
  std::vector<std::vector<double> > y2(pmx_integrate_ode_rk45(f, y0, t0, ts, theta , x_r, x_i));
  torsten::test::test_val(y1, y2);
}

TEST_F(TorstenOdeTest_sho, odeint_rk45_ivp_system_matrix_result) {
  std::vector<std::vector<double> > y1(integrate_ode_rk45(f, y0, t0, ts, theta, x_r, x_i, msgs, atol, rtol, max_num_steps));

  using Ode = dsolve::PMXOdeintSystem<harm_osc_ode_fun, double, double, double>;
  dsolve::PMXOdeService<Ode, dsolve::Odeint> serv(y0.size(), theta.size());
  Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, msgs};
  using scheme_t = boost::numeric::odeint::runge_kutta_dopri5<std::vector<double>, double, std::vector<double>, double>;
  dsolve::PMXOdeintIntegrator<scheme_t> solver(rtol, atol, max_num_steps);
  Eigen::MatrixXd y2 = solver.integrate<Ode, false>(ode);

  torsten::test::test_val(y1, y2);
}

TEST_F(TorstenOdeTest_lorenz, odeint_rk45_ivp_system) {
  std::vector<std::vector<double> > y1(integrate_ode_rk45(f, y0, t0, ts, theta , x_r, x_i));
  std::vector<std::vector<double> > y2(pmx_integrate_ode_rk45(f, y0, t0, ts, theta , x_r, x_i));
  torsten::test::test_val(y1, y2);
}

TEST_F(TorstenOdeTest_lorenz, odeint_rk45_ivp_system_matrix_result) {
 std::vector<std::vector<double> > y1(integrate_ode_rk45(f, y0, t0, ts, theta, x_r, x_i, msgs, atol, rtol, max_num_steps));

  using Ode = dsolve::PMXOdeintSystem<lorenz_ode_fun, double, double, double>;
  dsolve::PMXOdeService<Ode, dsolve::Odeint> serv(y0.size(), theta.size());
  Ode ode{serv, f, t0, ts, y0, theta, x_r, x_i, msgs};
  using scheme_t = boost::numeric::odeint::runge_kutta_dopri5<std::vector<double>, double, std::vector<double>, double>;
  dsolve::PMXOdeintIntegrator<scheme_t> solver(rtol, atol, max_num_steps);
  Eigen::MatrixXd y2 = solver.integrate<Ode, false>(ode);

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
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.E-10, 1.E-6);
}

// TEST_F(TorstenOdeTest_chem, odeint_rk45_fwd_sensitivity_theta_matrix_result) {
//   std::vector<var> theta_var1 = stan::math::to_var(theta);
//   std::vector<var> theta_var2 = stan::math::to_var(theta);

//   std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0, t0, ts, theta_var1, x_r, x_i);
//   std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0, t0, ts, theta_var2, x_r, x_i);
//   torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.E-10, 1.E-8);
// }

TEST_F(TorstenOdeTest_lorenz, odeint_rk45_fwd_sensitivity_theta) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> theta_var2 = stan::math::to_var(theta);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.E-10, 1.E-8);
}

TEST_F(TorstenOdeTest_sho, odeint_rk45_fwd_sensitivity_theta) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> theta_var2 = stan::math::to_var(theta);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.E-10, 1.E-8);
}

TEST_F(TorstenOdeTest_chem, odeint_rk45_fwd_sensitivity_y0) {
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0_var2, t0, ts, theta, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 1.E-10, 1.E-8);
}

TEST_F(TorstenOdeTest_lorenz, odeint_rk45_fwd_sensitivity_y0) {
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0_var2, t0, ts, theta, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 1.E-10, 1.E-8);
}

TEST_F(TorstenOdeTest_sho, odeint_rk45_fwd_sensitivity_y0) {
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0_var2, t0, ts, theta, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 1.E-10, 1.E-8);
}

TEST_F(TorstenOdeTest_sho, fwd_sensitivity_theta_y0) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> theta_var2 = stan::math::to_var(theta);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0_var2, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 1.E-10, 1.E-8);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.E-10, 1.E-8);
}

TEST_F(TorstenOdeTest_lorenz, fwd_sensitivity_theta_y0) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> theta_var2 = stan::math::to_var(theta);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0_var2, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 1.E-10, 1.E-8);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.E-10, 1.E-8);
}

TEST_F(TorstenOdeTest_chem, fwd_sensitivity_theta_y0) {
  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> y0_var1 = stan::math::to_var(y0);
  std::vector<var> theta_var2 = stan::math::to_var(theta);
  std::vector<var> y0_var2 = stan::math::to_var(y0);

  std::vector<std::vector<stan::math::var>> y1 = integrate_ode_rk45(f, y0_var1, t0, ts, theta_var1, x_r, x_i);
  std::vector<std::vector<stan::math::var>> y2 = pmx_integrate_ode_rk45(f, y0_var2, t0, ts, theta_var2, x_r, x_i);
  torsten::test::test_grad(y0_var1, y0_var2, y1, y2, 1.E-10, 1.E-8);
  torsten::test::test_grad(theta_var1, theta_var2, y1, y2, 1.E-10, 1.E-8);
}

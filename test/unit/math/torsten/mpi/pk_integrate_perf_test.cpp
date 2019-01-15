#ifdef TORSTEN_MPI

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

TEST_F(TorstenOdeTest_chem, fwd_sensitivity_theta_bdf_mpi) {
  using torsten::dsolve::PKCvodesFwdSystem;
  using torsten::dsolve::pk_integrate_ode_bdf;
  using stan::math::var;
  using std::vector;

  torsten::mpi::init();

  // size of population
  const int np = 1000;
  std::vector<double> ts0 {ts};

  vector<var> theta_var = stan::math::to_var(theta);

  vector<vector<double> > ts_m (np, ts0);
  vector<vector<double> > y0_m (np, y0);
  vector<vector<var> > theta_var_m (np, theta_var);
  vector<vector<double> > x_r_m (np, x_r);
  vector<vector<int> > x_i_m (np, x_i);

  vector<Eigen::Matrix<var, -1, -1> > y_m = pk_integrate_ode_bdf(f, y0_m, t0, ts_m, theta_var_m , x_r_m, x_i_m);
}

TEST_F(TorstenOdeTest_lorenz, fwd_sensitivity_theta_bdf_mpi) {
  using torsten::dsolve::PKCvodesFwdSystem;
  using torsten::dsolve::pk_integrate_ode_bdf;
  using stan::math::var;
  using std::vector;

  torsten::mpi::init();

  // size of population
  const int np = 1000;
  std::vector<double> ts0 {ts};

  vector<var> theta_var = stan::math::to_var(theta);

  vector<vector<double> > ts_m (np, ts0);
  vector<vector<double> > y0_m (np, y0);
  vector<vector<var> > theta_var_m (np, theta_var);
  vector<vector<double> > x_r_m (np, x_r);
  vector<vector<int> > x_i_m (np, x_i);

  vector<Eigen::Matrix<var, -1, -1> > y_m = pk_integrate_ode_bdf(f, y0_m, t0, ts_m, theta_var_m , x_r_m, x_i_m);
}

TEST_F(TorstenOdeTest_neutropenia, fwd_sensitivity_theta_bdf_mpi) {
  using torsten::dsolve::PKCvodesFwdSystem;
  using torsten::dsolve::pk_integrate_ode_bdf;
  using stan::math::var;
  using std::vector;

  torsten::mpi::init();

  // size of population
  const int np = 1000;
  std::vector<double> ts0 {ts};

  vector<var> theta_var = stan::math::to_var(theta);

  vector<vector<double> > ts_m (np, ts0);
  vector<vector<double> > y0_m (np, y0);
  vector<vector<var> > theta_var_m (np, theta_var);
  vector<vector<double> > x_r_m (np, x_r);
  vector<vector<int> > x_i_m (np, x_i);

  vector<Eigen::Matrix<var, -1, -1> > y_m = pk_integrate_ode_bdf(f, y0_m, t0, ts_m, theta_var_m , x_r_m, x_i_m);
}

TEST_F(TorstenOdeTest_chem, fwd_sensitivity_theta_adams_mpi) {
  using torsten::dsolve::PKCvodesFwdSystem;
  using torsten::dsolve::pk_integrate_ode_adams;
  using stan::math::var;
  using std::vector;

  torsten::mpi::init();

  // size of population
  const int np = 1000;
  std::vector<double> ts0 {ts};

  vector<var> theta_var = stan::math::to_var(theta);

  vector<vector<double> > ts_m (np, ts0);
  vector<vector<double> > y0_m (np, y0);
  vector<vector<var> > theta_var_m (np, theta_var);
  vector<vector<double> > x_r_m (np, x_r);
  vector<vector<int> > x_i_m (np, x_i);

  vector<Eigen::Matrix<var, -1, -1> > y_m = pk_integrate_ode_adams(f, y0_m, t0, ts_m, theta_var_m , x_r_m, x_i_m);
}

TEST_F(TorstenOdeTest_lorenz, fwd_sensitivity_theta_adams_mpi) {
  using torsten::dsolve::PKCvodesFwdSystem;
  using torsten::dsolve::pk_integrate_ode_adams;
  using stan::math::var;
  using std::vector;

  torsten::mpi::init();

  // size of population
  const int np = 1000;
  std::vector<double> ts0 {ts};

  vector<var> theta_var = stan::math::to_var(theta);

  vector<vector<double> > ts_m (np, ts0);
  vector<vector<double> > y0_m (np, y0);
  vector<vector<var> > theta_var_m (np, theta_var);
  vector<vector<double> > x_r_m (np, x_r);
  vector<vector<int> > x_i_m (np, x_i);

  vector<Eigen::Matrix<var, -1, -1> > y_m = pk_integrate_ode_adams(f, y0_m, t0, ts_m, theta_var_m , x_r_m, x_i_m);
}

TEST_F(TorstenOdeTest_neutropenia, fwd_sensitivity_theta_adams_mpi) {
  using torsten::dsolve::PKCvodesFwdSystem;
  using torsten::dsolve::pk_integrate_ode_adams;
  using stan::math::var;
  using std::vector;

  torsten::mpi::init();

  // size of population
  const int np = 1000;
  std::vector<double> ts0 {ts};

  vector<var> theta_var = stan::math::to_var(theta);

  vector<vector<double> > ts_m (np, ts0);
  vector<vector<double> > y0_m (np, y0);
  vector<vector<var> > theta_var_m (np, theta_var);
  vector<vector<double> > x_r_m (np, x_r);
  vector<vector<int> > x_i_m (np, x_i);

  vector<Eigen::Matrix<var, -1, -1> > y_m = pk_integrate_ode_adams(f, y0_m, t0, ts_m, theta_var_m , x_r_m, x_i_m);
}

#endif
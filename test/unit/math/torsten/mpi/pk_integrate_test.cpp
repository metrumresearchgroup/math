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

TEST_F(TorstenOdeTest_chem, cvodes_ivp_system_bdf_mpi) {
  using torsten::dsolve::PKCvodesFwdSystem;
  using stan::math::integrate_ode_bdf;
  using torsten::dsolve::pk_integrate_ode_bdf;
  using std::vector;

  const int np = 1000;
  vector<vector<double> > ts_m(np, ts);
  vector<vector<double> > y0_m (np, y0);
  vector<vector<double> > theta_m (np, theta);
  vector<vector<double> > x_r_m (np, x_r);
  vector<vector<int> > x_i_m (np, x_i);

  torsten::mpi::init();

  vector<vector<double> > y = integrate_ode_bdf(f, y0, t0, ts, theta , x_r, x_i); // NOLINT
  vector<Eigen::MatrixXd> y_m = pk_integrate_ode_bdf(f, y0_m, t0, ts_m, theta_m , x_r_m, x_i_m);
  EXPECT_EQ(y_m.size(), theta_m.size());
  for (size_t i = 0; i < y_m.size(); ++i) {
    EXPECT_EQ(y_m[i].rows(), ts.size());
    EXPECT_EQ(y_m[i].cols(), y0.size());
    for (size_t j = 0; j < ts.size(); ++j) {
      for (size_t k = 0; k < y0.size(); ++k) {
        EXPECT_FLOAT_EQ(y_m[i](j, k), y[j][k]);
      }
    }
  }
}

TEST_F(TorstenOdeTest_chem, fwd_sensitivity_theta_AD_bdf_mpi) {
  using torsten::dsolve::PKCvodesFwdSystem;
  using torsten::dsolve::pk_integrate_ode_bdf;
  using stan::math::var;
  using std::vector;

  torsten::mpi::init();

  std::vector<var> theta_var1 = stan::math::to_var(theta);
  std::vector<var> theta_var2 = stan::math::to_var(theta);
  theta_var2[0] = 1.2 * theta_var2[0];

  vector<vector<double> > ts_m {ts, ts};
  vector<vector<double> > y0_m {y0, y0};
  vector<vector<var> > theta_var_m {theta_var1, theta_var2};
  vector<vector<double> > x_r_m {x_r, x_r};
  vector<vector<int> > x_i_m {x_i, x_i};

  vector<vector<var> > y1 = stan::math::integrate_ode_bdf(f, y0, t0, ts, theta_var1, x_r, x_i);
  vector<vector<var> > y2 = stan::math::integrate_ode_bdf(f, y0, t0, ts, theta_var2, x_r, x_i);
  vector<Eigen::Matrix<var, -1, -1> > y_m = pk_integrate_ode_bdf(f, y0_m, t0, ts_m, theta_var_m , x_r_m, x_i_m);

  // y_m[0]
  for (int j = 0; j < ts.size(); ++j) {
    for (int k = 0; k < y0.size(); ++k) {
      EXPECT_FLOAT_EQ(y_m[0](j, k).val(), y1[j][k].val());
      std::vector<double> g, g1;
      stan::math::set_zero_all_adjoints();
      y_m[0](j, k).grad(theta_var1, g);
      stan::math::set_zero_all_adjoints();
      y1[j][k].grad(theta_var1, g1);
      for (int l = 0 ; l < theta.size(); ++l) {
        EXPECT_NEAR(g[l], g1[l], 1e-7);
      }
    }
  }

  // y_m[1]
  for (int j = 0; j < ts.size(); ++j) {
    for (int k = 0; k < y0.size(); ++k) {
      EXPECT_FLOAT_EQ(y_m[1](j, k).val(), y2[j][k].val());
      std::vector<double> g, g1;
      stan::math::set_zero_all_adjoints();
      y_m[1](j, k).grad(theta_var2, g);
      stan::math::set_zero_all_adjoints();
      y2[j][k].grad(theta_var2, g1);
      for (int l = 0 ; l < theta.size(); ++l) {
        EXPECT_NEAR(g[l], g1[l], 1e-7);
      }
    }
  }
}

TEST_F(TorstenOdeTest_chem, fwd_sensitivity_theta_AD_bdf_mpi_performance) {
  using torsten::dsolve::PKCvodesFwdSystem;
  using torsten::dsolve::pk_integrate_ode_bdf;
  using stan::math::var;
  using std::vector;

  torsten::mpi::init();

  // size of population
  const int np = 1000;
  std::vector<double> ts0 {ts};
  ts0.push_back(400);

  vector<var> theta_var = stan::math::to_var(theta);

  vector<vector<double> > ts_m (np, ts0);
  vector<vector<double> > y0_m (np, y0);
  vector<vector<var> > theta_var_m (np, theta_var);
  vector<vector<double> > x_r_m (np, x_r);
  vector<vector<int> > x_i_m (np, x_i);

  vector<vector<var> > y = stan::math::integrate_ode_bdf(f, y0, t0, ts, theta_var, x_r, x_i);
  vector<Eigen::Matrix<var, -1, -1> > y_m = pk_integrate_ode_bdf(f, y0_m, t0, ts_m, theta_var_m , x_r_m, x_i_m);

  for (int i = 0; i < np; ++i) {
    for (int j = 0; j < ts.size(); ++j) {
      for (int k = 0; k < y0.size(); ++k) {
        EXPECT_FLOAT_EQ(y_m[i](j, k).val(), y[j][k].val());
        std::vector<double> g, g1;
        stan::math::set_zero_all_adjoints();
        y_m[i](j, k).grad(theta_var, g);
        stan::math::set_zero_all_adjoints();
        y[j][k].grad(theta_var, g1);
        for (int l = 0 ; l < theta.size(); ++l) {
          EXPECT_NEAR(g[l], g1[l], 1e-7);
        }
      }
    }
  }
}

TEST_F(TorstenOdeTest_lorenz, fwd_sensitivity_theta_AD_bdf_mpi_performance) {
  using torsten::dsolve::PKCvodesFwdSystem;
  using torsten::dsolve::pk_integrate_ode_bdf;
  using stan::math::var;
  using std::vector;

  torsten::mpi::init();

  // size of population
  const int np = 100;
  std::vector<double> ts0 {ts};
  ts0.push_back(100);

  vector<var> theta_var = stan::math::to_var(theta);

  vector<vector<double> > ts_m (np, ts0);
  vector<vector<double> > y0_m (np, y0);
  vector<vector<var> > theta_var_m (np, theta_var);
  vector<vector<double> > x_r_m (np, x_r);
  vector<vector<int> > x_i_m (np, x_i);

  vector<vector<var> > y = stan::math::integrate_ode_bdf(f, y0, t0, ts, theta_var, x_r, x_i);
  vector<Eigen::Matrix<var, -1, -1> > y_m = pk_integrate_ode_bdf(f, y0_m, t0, ts_m, theta_var_m , x_r_m, x_i_m);

  for (int i = 0; i < np; ++i) {
    for (int j = 0; j < ts.size(); ++j) {
      for (int k = 0; k < y0.size(); ++k) {
        EXPECT_FLOAT_EQ(y_m[i](j, k).val(), y[j][k].val());
        std::vector<double> g, g1;
        stan::math::set_zero_all_adjoints();
        y_m[i](j, k).grad(theta_var, g);
        stan::math::set_zero_all_adjoints();
        y[j][k].grad(theta_var, g1);
        for (int l = 0 ; l < theta.size(); ++l) {
          EXPECT_NEAR(g[l], g1[l], 1e-6);
        }
      }
    }
  }
}

TEST_F(TorstenOdeTest_neutropenia, fwd_sensitivity_theta_AD_bdf_mpi) {
  using torsten::dsolve::PKCvodesFwdSystem;
  using torsten::dsolve::pk_integrate_ode_bdf;
  using stan::math::var;
  using std::vector;

  torsten::mpi::init();

  // size of population
  const int np = 100;

  vector<var> theta_var = stan::math::to_var(theta);

  vector<vector<double> > ts_m (np, ts);
  vector<vector<double> > y0_m (np, y0);
  vector<vector<var> > theta_var_m (np, theta_var);
  vector<vector<double> > x_r_m (np, x_r);
  vector<vector<int> > x_i_m (np, x_i);

  vector<vector<var> > y = stan::math::integrate_ode_bdf(f, y0, t0, ts, theta_var, x_r, x_i);
  vector<Eigen::Matrix<var, -1, -1> > y_m = pk_integrate_ode_bdf(f, y0_m, t0, ts_m, theta_var_m , x_r_m, x_i_m);

  for (int i = 0; i < np; ++i) {
    for (int j = 0; j < ts.size(); ++j) {
      for (int k = 0; k < y0.size(); ++k) {
        EXPECT_NEAR(y_m[i](j, k).val(), y[j][k].val(), 1.0e-7);
        std::vector<double> g, g1;
        stan::math::set_zero_all_adjoints();
        y_m[i](j, k).grad(theta_var, g);
        stan::math::set_zero_all_adjoints();
        y[j][k].grad(theta_var, g1);
        for (int l = 0 ; l < theta.size(); ++l) {
          EXPECT_NEAR(g[l], g1[l], 1e-6);
        }
      }
    }
  }
}

TEST_F(TorstenOdeTest_chem, fwd_sensitivity_theta_AD_adams_mpi_performance) {
  using torsten::dsolve::PKCvodesFwdSystem;
  using torsten::dsolve::pk_integrate_ode_adams;
  using stan::math::var;
  using std::vector;

  torsten::mpi::init();

  // size of population
  const int np = 100;
  std::vector<double> ts0 {ts};
  ts0.push_back(400);

  vector<var> theta_var = stan::math::to_var(theta);

  vector<vector<double> > ts_m (np, ts0);
  vector<vector<double> > y0_m (np, y0);
  vector<vector<var> > theta_var_m (np, theta_var);
  vector<vector<double> > x_r_m (np, x_r);
  vector<vector<int> > x_i_m (np, x_i);

  vector<vector<var> > y = stan::math::integrate_ode_adams(f, y0, t0, ts, theta_var, x_r, x_i);
  vector<Eigen::Matrix<var, -1, -1> > y_m = pk_integrate_ode_adams(f, y0_m, t0, ts_m, theta_var_m , x_r_m, x_i_m);

  for (int i = 0; i < np; ++i) {
    for (int j = 0; j < ts.size(); ++j) {
      for (int k = 0; k < y0.size(); ++k) {
        EXPECT_NEAR(y_m[i](j, k).val(), y[j][k].val(), 1e-5);
        std::vector<double> g, g1;
        stan::math::set_zero_all_adjoints();
        y_m[i](j, k).grad(theta_var, g);
        stan::math::set_zero_all_adjoints();
        y[j][k].grad(theta_var, g1);
        for (int l = 0 ; l < theta.size(); ++l) {
          EXPECT_NEAR(g[l], g1[l], 1e-5);
        }
      }
    }
  }
}

TEST_F(TorstenOdeTest_lorenz, fwd_sensitivity_theta_AD_adams_mpi_performance) {
  using torsten::dsolve::PKCvodesFwdSystem;
  using torsten::dsolve::pk_integrate_ode_adams;
  using stan::math::var;
  using std::vector;

  torsten::mpi::init();

  // size of population
  const int np = 100;
  std::vector<double> ts0 {ts};
  ts0.push_back(100);

  vector<var> theta_var = stan::math::to_var(theta);

  vector<vector<double> > ts_m (np, ts0);
  vector<vector<double> > y0_m (np, y0);
  vector<vector<var> > theta_var_m (np, theta_var);
  vector<vector<double> > x_r_m (np, x_r);
  vector<vector<int> > x_i_m (np, x_i);

  vector<vector<var> > y = stan::math::integrate_ode_adams(f, y0, t0, ts, theta_var, x_r, x_i);
  vector<Eigen::Matrix<var, -1, -1> > y_m = pk_integrate_ode_adams(f, y0_m, t0, ts_m, theta_var_m , x_r_m, x_i_m);

  for (int i = 0; i < np; ++i) {
    for (int j = 0; j < ts.size(); ++j) {
      for (int k = 0; k < y0.size(); ++k) {
        EXPECT_NEAR(y_m[i](j, k).val(), y[j][k].val(), 1e-6);
        std::vector<double> g, g1;
        stan::math::set_zero_all_adjoints();
        y_m[i](j, k).grad(theta_var, g);
        stan::math::set_zero_all_adjoints();
        y[j][k].grad(theta_var, g1);
        for (int l = 0 ; l < theta.size(); ++l) {
          EXPECT_NEAR(g[l], g1[l], 1.2e-5);
        }
      }
    }
  }
}

TEST_F(TorstenOdeTest_neutropenia, fwd_sensitivity_theta_AD_adams_mpi) {
  using torsten::dsolve::PKCvodesFwdSystem;
  using torsten::dsolve::pk_integrate_ode_adams;
  using stan::math::var;
  using std::vector;

  torsten::mpi::init();

  // size of population
  const int np = 100;

  vector<var> theta_var = stan::math::to_var(theta);

  vector<vector<double> > ts_m (np, ts);
  vector<vector<double> > y0_m (np, y0);
  vector<vector<var> > theta_var_m (np, theta_var);
  vector<vector<double> > x_r_m (np, x_r);
  vector<vector<int> > x_i_m (np, x_i);

  vector<vector<var> > y = stan::math::integrate_ode_adams(f, y0, t0, ts, theta_var, x_r, x_i);
  vector<Eigen::Matrix<var, -1, -1> > y_m = pk_integrate_ode_adams(f, y0_m, t0, ts_m, theta_var_m , x_r_m, x_i_m);

  for (int i = 0; i < np; ++i) {
    for (int j = 0; j < ts.size(); ++j) {
      for (int k = 0; k < y0.size(); ++k) {
        EXPECT_NEAR(y_m[i](j, k).val(), y[j][k].val(), 1.0e-6);
        std::vector<double> g, g1;
        stan::math::set_zero_all_adjoints();
        y_m[i](j, k).grad(theta_var, g);
        stan::math::set_zero_all_adjoints();
        y[j][k].grad(theta_var, g1);
        for (int l = 0 ; l < theta.size(); ++l) {
          EXPECT_NEAR(g[l], g1[l], 1e-5);
        }
      }
    }
  }
}

#endif

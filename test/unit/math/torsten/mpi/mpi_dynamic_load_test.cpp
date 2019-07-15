#ifdef TORSTEN_MPI

#include <stan/math.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/torsten/mpi/dynamic_load.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/torsten/pmx_ode_test_fixture.hpp>
#include <test/unit/math/prim/arr/functor/harmonic_oscillator.hpp>
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
#include <limits>
#include <vector>
#include <chrono>
#include <ctime>
#include <random>

template<typename... Args>
inline auto torsten::dsolve::pmx_ode_group_mpi_functor::operator()(Args&&... args) const {
    if (id == 0) { const TwoCptNeutModelODE f; return f(std::forward<Args>(args)...); }

    // return default
    TwoCptNeutModelODE f;
    return f(std::forward<Args>(args)...);
}

TEST_F(TorstenOdeTest_neutropenia, mpi_dynamic_load_set_use_uniform_work) {
  using torsten::dsolve::PMXCvodesFwdSystem;
  using torsten::pmx_integrate_ode_group_adams;
  using torsten::pmx_integrate_ode_adams;
  using stan::math::var;
  using std::vector;

  torsten::mpi::Envionment::init();

  // size of population
  const int np = 8;
  std::vector<double> ts0 {ts};

  vector<var> theta_var = stan::math::to_var(theta);

  vector<int> len(np, ts0.size());
  vector<double> ts_m;
  ts_m.reserve(np * ts0.size());
  for (int i = 0; i < np; ++i) ts_m.insert(ts_m.end(), ts0.begin(), ts0.end());
  
  vector<vector<double> > y0_m (np, y0);
  vector<vector<var> > theta_var_m (np, theta_var);
  vector<vector<double> > theta_m (np, theta);
  vector<vector<double> > x_r_m (np, x_r);
  vector<vector<int> > x_i_m (np, x_i);

  torsten::mpi::Communicator pmx_comm(torsten::mpi::Session<NUM_TORSTEN_COMM>::env, MPI_COMM_WORLD);
  torsten::mpi::PMXDynamicLoad<torsten::dsolve::pmx_ode_group_mpi_functor> load(pmx_comm);

  load.init_buf = std::vector<size_t>{0, 1, y0.size(), theta.size(), x_r.size(), x_i.size(), 0, 0, 0};
  load.set_work(1, y0_m, t0, len, ts_m, theta_m, x_r_m, x_i_m, 1.e-8, 1.e-10, 10000);
  std::vector<double> y0_slave;
  double t0_slave;
  std::vector<double> ts_slave;
  std::vector<double> theta_slave;
  std::vector<double> x_r_slave;
  std::vector<int> x_i_slave;
  double rtol_slave;
  double atol_slave;
  int max_num_step_slave;
  load.use_work(y0_slave, t0_slave, ts_slave, theta_slave, x_r_slave, x_i_slave, rtol_slave, atol_slave, max_num_step_slave);
  torsten::test::test_val(t0_slave, t0);
  torsten::test::test_val(y0_slave, y0);
  torsten::test::test_val(ts_slave, ts);
  torsten::test::test_val(theta_slave, theta);
  torsten::test::test_val(x_r_slave, x_r);
  torsten::test::test_val(rtol_slave, 1.e-8);
  torsten::test::test_val(atol_slave, 1.e-10);
  torsten::test::test_val(max_num_step_slave, 10000);
}

TEST_F(TorstenOdeTest_neutropenia, mpi_dynamic_load_set_use_non_uniform_work) {
  using torsten::dsolve::PMXCvodesFwdSystem;
  using torsten::pmx_integrate_ode_group_adams;
  using torsten::pmx_integrate_ode_adams;
  using stan::math::var;
  using std::vector;

  torsten::mpi::Envionment::init();

  // size of population
  const int np = 3;
  std::vector<std::vector<double>> tss(3, ts);
  tss[0].resize(ts.size() - 1);
  tss[2].resize(ts.size() - 3);

  vector<var> theta_var = stan::math::to_var(theta);

  vector<size_t> len{tss[0].size(), tss[1].size(), tss[2].size()};
  vector<double> ts_m;
  ts_m.reserve(tss[0].size() + tss[1].size() + tss[2].size());
  for (int i = 0; i < np; ++i) ts_m.insert(ts_m.end(), tss[i].begin(), tss[i].end());
  
  vector<vector<double> > y0_m (np, y0);
  vector<vector<var> > theta_var_m (np, theta_var);
  vector<vector<double> > theta_m (np, theta);
  x_r.push_back(1.0); vector<vector<double> > x_r_m (np, x_r);
  x_i.push_back(1.0); vector<vector<int> > x_i_m (np, x_i);

  torsten::mpi::Communicator pmx_comm(torsten::mpi::Session<NUM_TORSTEN_COMM>::env, MPI_COMM_WORLD);
  torsten::mpi::PMXDynamicLoad<torsten::dsolve::pmx_ode_group_mpi_functor> load(pmx_comm);

  load.init_buf = std::vector<size_t>{0, 1, y0.size(), theta.size(), x_r.size(), x_i.size(), 0, 0, 0};
  load.set_work(1, y0_m, t0, len, ts_m, theta_m, x_r_m, x_i_m, 1.e-8, 1.e-10, 10000);
  std::vector<double> y0_slave;
  double t0_slave;
  std::vector<double> ts_slave;
  std::vector<double> theta_slave;
  std::vector<double> x_r_slave;
  std::vector<int> x_i_slave;
  double rtol_slave;
  double atol_slave;
  int max_num_step_slave;
  load.use_work(y0_slave, t0_slave, ts_slave, theta_slave, x_r_slave, x_i_slave, rtol_slave, atol_slave, max_num_step_slave);
  torsten::test::test_val(y0_slave, y0);
  torsten::test::test_val(theta_slave, theta);
  torsten::test::test_val(x_r_slave, x_r);
  torsten::test::test_val(rtol_slave, 1.e-8);
  torsten::test::test_val(atol_slave, 1.e-10);
  torsten::test::test_val(max_num_step_slave, 10000);

  for (size_t i = 0; i < tss.size(); ++i) {
    load.set_work(i, y0_m, t0, len, ts_m, theta_m, x_r_m, x_i_m, 1.e-8, 1.e-10, 10000);      
    load.use_work(y0_slave, t0_slave, ts_slave, theta_slave, x_r_slave, x_i_slave, rtol_slave, atol_slave, max_num_step_slave);    
    torsten::test::test_val(ts_slave, tss[i]);    
  }

}

#endif

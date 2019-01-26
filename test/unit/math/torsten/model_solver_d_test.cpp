#include <stan/math/rev/mat.hpp>  // FIX ME - includes should be more specific
#include <test/unit/math/torsten/expect_near_matrix_eq.hpp>
#include <test/unit/math/torsten/expect_matrix_eq.hpp>
#include <test/unit/math/torsten/pk_twocpt_test_fixture.hpp>
#include <test/unit/math/torsten/util_generalOdeModel.hpp>
#include <test/unit/math/torsten/test_util.hpp>
#include <stan/math/torsten/PKModelTwoCpt.hpp>
#include <stan/math/torsten/pk_onecpt_model.hpp>
#include <stan/math/torsten/pk_twocpt_model.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/mat.hpp>  // FIX ME - include should be more specific
#include <test/unit/math/torsten/util_PKModelTwoCpt.hpp>
#include <vector>

using std::vector;
using Eigen::Matrix;
using Eigen::Dynamic;
using stan::math::var;
using refactor::PKRec;

TEST_F(TorstenPKTwoCptTest, model_solve_d_data_only) {
  using torsten::EventsManager;
  using model_t = refactor::PKTwoCptModel<double, double, double, double>;

  const int ncmt = model_t::Ncmt;
  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1], dt = 0.1;
  std::vector<double> rate{1000., 0., 0.};

  model_t model(t, init, rate, pMatrix[0]);
  
  Eigen::VectorXd sol1 = model.solve(dt);
  Eigen::VectorXd sol2 = model.solve_d(dt);
  
  torsten::test::test_val(sol1, sol2);
}

TEST_F(TorstenPKTwoCptTest, model_solve_d_init_var) {
  using torsten::EventsManager;
  using stan::math::vector_v;
  using stan::math::var;
  using model_t = refactor::PKTwoCptModel<double, var, double, double>;

  const int ncmt = model_t::Ncmt;
  refactor::PKRec<var> init(ncmt);
  init << 200, 100, 0;

  double t = time[1], dt = 0.1;
  std::vector<double> rate{400., 1000., 0.};

  model_t model(t, init, rate, pMatrix[0]);

  std::vector<var> vars(model.vars(dt));
  EXPECT_EQ(vars.size(), init.size());

  vector_v sol1 = model.solve(dt);
  Eigen::VectorXd sol2_d = model.solve_d(dt);
  vector_v sol2 = torsten::mpi::precomputed_gradients(sol2_d, vars);
  torsten::test::test_grad(vars, sol1, sol2, 1.E-8, 1.E-5);
}

TEST_F(TorstenPKTwoCptTest, model_solve_d_rate_var) {
  using torsten::EventsManager;
  using stan::math::vector_v;
  using stan::math::var;
  using model_t = refactor::PKTwoCptModel<double, double, var, double>;

  const int ncmt = model_t::Ncmt;
  refactor::PKRec<double> init(ncmt);
  init << 200, 100, 0;

  double t = time[1], dt = 0.1;
  std::vector<var> rate{400., 1000., 0.};

  model_t model(t, init, rate, pMatrix[0]);

  std::vector<var> vars(model.vars(dt));
  EXPECT_EQ(vars.size(), rate.size());

  vector_v sol1 = model.solve(dt);
  Eigen::VectorXd sol2_d = model.solve_d(dt);
  vector_v sol2 = torsten::mpi::precomputed_gradients(sol2_d, vars);
  torsten::test::test_grad(vars, sol1, sol2, 1.E-8, 1.E-5);
}

TEST_F(TorstenPKTwoCptTest, model_solve_d_par_var) {
  using torsten::EventsManager;
  using stan::math::vector_v;
  using stan::math::var;
  using model_t = refactor::PKTwoCptModel<double, double, double, var>;

  const int ncmt = model_t::Ncmt;
  refactor::PKRec<double> init(ncmt);
  init << 200, 100, 0;

  double t = time[1], dt = 0.1;
  std::vector<double> rate{400., 1000., 0.};

  std::vector<var> pars(stan::math::to_var(pMatrix[0]));
  model_t model(t, init, rate, pars);
  
  std::vector<var> vars(model.vars(dt));
  EXPECT_EQ(vars.size(), pars.size());

  vector_v sol1 = model.solve(dt);
  Eigen::VectorXd sol2_d = model.solve_d(dt);
  vector_v sol2 = torsten::mpi::precomputed_gradients(sol2_d, vars);
  torsten::test::test_grad(vars, sol1, sol2, 1.E-8, 1.E-5);
}

TEST_F(TorstenPKTwoCptTest, model_solve_d_rate_par_var) {
  using torsten::EventsManager;
  using stan::math::vector_v;
  using stan::math::var;
  using model_t = refactor::PKTwoCptModel<double, double, var, var>;

  const int ncmt = model_t::Ncmt;
  refactor::PKRec<double> init(ncmt);
  init << 200, 100, 0;

  double t = time[1], dt = 0.1;
  std::vector<var> rate{400., 1000., 0.};

  std::vector<var> pars(stan::math::to_var(pMatrix[0]));
  model_t model(t, init, rate, pars);

  std::vector<var> vars(model.vars(dt));
  EXPECT_EQ(vars.size(), pars.size() + rate.size());

  vector_v sol1 = model.solve(dt);
  Eigen::VectorXd sol2_d = model.solve_d(dt);
  vector_v sol2 = torsten::mpi::precomputed_gradients(sol2_d, vars);
  torsten::test::test_grad(vars, sol1, sol2, 1.E-8, 1.E-5);
}


TEST_F(TorstenPKTwoCptTest, model_solve_d_data_only_ss) {
  using torsten::EventsManager;
  using model_t = refactor::PKTwoCptModel<double, double, double, double>;

  const int ncmt = model_t::Ncmt;
  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1];
  std::vector<double> rate{1000., 0., 0.};

  model_t model(t, init, rate, pMatrix[0]);
  
  double a = 1500, r = 100, ii = 18.0;
  int cmt = 1;
  Eigen::VectorXd sol1 = model.solve(a, r, ii, cmt);
  Eigen::VectorXd sol2 = model.solve_d(a, r, ii, cmt);
  torsten::test::test_val(sol1, sol2);
}

TEST_F(TorstenPKTwoCptTest, model_solve_d_init_var_ss) {
  using torsten::EventsManager;
  using stan::math::var;
  using stan::math::vector_v;
  using model_t = refactor::PKTwoCptModel<double, var, double, double>;

  const int ncmt = model_t::Ncmt;
  refactor::PKRec<var> init(ncmt);
  init << 0, 100, 0;

  double t = time[1];
  std::vector<double> rate{1000., 0., 0.};

  model_t model(t, init, rate, pMatrix[0]);
  
  double a = 1500, r = 100, ii = 18.0;
  int cmt = 1;

  // initial condition should not affect steady-state solution
  std::vector<var> vars(model.vars(a, r, ii));
  EXPECT_EQ(vars.size(), 0);

  Eigen::VectorXd sol1 = model.solve(a, r, ii, cmt);
  Eigen::VectorXd sol2 = model.solve_d(a, r, ii, cmt);
  EXPECT_EQ(sol2.size(), sol1.size());
  torsten::test::test_val(sol1, sol2);
}

TEST_F(TorstenPKTwoCptTest, model_solve_d_rate_var_ss) {
  using torsten::EventsManager;
  using stan::math::var;
  using stan::math::vector_v;
  using model_t = refactor::PKTwoCptModel<double, double, var, double>;

  const int ncmt = model_t::Ncmt;
  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1];
  std::vector<var> rate{1000., 0., 0.};

  model_t model(t, init, rate, pMatrix[0]);
  
  double a = 1500, r = 100, ii = 18.0;
  int cmt = 1;

  // rate in model constructor should not affect steady-state solution
  std::vector<var> vars(model.vars(a, r, ii));
  EXPECT_EQ(vars.size(), 0);

  Eigen::VectorXd sol1 = model.solve(a, r, ii, cmt);
  Eigen::VectorXd sol2 = model.solve_d(a, r, ii, cmt);
  EXPECT_EQ(sol2.size(), sol1.size());
  torsten::test::test_val(sol1, sol2);
}

TEST_F(TorstenPKTwoCptTest, model_solve_d_par_var_ss) {
  using torsten::EventsManager;
  using stan::math::var;
  using stan::math::vector_v;
  using model_t = refactor::PKTwoCptModel<double, double, double, var>;

  const int ncmt = model_t::Ncmt;
  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1];
  std::vector<double> rate{1000., 0., 0.};

  std::vector<var> pars(stan::math::to_var(pMatrix[0]));

  model_t model(t, init, rate, pars);
  
  double a = 1500, r = 100, ii = 18.0;
  int cmt = 1;

  // rate in model constructor should not affect steady-state solution
  std::vector<var> vars(model.vars(a, r, ii));
  EXPECT_EQ(vars.size(), pars.size());

  vector_v sol1 = model.solve(a, r, ii, cmt);
  Eigen::VectorXd sol2_d = model.solve_d(a, r, ii, cmt);
  vector_v sol2 = torsten::mpi::precomputed_gradients(sol2_d, vars);
  
  torsten::test::test_grad(vars, pars, sol1, sol2, 1.E-8, 1.E-5);
}

TEST_F(TorstenPKTwoCptTest, model_solve_d_amt_par_var_ss) {
  using torsten::EventsManager;
  using stan::math::var;
  using stan::math::vector_v;
  using model_t = refactor::PKTwoCptModel<double, double, double, var>;

  const int ncmt = model_t::Ncmt;
  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1];
  std::vector<double> rate{1000., 0., 0.};

  std::vector<var> pars(stan::math::to_var(pMatrix[0]));

  model_t model(t, init, rate, pars);
  
  var a = 1500;
  double r = 100, ii = 18.0;
  int cmt = 1;

  // rate in model constructor should not affect steady-state solution
  std::vector<var> vars(model.vars(a, r, ii));
  EXPECT_EQ(vars.size(), pars.size() + 1);

  vector_v sol1 = model.solve(a, r, ii, cmt);
  Eigen::VectorXd sol2_d = model.solve_d(a, r, ii, cmt);
  vector_v sol2 = torsten::mpi::precomputed_gradients(sol2_d, vars);
  
  torsten::test::test_grad(vars, sol1, sol2, 1.E-8, 1.E-5);
}

TEST_F(TorstenPKTwoCptTest, model_solve_d_amt_ii_var_ss) {
  using torsten::EventsManager;
  using stan::math::var;
  using stan::math::vector_v;
  using model_t = refactor::PKTwoCptModel<double, double, double, double>;

  const int ncmt = model_t::Ncmt;
  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1];
  std::vector<double> rate{1000., 0., 0.};

  model_t model(t, init, rate, pMatrix[0]);
  
  var a = 1500, ii = 18.0;
  double r = 100;
  int cmt = 1;

  // rate in model constructor should not affect steady-state solution
  std::vector<var> vars(model.vars(a, r, ii));
  EXPECT_EQ(vars.size(), 2);

  vector_v sol1 = model.solve(a, r, ii, cmt);
  Eigen::VectorXd sol2_d = model.solve_d(a, r, ii, cmt);
  vector_v sol2 = torsten::mpi::precomputed_gradients(sol2_d, vars);
  
  torsten::test::test_grad(vars, sol1, sol2, 1.E-8, 1.E-5);
}

TEST_F(TorstenPKTwoCptTest, rk45_model_solve_d_data_only) {
  using torsten::EventsManager;

  using refactor::PKTwoCptModel;
  using refactor::PKTwoCptODE;
  using refactor::PkOdeIntegrator;

  const int ncmt = PKTwoCptModel<double, double, double, double>::Ncmt;
  PKTwoCptODE f;

  using model_t = refactor::PKODEModel<double, double, double, double, PKTwoCptODE>;
  PkOdeIntegrator<torsten::StanRk45> integrator;

  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1], dt = 0.1;
  std::vector<double> rate{1000., 0., 0.};

  model_t model(t, init, rate, pMatrix[0], f);
  
  Eigen::VectorXd sol1 = model.solve(dt, integrator);
  Eigen::VectorXd sol2 = model.solve_d(dt, integrator);

  torsten::test::test_val(sol1, sol2);
}

TEST_F(TorstenPKTwoCptTest, rk45_model_solve_d_init_var) {
  using torsten::EventsManager;
  using refactor::PKTwoCptModel;
  using refactor::PKTwoCptODE;
  using refactor::PkOdeIntegrator;
  using stan::math::var;
  using stan::math::vector_v;

  const int ncmt = PKTwoCptModel<double, double, double, double>::Ncmt;
  PKTwoCptODE f;

  using model_t = refactor::PKODEModel<double, var, double, double, PKTwoCptODE>;
  PkOdeIntegrator<torsten::StanRk45> integrator;

  refactor::PKRec<var> init(ncmt);
  init << 0, 100, 0;

  double t = time[1], dt = 0.1;
  std::vector<double> rate{1000., 0., 0.};

  model_t model(t, init, rate, pMatrix[0], f);
  std::vector<var> vars(model.vars(dt));
  EXPECT_EQ(vars.size(), init.size());
  
  vector_v sol1 = model.solve(dt, integrator);
  Eigen::VectorXd sol2_d = model.solve_d(dt, integrator);
  vector_v sol2 = torsten::mpi::precomputed_gradients(sol2_d, vars);
  
  // vars and init should be pointing to the same @c vari
  torsten::test::test_grad(vars, init, sol1, sol2, 1.E-8, 1.E-5);
}

TEST_F(TorstenPKTwoCptTest, rk45_model_solve_d_rate_var) {
  using torsten::EventsManager;
  using refactor::PKTwoCptModel;
  using refactor::PKTwoCptODE;
  using refactor::PkOdeIntegrator;
  using stan::math::var;
  using stan::math::vector_v;

  const int ncmt = PKTwoCptModel<double, double, double, double>::Ncmt;
  PKTwoCptODE f;

  using model_t = refactor::PKODEModel<double, double, var, double, PKTwoCptODE>;
  PkOdeIntegrator<torsten::StanRk45> integrator;

  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1], dt = 0.1;
  std::vector<var> rate{1000., 0., 0.};

  std::vector<double>& par = pMatrix[0];
  model_t model(t, init, rate, par, f);
  std::vector<var> vars(model.vars(dt));
  std::vector<var> par_rate;
  par_rate.insert(par_rate.end(), par.begin(), par.end());
  par_rate.insert(par_rate.end(), rate.begin(), rate.end());
  EXPECT_EQ(vars.size(), par_rate.size());
  
  vector_v sol1 = model.solve(dt, integrator);
  Eigen::VectorXd sol2_d = model.solve_d(dt, integrator);
  vector_v sol2 = torsten::mpi::precomputed_gradients(sol2_d, vars);
  
  // vars and init should be pointing to the same @c vari
  torsten::test::test_grad(vars, par_rate, sol1, sol2, 1.E-8, 1.E-5);
}

TEST_F(TorstenPKTwoCptTest, rk45_model_solve_d_par_var) {
  using torsten::EventsManager;
  using refactor::PKTwoCptModel;
  using refactor::PKTwoCptODE;
  using refactor::PkOdeIntegrator;
  using stan::math::var;
  using stan::math::vector_v;

  const int ncmt = PKTwoCptModel<double, double, double, double>::Ncmt;
  PKTwoCptODE f;

  using model_t = refactor::PKODEModel<double, double, double, var, PKTwoCptODE>;
  PkOdeIntegrator<torsten::StanRk45> integrator;

  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1], dt = 0.1;
  std::vector<double> rate{1000., 0., 0.};

  std::vector<var> par(stan::math::to_var(pMatrix[0]));
  model_t model(t, init, rate, par, f);
  std::vector<var> vars(model.vars(dt));
  EXPECT_EQ(vars.size(), par.size());
  
  vector_v sol1 = model.solve(dt, integrator);
  Eigen::VectorXd sol2_d = model.solve_d(dt, integrator);
  vector_v sol2 = torsten::mpi::precomputed_gradients(sol2_d, vars);
  
  // vars and init should be pointing to the same @c vari
  torsten::test::test_grad(vars, par, sol1, sol2, 1.E-8, 1.E-5);
}

TEST_F(TorstenPKTwoCptTest, rk45_model_solve_d_data_only_ss) {
  using torsten::EventsManager;
  using refactor::PKTwoCptModel;
  using refactor::PKTwoCptODE;
  using refactor::PkOdeIntegrator;

  const int ncmt = PKTwoCptModel<double, double, double, double>::Ncmt;
  PKTwoCptODE f;

  using model_t = refactor::PKODEModel<double, double, double, double, PKTwoCptODE>;
  PkOdeIntegrator<torsten::StanRk45> integrator;

  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1];
  std::vector<double> rate{1000., 0., 0.};

  double a = 1500, r = 100, ii = 18.0;
  int cmt = 1;

  model_t model(t, init, rate, pMatrix[0], f);
  
  Eigen::VectorXd sol1 = model.solve(a, r, ii, cmt, integrator);
  Eigen::VectorXd sol2 = model.solve_d(a, r, ii, cmt, integrator);
  
  torsten::test::test_val(sol1, sol2);
}

TEST_F(TorstenPKTwoCptTest, rk45_model_solve_d_init_var_ss) {
  using torsten::EventsManager;

  using refactor::PKTwoCptModel;
  using refactor::PKTwoCptODE;
  using refactor::PkOdeIntegrator;
  using stan::math::var;
  using stan::math::vector_v;

  const int ncmt = PKTwoCptModel<double, double, double, double>::Ncmt;
  PKTwoCptODE f;

  using model_t = refactor::PKODEModel<double, var, double, double, PKTwoCptODE>;
  PkOdeIntegrator<torsten::StanRk45> integrator;

  refactor::PKRec<var> init(ncmt);
  init << 0, 100, 0;

  double t = time[1];
  std::vector<double> rate{1000., 0., 0.};

  double a = 1500, r = 100, ii = 18.0;
  int cmt = 1;

  model_t model(t, init, rate, pMatrix[0], f);
  std::vector<var> vars(model.vars(a, r, ii));

  // type of @c init should not affect steady state solution type.
  EXPECT_EQ(vars.size(), 0);
  
  Eigen::VectorXd sol1 = model.solve(a, r, ii, cmt, integrator);
  Eigen::VectorXd sol2 = model.solve_d(a, r, ii, cmt, integrator);
  
  torsten::test::test_val(sol1, sol2);
}

TEST_F(TorstenPKTwoCptTest, rk45_model_solve_d_par_var_ss) {
  using torsten::EventsManager;

  using refactor::PKTwoCptModel;
  using refactor::PKTwoCptODE;
  using refactor::PkOdeIntegrator;
  using stan::math::var;
  using stan::math::vector_v;

  const int ncmt = PKTwoCptModel<double, double, double, double>::Ncmt;
  PKTwoCptODE f;

  using model_t = refactor::PKODEModel<double, double, double, var, PKTwoCptODE>;
  PkOdeIntegrator<torsten::StanRk45> integrator;

  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1];
  std::vector<double> rate{1000., 3000., 0.};

  double a = 1500, r = 100, ii = 18.0;
  int cmt = 1;

  std::vector<var> par(stan::math::to_var(pMatrix[0]));
  model_t model(t, init, rate, par, f);
  std::vector<var> vars(model.vars(a, r, ii));

  EXPECT_EQ(vars.size(), par.size());
  
  vector_v sol1 = model.solve(a, r, ii, cmt, integrator);
  Eigen::VectorXd sol2_d = model.solve_d(a, r, ii, cmt, integrator);
  vector_v sol2 = torsten::mpi::precomputed_gradients(sol2_d, vars);
  
  // vars and init should be pointing to the same @c vari
  torsten::test::test_grad(vars, par, sol1, sol2, 1.E-8, 1.E-5);
}

// PkBdf
TEST_F(TorstenPKTwoCptTest, PkBdf_model_solve_d_data_only) {
  using torsten::EventsManager;

  using refactor::PKTwoCptModel;
  using refactor::PKTwoCptODE;
  using refactor::PkOdeIntegrator;

  const int ncmt = PKTwoCptModel<double, double, double, double>::Ncmt;
  PKTwoCptODE f;

  using model_t = refactor::PKODEModel<double, double, double, double, PKTwoCptODE>;
  PkOdeIntegrator<torsten::PkBdf> integrator;

  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1], dt = 0.1;
  std::vector<double> rate{1000., 0., 0.};

  model_t model(t, init, rate, pMatrix[0], f);
  
  Eigen::VectorXd sol1 = model.solve(dt, integrator);
  Eigen::VectorXd sol2 = model.solve_d(dt, integrator);

  torsten::test::test_val(sol1, sol2);
}

TEST_F(TorstenPKTwoCptTest, PkBdf_model_solve_d_init_var) {
  using torsten::EventsManager;
  using refactor::PKTwoCptModel;
  using refactor::PKTwoCptODE;
  using refactor::PkOdeIntegrator;
  using stan::math::var;
  using stan::math::vector_v;

  const int ncmt = PKTwoCptModel<double, double, double, double>::Ncmt;
  PKTwoCptODE f;

  using model_t = refactor::PKODEModel<double, var, double, double, PKTwoCptODE>;
  PkOdeIntegrator<torsten::PkBdf> integrator;

  refactor::PKRec<var> init(ncmt);
  init << 0, 100, 0;

  double t = time[1], dt = 0.1;
  std::vector<double> rate{1000., 0., 0.};

  model_t model(t, init, rate, pMatrix[0], f);
  std::vector<var> vars(model.vars(dt));
  EXPECT_EQ(vars.size(), init.size());
  
  vector_v sol1 = model.solve(dt, integrator);
  Eigen::VectorXd sol2_d = model.solve_d(dt, integrator);
  vector_v sol2 = torsten::mpi::precomputed_gradients(sol2_d, vars);
  
  // vars and init should be pointing to the same @c vari
  torsten::test::test_grad(vars, init, sol1, sol2, 1.E-8, 1.E-5);
}

TEST_F(TorstenPKTwoCptTest, PkBdf_model_solve_d_rate_var) {
  using torsten::EventsManager;
  using refactor::PKTwoCptModel;
  using refactor::PKTwoCptODE;
  using refactor::PkOdeIntegrator;
  using stan::math::var;
  using stan::math::vector_v;

  const int ncmt = PKTwoCptModel<double, double, double, double>::Ncmt;
  PKTwoCptODE f;

  using model_t = refactor::PKODEModel<double, double, var, double, PKTwoCptODE>;
  PkOdeIntegrator<torsten::PkBdf> integrator;

  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1], dt = 0.1;
  std::vector<var> rate{1000., 0., 0.};

  std::vector<double>& par = pMatrix[0];
  model_t model(t, init, rate, par, f);
  std::vector<var> vars(model.vars(dt));
  std::vector<var> par_rate;
  par_rate.insert(par_rate.end(), par.begin(), par.end());
  par_rate.insert(par_rate.end(), rate.begin(), rate.end());
  EXPECT_EQ(vars.size(), par_rate.size());
  
  vector_v sol1 = model.solve(dt, integrator);
  Eigen::VectorXd sol2_d = model.solve_d(dt, integrator);
  vector_v sol2 = torsten::mpi::precomputed_gradients(sol2_d, vars);
  
  // vars and init should be pointing to the same @c vari
  torsten::test::test_grad(vars, par_rate, sol1, sol2, 1.E-8, 1.E-5);
}

TEST_F(TorstenPKTwoCptTest, PkBdf_model_solve_d_par_var) {
  using torsten::EventsManager;
  using refactor::PKTwoCptModel;
  using refactor::PKTwoCptODE;
  using refactor::PkOdeIntegrator;
  using stan::math::var;
  using stan::math::vector_v;

  const int ncmt = PKTwoCptModel<double, double, double, double>::Ncmt;
  PKTwoCptODE f;

  using model_t = refactor::PKODEModel<double, double, double, var, PKTwoCptODE>;
  PkOdeIntegrator<torsten::PkBdf> integrator;

  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1], dt = 0.1;
  std::vector<double> rate{1000., 0., 0.};

  std::vector<var> par(stan::math::to_var(pMatrix[0]));
  model_t model(t, init, rate, par, f);
  std::vector<var> vars(model.vars(dt));
  EXPECT_EQ(vars.size(), par.size());
  
  vector_v sol1 = model.solve(dt, integrator);
  Eigen::VectorXd sol2_d = model.solve_d(dt, integrator);
  vector_v sol2 = torsten::mpi::precomputed_gradients(sol2_d, vars);
  
  // vars and init should be pointing to the same @c vari
  torsten::test::test_grad(vars, par, sol1, sol2, 1.E-8, 1.E-5);
}

TEST_F(TorstenPKTwoCptTest, PkBdf_model_solve_d_data_only_ss) {
  using torsten::EventsManager;
  using refactor::PKTwoCptModel;
  using refactor::PKTwoCptODE;
  using refactor::PkOdeIntegrator;

  const int ncmt = PKTwoCptModel<double, double, double, double>::Ncmt;
  PKTwoCptODE f;

  using model_t = refactor::PKODEModel<double, double, double, double, PKTwoCptODE>;
  PkOdeIntegrator<torsten::PkBdf> integrator;

  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1];
  std::vector<double> rate{1000., 0., 0.};

  double a = 1500, r = 100, ii = 18.0;
  int cmt = 1;

  model_t model(t, init, rate, pMatrix[0], f);
  
  Eigen::VectorXd sol1 = model.solve(a, r, ii, cmt, integrator);
  Eigen::VectorXd sol2 = model.solve_d(a, r, ii, cmt, integrator);
  
  torsten::test::test_val(sol1, sol2);
}

TEST_F(TorstenPKTwoCptTest, PkBdf_model_solve_d_init_var_ss) {
  using torsten::EventsManager;

  using refactor::PKTwoCptModel;
  using refactor::PKTwoCptODE;
  using refactor::PkOdeIntegrator;
  using stan::math::var;
  using stan::math::vector_v;

  const int ncmt = PKTwoCptModel<double, double, double, double>::Ncmt;
  PKTwoCptODE f;

  using model_t = refactor::PKODEModel<double, var, double, double, PKTwoCptODE>;
  PkOdeIntegrator<torsten::PkBdf> integrator;

  refactor::PKRec<var> init(ncmt);
  init << 0, 100, 0;

  double t = time[1];
  std::vector<double> rate{1000., 0., 0.};

  double a = 1500, r = 100, ii = 18.0;
  int cmt = 1;

  model_t model(t, init, rate, pMatrix[0], f);
  std::vector<var> vars(model.vars(a, r, ii));

  // type of @c init should not affect steady state solution type.
  EXPECT_EQ(vars.size(), 0);
  
  Eigen::VectorXd sol1 = model.solve(a, r, ii, cmt, integrator);
  Eigen::VectorXd sol2 = model.solve_d(a, r, ii, cmt, integrator);
  
  torsten::test::test_val(sol1, sol2);
}

TEST_F(TorstenPKTwoCptTest, PkBdf_model_solve_d_par_var_ss) {
  using torsten::EventsManager;

  using refactor::PKTwoCptModel;
  using refactor::PKTwoCptODE;
  using refactor::PkOdeIntegrator;
  using stan::math::var;
  using stan::math::vector_v;

  const int ncmt = PKTwoCptModel<double, double, double, double>::Ncmt;
  PKTwoCptODE f;

  using model_t = refactor::PKODEModel<double, double, double, var, PKTwoCptODE>;
  PkOdeIntegrator<torsten::PkBdf> integrator;

  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1];
  std::vector<double> rate{1000., 3000., 0.};

  double a = 1500, r = 100, ii = 18.0;
  int cmt = 1;

  std::vector<var> par(stan::math::to_var(pMatrix[0]));
  model_t model(t, init, rate, par, f);
  std::vector<var> vars(model.vars(a, r, ii));

  EXPECT_EQ(vars.size(), par.size());
  
  vector_v sol1 = model.solve(a, r, ii, cmt, integrator);
  Eigen::VectorXd sol2_d = model.solve_d(a, r, ii, cmt, integrator);
  vector_v sol2 = torsten::mpi::precomputed_gradients(sol2_d, vars);
  
  // vars and init should be pointing to the same @c vari
  torsten::test::test_grad(vars, par, sol1, sol2, 1.E-8, 1.E-5);
}

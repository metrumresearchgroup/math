#include <stan/math/rev/mat.hpp>  // FIX ME - includes should be more specific
#include <test/unit/math/torsten/expect_near_matrix_eq.hpp>
#include <test/unit/math/torsten/expect_matrix_eq.hpp>
#include <test/unit/math/torsten/pk_twocpt_test_fixture.hpp>
#include <test/unit/math/torsten/util_generalOdeModel.hpp>
#include <test/unit/math/torsten/test_util.hpp>
#include <stan/math/torsten/model_solver_dispatcher.hpp>
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


TEST_F(TorstenPKTwoCptTest, solver_dispatcher_data_only) {
  using torsten::EventsManager;
  using torsten::ModelSolverDispatcher;
  using model_t = refactor::PKTwoCptModel<double, double, double, double>;

  const int ncmt = model_t::Ncmt;
  refactor::PKRec<double> init(ncmt);
  init << 0, 100, 0;

  double t = time[1], dt = 0.1;
  std::vector<double> rate{1000., 0., 0.};

  model_t model(t, init, rate, pMatrix[0]);
  
  Eigen::VectorXd sol1 = model.solve(dt);
  Eigen::VectorXd sol2 = ModelSolverDispatcher<true, model_t>::solve(model, dt);
  Eigen::VectorXd sol3 = ModelSolverDispatcher<false, model_t>::solve(model, dt);
  
  torsten::test::test_val(sol1, sol2);
  torsten::test::test_val(sol1, sol3);
}

TEST_F(TorstenPKTwoCptTest, solver_dispatcher_data_only_ss) {
  using torsten::EventsManager;
  using torsten::ModelSolverDispatcher;
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
  Eigen::VectorXd sol2 = ModelSolverDispatcher<true, model_t>::solve(model, a, r, ii, cmt);
  Eigen::VectorXd sol3 = ModelSolverDispatcher<false, model_t>::solve(model, a, r, ii, cmt);
  
  torsten::test::test_val(sol1, sol2);
  torsten::test::test_val(sol1, sol3);
}

TEST_F(TorstenPKTwoCptTest, solver_dispatcher_init_var) {
  using torsten::EventsManager;
  using torsten::ModelSolverDispatcher;
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
  vector_v sol2 = ModelSolverDispatcher<true, model_t>::solve(model, dt);
  torsten::test::test_grad(vars, sol1, sol2, 1.E-8, 1.E-5);


  Eigen::VectorXd sol3_d = ModelSolverDispatcher<false, model_t>::solve(model, dt);
  vector_v sol3 = torsten::mpi::precomputed_gradients(sol3_d, vars);

  torsten::test::test_grad(vars, sol1, sol3, 1.E-8, 1.E-5);
}

TEST_F(TorstenPKTwoCptTest, solver_dispatcher_rate_var) {
  using torsten::EventsManager;
  using torsten::ModelSolverDispatcher;
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
  vector_v sol2 = ModelSolverDispatcher<true, model_t>::solve(model, dt);
  torsten::test::test_grad(vars, sol1, sol2, 1.E-8, 1.E-5);


  Eigen::VectorXd sol3_d = ModelSolverDispatcher<false, model_t>::solve(model, dt);
  vector_v sol3 = torsten::mpi::precomputed_gradients(sol3_d, vars);

  torsten::test::test_grad(vars, sol1, sol3, 1.E-8, 1.E-5);
}

TEST_F(TorstenPKTwoCptTest, solver_dispatcher_par_var) {
  using torsten::EventsManager;
  using torsten::ModelSolverDispatcher;
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
  vector_v sol2 = ModelSolverDispatcher<true, model_t>::solve(model, dt);
  torsten::test::test_grad(vars, sol1, sol2, 1.E-8, 1.E-5);


  Eigen::VectorXd sol3_d = ModelSolverDispatcher<false, model_t>::solve(model, dt);
  vector_v sol3 = torsten::mpi::precomputed_gradients(sol3_d, vars);
  torsten::test::test_grad(vars, sol1, sol3, 1.E-8, 1.E-5);
}

TEST_F(TorstenPKTwoCptTest, solver_dispatcher_rate_par_var) {
  using torsten::EventsManager;
  using torsten::ModelSolverDispatcher;
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
  vector_v sol2 = ModelSolverDispatcher<true, model_t>::solve(model, dt);
  torsten::test::test_grad(vars, sol1, sol2, 1.E-8, 1.E-5);


  Eigen::VectorXd sol3_d = ModelSolverDispatcher<false, model_t>::solve(model, dt);
  vector_v sol3 = torsten::mpi::precomputed_gradients(sol3_d, vars);
  torsten::test::test_grad(vars, sol1, sol3, 1.E-8, 1.E-5);
}

TEST_F(TorstenPKTwoCptTest, solver_dispatcher_init_var_ss) {
  using torsten::EventsManager;
  using torsten::ModelSolverDispatcher;
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

  vector_v sol1 = model.solve(a, r, ii, cmt);
  vector_v sol2 = ModelSolverDispatcher<true, model_t>::solve(model, a, r, ii, cmt);
  Eigen::VectorXd sol3 = ModelSolverDispatcher<false, model_t>::solve(model, a, r, ii, cmt);
  EXPECT_EQ(sol3.size(), sol1.size());
  
  torsten::test::test_grad(vars, sol1, sol2, 1.E-8, 1.E-5);
  torsten::test::test_val(sol1, sol3);
}

TEST_F(TorstenPKTwoCptTest, solver_dispatcher_rate_var_ss) {
  using torsten::EventsManager;
  using torsten::ModelSolverDispatcher;
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

  vector_v sol1 = model.solve(a, r, ii, cmt);
  vector_v sol2 = ModelSolverDispatcher<true, model_t>::solve(model, a, r, ii, cmt);
  Eigen::VectorXd sol3 = ModelSolverDispatcher<false, model_t>::solve(model, a, r, ii, cmt);
  EXPECT_EQ(sol3.size(), sol1.size());
  
  torsten::test::test_grad(vars, sol1, sol2, 1.E-8, 1.E-5);
  torsten::test::test_val(sol1, sol3);
}

TEST_F(TorstenPKTwoCptTest, solver_dispatcher_par_var_ss) {
  using torsten::EventsManager;
  using torsten::ModelSolverDispatcher;
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
  vector_v sol2 = ModelSolverDispatcher<true, model_t>::solve(model, a, r, ii, cmt);
  Eigen::VectorXd sol3_d = ModelSolverDispatcher<false, model_t>::solve(model, a, r, ii, cmt);
  vector_v sol3 = torsten::mpi::precomputed_gradients(sol3_d, vars);
  
  torsten::test::test_grad(vars, sol1, sol2, 1.E-8, 1.E-5);
  torsten::test::test_grad(vars, sol1, sol3, 1.E-8, 1.E-5);
}

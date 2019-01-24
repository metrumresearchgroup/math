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


TEST_F(TorstenPKTwoCptTest, solver_dispatcher_double) {
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

TEST_F(TorstenPKTwoCptTest, solver_dispatcher_double_ss) {
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

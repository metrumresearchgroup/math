#include <stan/math/rev/mat.hpp>  // FIX ME - includes should be more specific
#include <test/unit/math/torsten/expect_near_matrix_eq.hpp>
#include <test/unit/math/torsten/expect_matrix_eq.hpp>
#include <test/unit/math/torsten/pk_twocpt_mpi_test_fixture.hpp>
#include <test/unit/math/torsten/util_generalOdeModel.hpp>
#include <test/unit/math/torsten/test_util.hpp>
#include <stan/math/torsten/mpi/init.hpp>
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

TEST_F(TorstenPKTwoCptMPITest, bdf_solver_multiple_bolus_doses_data_only) {
    using model_t = refactor::PKTwoCptModel<double, double, double, double>;

    Matrix<double, Dynamic, Dynamic> x =
      torsten::generalOdeModel_bdf(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag); // NOLINT

    vector<Matrix<double, Dynamic, Dynamic> > x_m =
      torsten::pop_pk_generalOdeModel_bdf(model_t::f_, model_t::Ncmt, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, pMatrix_m, biovar_m, tlag_m); // NOLINT

    for (int i = 0; i < np; ++i) {
      torsten::test::test_val(x_m[i], x);
    }
}

TEST_F(TorstenPKTwoCptMPITest, bdf_solver_multiple_IV_doses_data_only) {
  rate[0] = 300;
  for (int i = 0; i < np; ++i) {
    rate_m[i] = rate;
  }

  using model_t = refactor::PKTwoCptModel<double, double, double, double>;

  Matrix<double, Dynamic, Dynamic> x =
    torsten::generalOdeModel_bdf(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag); // NOLINT

  vector<Matrix<double, Dynamic, Dynamic> > x_m =
    torsten::pop_pk_generalOdeModel_bdf(model_t::f_, model_t::Ncmt, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, pMatrix_m, biovar_m, tlag_m); // NOLINT

  for (int i = 0; i < np; ++i) {
    torsten::test::test_val(x_m[i], x);
  }
}

TEST_F(TorstenPKTwoCptMPITest, bdf_solver_multiple_bolus_doses_par_var) {
    vector<vector<vector<var> > > pMatrix_m_v(pMatrix_m.size());
    vector<vector<var> > pMatrix_v(torsten::test::to_var(pMatrix));
    for (int i = 0; i < np; ++i) {
      pMatrix_m_v[i] = torsten::test::to_var(pMatrix_m[i]);
    }

    using model_t = refactor::PKTwoCptModel<double, double, double, double>;

    Matrix<var, Dynamic, Dynamic> x =
      torsten::generalOdeModel_bdf(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix_v, biovar, tlag); // NOLINT

    vector<Matrix<var, Dynamic, Dynamic> > x_m =
      torsten::pop_pk_generalOdeModel_bdf(model_t::f_, model_t::Ncmt, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, pMatrix_m_v, biovar_m, tlag_m); // NOLINT

    for (int i = 0; i < np; ++i) {
      torsten::test::test_grad(pMatrix_m_v[i][0], pMatrix_v[0], x_m[i], x, 1.E-8, 1.E-5);
    }
}

TEST_F(TorstenPKTwoCptMPITest, bdf_solver_multiple_IV_doses_par_var) {
  rate[0] = 300;
  for (int i = 0; i < np; ++i) {
    rate_m[i] = rate;
  }

  vector<vector<vector<var> > > pMatrix_m_v(pMatrix_m.size());
  vector<vector<var> > pMatrix_v(torsten::test::to_var(pMatrix));
  for (int i = 0; i < np; ++i) {
    pMatrix_m_v[i] = torsten::test::to_var(pMatrix_m[i]);
  }

  using model_t = refactor::PKTwoCptModel<double, double, double, double>;

  Matrix<var, Dynamic, Dynamic> x =
    torsten::generalOdeModel_bdf(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix_v, biovar, tlag); // NOLINT

  vector<Matrix<var, Dynamic, Dynamic> > x_m =
    torsten::pop_pk_generalOdeModel_bdf(model_t::f_, model_t::Ncmt, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, pMatrix_m_v, biovar_m, tlag_m); // NOLINT

  for (int i = 0; i < np; ++i) {
    torsten::test::test_grad(pMatrix_m_v[i][0], pMatrix_v[0], x_m[i], x, 1.E-8, 1.E-5);
  }
}

TEST_F(TorstenPKTwoCptMPITest, rk45_solver_multiple_bolus_doses_data_only) {
    using model_t = refactor::PKTwoCptModel<double, double, double, double>;

    Matrix<double, Dynamic, Dynamic> x =
      torsten::generalOdeModel_rk45(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag); // NOLINT

    vector<Matrix<double, Dynamic, Dynamic> > x_m =
      torsten::pop_pk_generalOdeModel_rk45(model_t::f_, model_t::Ncmt, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, pMatrix_m, biovar_m, tlag_m); // NOLINT

    for (int i = 0; i < np; ++i) {
      torsten::test::test_val(x_m[i], x);
    }
}

TEST_F(TorstenPKTwoCptMPITest, rk45_solver_multiple_IV_doses_data_only) {
  rate[0] = 300;
  for (int i = 0; i < np; ++i) {
    rate_m[i] = rate;
  }

  using model_t = refactor::PKTwoCptModel<double, double, double, double>;

  Matrix<double, Dynamic, Dynamic> x =
    torsten::generalOdeModel_rk45(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag); // NOLINT

  vector<Matrix<double, Dynamic, Dynamic> > x_m =
    torsten::pop_pk_generalOdeModel_rk45(model_t::f_, model_t::Ncmt, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, pMatrix_m, biovar_m, tlag_m); // NOLINT

  for (int i = 0; i < np; ++i) {
    torsten::test::test_val(x_m[i], x);
  }
}

TEST_F(TorstenPKTwoCptMPITest, rk45_solver_multiple_bolus_doses_par_var) {
    vector<vector<vector<var> > > pMatrix_m_v(pMatrix_m.size());
    vector<vector<var> > pMatrix_v(torsten::test::to_var(pMatrix));
    for (int i = 0; i < np; ++i) {
      pMatrix_m_v[i] = torsten::test::to_var(pMatrix_m[i]);
    }

    using model_t = refactor::PKTwoCptModel<double, double, double, double>;

    Matrix<var, Dynamic, Dynamic> x =
      torsten::generalOdeModel_rk45(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix_v, biovar, tlag); // NOLINT

    vector<Matrix<var, Dynamic, Dynamic> > x_m =
      torsten::pop_pk_generalOdeModel_rk45(model_t::f_, model_t::Ncmt, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, pMatrix_m_v, biovar_m, tlag_m); // NOLINT

    for (int i = 0; i < np; ++i) {
      torsten::test::test_grad(pMatrix_m_v[i][0], pMatrix_v[0], x_m[i], x, 1.E-8, 1.E-5);
    }
}

TEST_F(TorstenPKTwoCptMPITest, rk45_solver_multiple_IV_doses_par_var) {
  rate[0] = 300;
  for (int i = 0; i < np; ++i) {
    rate_m[i] = rate;
  }

  vector<vector<vector<var> > > pMatrix_m_v(pMatrix_m.size());
  vector<vector<var> > pMatrix_v(torsten::test::to_var(pMatrix));
  for (int i = 0; i < np; ++i) {
    pMatrix_m_v[i] = torsten::test::to_var(pMatrix_m[i]);
  }

  using model_t = refactor::PKTwoCptModel<double, double, double, double>;

  Matrix<var, Dynamic, Dynamic> x =
    torsten::generalOdeModel_rk45(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix_v, biovar, tlag); // NOLINT

  vector<Matrix<var, Dynamic, Dynamic> > x_m =
    torsten::pop_pk_generalOdeModel_rk45(model_t::f_, model_t::Ncmt, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, pMatrix_m_v, biovar_m, tlag_m); // NOLINT

  for (int i = 0; i < np; ++i) {
    torsten::test::test_grad(pMatrix_m_v[i][0], pMatrix_v[0], x_m[i], x, 1.E-8, 1.E-5);
  }
}

TEST_F(TorstenPKTwoCptMPITest, exception_par_var) {
  // the first individual has parameters that will cause cvode fail
  pMatrix_m[0][0][0] = 1.0E-10;
  pMatrix_m[0][0][1] = 1.0E-10;
  pMatrix_m[0][0][2] = 1.0E-20;
  pMatrix_m[0][0][3] = 1.0E+80;
  pMatrix_m[0][0][4] = 1.0E+70;

  vector<vector<vector<var> > > pMatrix_m_v(pMatrix_m.size());
  for (int i = 0; i < np; ++i) {
    pMatrix_m_v[i] = torsten::test::to_var(pMatrix_m[i]);
  }

  using model_t = refactor::PKTwoCptModel<double, double, double, double>;

  MPI_Comm comm = MPI_COMM_WORLD;
  EXPECT_THROW(torsten::pop_pk_generalOdeModel_bdf(model_t::f_,
                                                   model_t::Ncmt, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m,
                                                   addl_m, ss_m, pMatrix_m_v, biovar_m, tlag_m),
               std::runtime_error);
  MPI_Barrier(comm);
}

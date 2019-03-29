#include <stan/math/rev/mat.hpp>  // FIX ME - includes should be more specific
#include <test/unit/math/torsten/expect_near_matrix_eq.hpp>
#include <test/unit/math/torsten/expect_matrix_eq.hpp>
#include <test/unit/math/torsten/pmx_twocpt_mpi_test_fixture.hpp>
#include <test/unit/math/torsten/pmx_neut_mpi_test_fixture.hpp>
#include <test/unit/math/torsten/util_generalOdeModel.hpp>
#include <test/unit/math/torsten/test_util.hpp>
#include <stan/math/torsten/mpi/envionment.hpp>
#include <stan/math/torsten/PKModelTwoCpt.hpp>
#include <stan/math/torsten/pmx_onecpt_model.hpp>
#include <stan/math/torsten/pmx_twocpt_model.hpp>
#include <stan/math/torsten/to_var.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/mat.hpp>  // FIX ME - include should be more specific
#include <vector>

using std::vector;
using Eigen::Matrix;
using Eigen::Dynamic;
using stan::math::var;

TEST_F(TorstenPopulationPMXTwoCptTest, rk45_solver_multiple_bolus_doses_data_only) {
  using model_t = refactor::PMXTwoCptModel<double, double, double, double>;

  Matrix<double, Dynamic, Dynamic> x =
    torsten::generalOdeModel_rk45(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag); // NOLINT

  vector<Matrix<double, Dynamic, Dynamic> > x_m =
    torsten::pmx_solve_group_rk45(model_t::f_, model_t::Ncmt,
                                        len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                        pMatrix_m,
                                        biovar_m,
                                        tlag_m);

  for (int i = 0; i < np; ++i) {
    torsten::test::test_val(x_m[i], x);
  }
}

TEST_F(TorstenPopulationPMXTwoCptTest, bdf_solver_multiple_bolus_doses_data_only) {
  using model_t = refactor::PMXTwoCptModel<double, double, double, double>;

  Matrix<double, Dynamic, Dynamic> x =
    torsten::generalOdeModel_bdf(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag); // NOLINT

  vector<Matrix<double, Dynamic, Dynamic> > x_m =
    torsten::pmx_solve_group_bdf(model_t::f_, model_t::Ncmt,
                                        len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                        pMatrix_m,
                                        biovar_m,
                                        tlag_m);

  for (int i = 0; i < np; ++i) {
    torsten::test::test_val(x_m[i], x);
  }
}

TEST_F(TorstenPopulationPMXTwoCptTest, adams_solver_multiple_bolus_doses_data_only) {
  using model_t = refactor::PMXTwoCptModel<double, double, double, double>;

  Matrix<double, Dynamic, Dynamic> x =
    torsten::generalOdeModel_adams(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag); // NOLINT

  vector<Matrix<double, Dynamic, Dynamic> > x_m =
    torsten::pmx_solve_group_adams(model_t::f_, model_t::Ncmt,
                                        len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                        pMatrix_m,
                                        biovar_m,
                                        tlag_m);

  for (int i = 0; i < np; ++i) {
    torsten::test::test_val(x_m[i], x);
  }
}

TEST_F(TorstenPopulationPMXTwoCptTest, rk45_solver_multiple_IV_doses_data_only) {
  rate[0] = 300;
  for (int i = 0; i < np; ++i) {
    for (int j = 0; j < nt; ++j) {      
      rate_m[i * nt + j] = rate[j];
    }
  }

  using model_t = refactor::PMXTwoCptModel<double, double, double, double>;

  Matrix<double, Dynamic, Dynamic> x =
    torsten::generalOdeModel_rk45(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag); // NOLINT

  vector<Matrix<double, Dynamic, Dynamic> > x_m =
    torsten::pmx_solve_group_rk45(model_t::f_, model_t::Ncmt,
                                        len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                        pMatrix_m,
                                        biovar_m,
                                        tlag_m);

  for (int i = 0; i < np; ++i) {
    torsten::test::test_val(x_m[i], x);
  }
}

TEST_F(TorstenPopulationPMXTwoCptTest, adams_solver_multiple_IV_doses_data_only) {
  rate[0] = 300;
  for (int i = 0; i < np; ++i) {
    for (int j = 0; j < nt; ++j) {      
      rate_m[i * nt + j] = rate[j];
    }
  }

  using model_t = refactor::PMXTwoCptModel<double, double, double, double>;

  Matrix<double, Dynamic, Dynamic> x =
    torsten::generalOdeModel_adams(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag); // NOLINT

  vector<Matrix<double, Dynamic, Dynamic> > x_m =
    torsten::pmx_solve_group_adams(model_t::f_, model_t::Ncmt,
                                        len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                        pMatrix_m,
                                        biovar_m,
                                        tlag_m);

  for (int i = 0; i < np; ++i) {
    torsten::test::test_val(x_m[i], x);
  }
}

TEST_F(TorstenPopulationPMXTwoCptTest, bdf_solver_multiple_IV_doses_data_only) {
  rate[0] = 300;
  for (int i = 0; i < np; ++i) {
    for (int j = 0; j < nt; ++j) {      
      rate_m[i * nt + j] = rate[j];
    }
  }

  using model_t = refactor::PMXTwoCptModel<double, double, double, double>;

  Matrix<double, Dynamic, Dynamic> x =
    torsten::generalOdeModel_bdf(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag); // NOLINT

  vector<Matrix<double, Dynamic, Dynamic> > x_m =
    torsten::pmx_solve_group_bdf(model_t::f_, model_t::Ncmt,
                                        len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                        pMatrix_m,
                                        biovar_m,
                                        tlag_m);

  for (int i = 0; i < np; ++i) {
    torsten::test::test_val(x_m[i], x);
  }
}

TEST_F(TorstenPopulationPMXTwoCptTest, rk45_solver_multiple_bolus_doses_par_var) {
  vector<vector<var> > pMatrix_m_v(np);
  vector<vector<var> > pMatrix_v(torsten::to_var(pMatrix));
  for (int i = 0; i < np; ++i) {
    pMatrix_m_v[i] = stan::math::to_var(pMatrix[0]);
  }

  using model_t = refactor::PMXTwoCptModel<double, double, double, double>;

  Matrix<var, Dynamic, Dynamic> x =
    torsten::generalOdeModel_rk45(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix_v, biovar, tlag); // NOLINT

  vector<Matrix<var, Dynamic, Dynamic> > x_m =
    torsten::pmx_solve_group_rk45(model_t::f_, model_t::Ncmt,
                                        len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                        pMatrix_m_v,
                                        biovar_m,
                                        tlag_m);

  for (int i = 0; i < np; ++i) {
    torsten::test::test_grad(pMatrix_m_v[i], pMatrix_v[0], x_m[i], x);
  }
}

TEST_F(TorstenPopulationPMXTwoCptTest, adams_solver_multiple_bolus_doses_par_var) {
  vector<vector<var> > pMatrix_m_v(np);
  vector<vector<var> > pMatrix_v(torsten::to_var(pMatrix));
  for (int i = 0; i < np; ++i) {
    pMatrix_m_v[i] = stan::math::to_var(pMatrix[0]);
  }

  using model_t = refactor::PMXTwoCptModel<double, double, double, double>;

  Matrix<var, Dynamic, Dynamic> x =
    torsten::generalOdeModel_adams(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix_v, biovar, tlag); // NOLINT

  vector<Matrix<var, Dynamic, Dynamic> > x_m =
    torsten::pmx_solve_group_adams(model_t::f_, model_t::Ncmt,
                                        len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                        pMatrix_m_v,
                                        biovar_m,
                                        tlag_m);

  for (int i = 0; i < np; ++i) {
    torsten::test::test_grad(pMatrix_m_v[i], pMatrix_v[0], x_m[i], x);
  }
}

TEST_F(TorstenPopulationPMXTwoCptTest, bdf_solver_multiple_bolus_doses_par_var) {
  vector<vector<var> > pMatrix_m_v(np);
  vector<vector<var> > pMatrix_v(torsten::to_var(pMatrix));
  for (int i = 0; i < np; ++i) {
    pMatrix_m_v[i] = stan::math::to_var(pMatrix[0]);
  }

  using model_t = refactor::PMXTwoCptModel<double, double, double, double>;

  Matrix<var, Dynamic, Dynamic> x =
    torsten::generalOdeModel_bdf(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix_v, biovar, tlag); // NOLINT

  vector<Matrix<var, Dynamic, Dynamic> > x_m =
    torsten::pmx_solve_group_bdf(model_t::f_, model_t::Ncmt,
                                        len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                        pMatrix_m_v,
                                        biovar_m,
                                        tlag_m);

  for (int i = 0; i < np; ++i) {
    torsten::test::test_grad(pMatrix_m_v[i], pMatrix_v[0], x_m[i], x);
  }
}

TEST_F(TorstenPopulationPMXTwoCptTest, rk45_solver_multiple_IV_doses_par_var) {
  rate[0] = 300;
  for (int i = 0; i < np; ++i) {
    for (int j = 0; j < nt; ++j) {      
      rate_m[i * nt + j] = rate[j];
    }
  }

  vector<vector<var> > pMatrix_m_v(np);
  vector<vector<var> > pMatrix_v(torsten::to_var(pMatrix));
  for (int i = 0; i < np; ++i) {
    pMatrix_m_v[i] = stan::math::to_var(pMatrix[0]);
  }

  using model_t = refactor::PMXTwoCptModel<double, double, double, double>;

  Matrix<var, Dynamic, Dynamic> x =
    torsten::generalOdeModel_rk45(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix_v, biovar, tlag); // NOLINT

  vector<Matrix<var, Dynamic, Dynamic> > x_m =
    torsten::pmx_solve_group_rk45(model_t::f_, model_t::Ncmt,
                                        len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                        pMatrix_m_v,
                                        biovar_m,
                                        tlag_m);

  for (int i = 0; i < np; ++i) {
    torsten::test::test_grad(pMatrix_m_v[i], pMatrix_v[0], x_m[i], x);
  }
}

TEST_F(TorstenPopulationPMXTwoCptTest, adams_solver_multiple_IV_doses_par_var) {
  rate[0] = 300;
  for (int i = 0; i < np; ++i) {
    for (int j = 0; j < nt; ++j) {      
      rate_m[i * nt + j] = rate[j];
    }
  }

  vector<vector<var> > pMatrix_m_v(np);
  vector<vector<var> > pMatrix_v(torsten::to_var(pMatrix));
  for (int i = 0; i < np; ++i) {
    pMatrix_m_v[i] = stan::math::to_var(pMatrix[0]);
  }

  using model_t = refactor::PMXTwoCptModel<double, double, double, double>;

  Matrix<var, Dynamic, Dynamic> x =
    torsten::generalOdeModel_adams(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix_v, biovar, tlag); // NOLINT

  vector<Matrix<var, Dynamic, Dynamic> > x_m =
    torsten::pmx_solve_group_adams(model_t::f_, model_t::Ncmt,
                                        len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                        pMatrix_m_v,
                                        biovar_m,
                                        tlag_m);

  for (int i = 0; i < np; ++i) {
    torsten::test::test_grad(pMatrix_m_v[i], pMatrix_v[0], x_m[i], x);
  }
}

TEST_F(TorstenPopulationPMXTwoCptTest, bdf_solver_multiple_IV_doses_par_var) {
  rate[0] = 300;
  for (int i = 0; i < np; ++i) {
    for (int j = 0; j < nt; ++j) {      
      rate_m[i * nt + j] = rate[j];
    }
  }

  vector<vector<var> > pMatrix_m_v(np);
  vector<vector<var> > pMatrix_v(torsten::to_var(pMatrix));
  for (int i = 0; i < np; ++i) {
    pMatrix_m_v[i] = stan::math::to_var(pMatrix[0]);
  }

  using model_t = refactor::PMXTwoCptModel<double, double, double, double>;

  Matrix<var, Dynamic, Dynamic> x =
    torsten::generalOdeModel_bdf(model_t::f_, model_t::Ncmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix_v, biovar, tlag); // NOLINT

  vector<Matrix<var, Dynamic, Dynamic> > x_m =
    torsten::pmx_solve_group_bdf(model_t::f_, model_t::Ncmt,
                                        len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                        pMatrix_m_v,
                                        biovar_m,
                                        tlag_m);

  for (int i = 0; i < np; ++i) {
    torsten::test::test_grad(pMatrix_m_v[i], pMatrix_v[0], x_m[i], x);
  }
}

#ifdef TORSTEN_MPI
TEST_F(TorstenPopulationPMXTwoCptTest, exception_sync) {
  using torsten::pmx_solve_group_adams;
  using model_t = refactor::PMXTwoCptModel<double, double, double, double>;
  using torsten::mpi::my_worker;

  torsten::mpi::Envionment::init();

  MPI_Comm comm;
  comm = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  int id = 4;
  pMatrix_m[id][4] = -1e30;
  if (rank == my_worker(id, np, size)) {
    EXPECT_THROW_MSG(pmx_solve_group_adams(model_t::f_, model_t::Ncmt,
                                                  len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                                  pMatrix_m,
                                                  biovar_m,
                                                  tlag_m,
                                                  0, 1e-6, 1e-6, 1e4),
                     std::runtime_error, "CVode(mem, ts[i], y, &t1, CV_NORMAL) failed with error flag -1");
  } else {
    EXPECT_THROW_MSG(pmx_solve_group_adams(model_t::f_, model_t::Ncmt,
                                                  len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                                  pMatrix_m,
                                                  biovar_m,
                                                  tlag_m,
                                                  0, 1e-6, 1e-6, 1e4),
                     std::runtime_error, "received invalid data for id 4");
  }
  MPI_Barrier(comm);

  id = 8;
  pMatrix_m[id][4] = -1e30;
  if (rank == my_worker(4, np, size) || rank == my_worker(8, np, size)) {
    EXPECT_THROW_MSG(pmx_solve_group_adams(model_t::f_, model_t::Ncmt,
                                                  len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                                  pMatrix_m,
                                                  biovar_m,
                                                  tlag_m,
                                                  0, 1e-6, 1e-6, 1e4),
                     std::runtime_error, "CVode(mem, ts[i], y, &t1, CV_NORMAL) failed with error flag -1");
  } else {
    EXPECT_THROW_MSG(pmx_solve_group_adams(model_t::f_, model_t::Ncmt,
                                                  len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                                  pMatrix_m,
                                                  biovar_m,
                                                  tlag_m,
                                                  0, 1e-6, 1e-6, 1e4),
                     std::runtime_error, "received invalid data for id");
  }
  MPI_Barrier(comm);
}
#endif

TEST_F(TorstenPopulationNeutropeniaTest, exception_max_num_steps_fails) {
  double rtol = 1e-12;
  double atol = 1e-12;
  long int max_num_steps = 1e1;

  EXPECT_THROW(torsten::pmx_solve_group_bdf(f, nCmt,
                                                   len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                                   theta_m,
                                                   biovar_m,
                                                   tlag_m,
                                                   0, rtol, atol, max_num_steps),
               std::runtime_error);
}

TEST_F(TorstenPopulationNeutropeniaTest, exception_var_max_num_steps_fails) {
  double rtol = 1e-12;
  double atol = 1e-12;
  long int max_num_steps = 1e1;

  std::vector<std::vector<stan::math::var> > theta_m_v(np);
  for (int i = 0; i < np; ++i) {
    theta_m_v[i] = stan::math::to_var(theta[0]);
  }

  EXPECT_THROW(torsten::pmx_solve_group_bdf(f, nCmt,
                                                   len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                                   theta_m_v, biovar_m, tlag_m,
                                                   0, rtol, atol, max_num_steps),
               std::runtime_error);
}

TEST_F(TorstenPopulationNeutropeniaTest, domain_error) {
  using torsten::pmx_solve_group_bdf;

  torsten::mpi::Envionment::init();

#ifdef TORSTEN_MPI
  MPI_Comm comm;
  comm = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
#endif

  double rtol = 1e-12;
  double atol = 1e-12;
  long int max_num_steps = 1e1;

  int id = 4;
  for (int j = 0; j < nt; ++j) {      
    rate_m[id * nt + j] = std::numeric_limits<double>::infinity();
  }
  EXPECT_THROW_MSG(pmx_solve_group_bdf(f, nCmt,
                                              len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                              theta_m, biovar_m, tlag_m,
                                              0, rtol, atol, max_num_steps),
                   std::domain_error,
                   "rate[109] is inf, but must be finite");
#ifdef TORSTEN_MPI
  MPI_Barrier(comm);
#endif

  for (int j = 0; j < nt; ++j) {      
    rate_m[id * nt + j] = 130;
  }
  theta_m[id][3] = std::numeric_limits<double>::infinity();
  EXPECT_THROW_MSG(pmx_solve_group_bdf(f, nCmt,
                                              len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                              theta_m, biovar_m, tlag_m,
                                              0, rtol, atol, max_num_steps),
                   std::domain_error,
                   "parameters[4] is inf, but must be finite!");
#ifdef TORSTEN_MPI
  MPI_Barrier(comm);
#endif

  theta_m[id][3] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW_MSG(pmx_solve_group_bdf(f, nCmt,
                                              len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                              theta_m, biovar_m, tlag_m,
                                              0, rtol, atol, max_num_steps),
                   std::domain_error,
                   "parameters[4] is nan, but must be finite!");
#ifdef TORSTEN_MPI
  MPI_Barrier(comm);
#endif

  theta_m[id][3] = 1.0;
  biovar_m[id][3] = -1.0;
  EXPECT_THROW_MSG(pmx_solve_group_bdf(f, nCmt,
                                              len, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, // NOLINT
                                              theta_m, biovar_m, tlag_m,
                                              0, rtol, atol, max_num_steps),
                   std::domain_error,
                   "bioavailability[4] is -1, but must be >= 0");
#ifdef TORSTEN_MPI
  MPI_Barrier(comm);
#endif
}

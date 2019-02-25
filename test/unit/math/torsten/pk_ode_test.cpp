#include <stan/math/rev/mat.hpp>  // FIX ME - includes should be more specific
#include <gtest/gtest.h>
#include <test/unit/math/torsten/pk_ode_test_fixture.hpp>
#include <test/unit/math/torsten/pk_onecpt_test_fixture.hpp>
#include <test/unit/math/torsten/pk_twocpt_test_fixture.hpp>
#include <test/unit/math/torsten/pk_friberg_karlsson_test_fixture.hpp>
#include <test/unit/math/torsten/expect_near_matrix_eq.hpp>
#include <test/unit/math/torsten/expect_matrix_eq.hpp>
#include <stan/math/torsten/generalOdeModel_rk45.hpp>
#include <stan/math/torsten/generalOdeModel_bdf.hpp>
#include <stan/math/torsten/pk_twocpt_model.hpp>
#include <stan/math/torsten/pk_onecpt_model.hpp>
#include <stan/math/torsten/pk_ode_model.hpp>
#include <test/unit/math/torsten/util_generalOdeModel.hpp>


auto f  = refactor::PKOneCptModel<double,double,double,double>::f_;
auto f2 = refactor::PKTwoCptModel<double,double,double,double>::f_;

TEST_F(TorstenOneCptTest, ode_with_steady_state_zero_rate) {
  // Steady state induced by multiple bolus doses (SS = 1, rate = 0)
  using std::vector;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::Dynamic;

  time[0] = 0.0;
  time[1] = 0.0;
  for(int i = 2; i < 10; i++) time[i] = time[i - 1] + 5;

  amt[0] = 1200;
  addl[0] = 10;
  ss[0] = 1;

  double rel_tol = 1e-8, abs_tol = 1e-8;
  long int max_num_steps = 1e8;
  MatrixXd x_rk45 = torsten::generalOdeModel_rk45(f, nCmt,
                                time, amt, rate, ii, evid, cmt, addl, ss,
                                pMatrix, biovar, tlag,
                                0,
                                rel_tol, abs_tol, max_num_steps);

  MatrixXd x_bdf = torsten::generalOdeModel_bdf(f, nCmt,
                              time, amt, rate, ii, evid, cmt, addl, ss,
                              pMatrix, biovar, tlag,
                              0,
                              rel_tol, abs_tol, max_num_steps);

  Eigen::MatrixXd x(10, 2);
  x << 1200.0      , 384.7363,
    1200.0      , 384.7363,
    2.974504    , 919.6159,
    7.373062e-3 , 494.0040,
    3.278849e+1 , 1148.4725,
    8.127454e-2 , 634.2335,
    3.614333e+2 , 1118.2043,
    8.959035e-1 , 813.4883,
    2.220724e-3 , 435.9617,
    9.875702    , 1034.7998;

  torsten::test::test_val(x_rk45, x, 1e-6, 1e-4);
  torsten::test::test_val(x_bdf, x, 1e-4, 1e-4);

  // Test AutoDiff against FiniteDiff
  double diff = 1e-8, diff2 = 5e-3;
  test_generalOdeModel2(f, nCmt,
                       time, amt, rate, ii, evid, cmt, addl, ss,
                       pMatrix, biovar, tlag,
                       rel_tol, abs_tol, max_num_steps, diff, diff2, "rk45");
  test_generalOdeModel2(f, nCmt,
                       time, amt, rate, ii, evid, cmt, addl, ss,
                       pMatrix, biovar, tlag,
                       rel_tol, abs_tol, max_num_steps, diff, diff2, "bdf");
}

TEST_F(TorstenOneCptTest, ode_with_steady_state_zero_rate_par_var) {
  // Steady state induced by multiple bolus doses (SS = 1, rate = 0)
  using std::vector;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::Dynamic;

  time[0] = 0.0;
  time[1] = 0.0;
  for(int i = 2; i < 10; i++) time[i] = time[i - 1] + 5;

  amt[0] = 1200;
  addl[0] = 10;
  ss[0] = 1;

  double rel_tol = 1e-8, abs_tol = 1e-8;
  long int max_num_steps = 1e8;

  {
    auto f1 = [&] (std::vector<double>& theta) {
      std::vector<std::vector<double> > theta1{theta};
      return torsten::generalOdeModel_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, theta1, biovar, tlag,
                                           0, rel_tol, abs_tol, max_num_steps);
    };

    auto f2 = [&] (std::vector<stan::math::var>& theta) {
      std::vector<std::vector<stan::math::var> > theta1{theta};
      return torsten::generalOdeModel_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, theta1, biovar, tlag,
                                           0, rel_tol, abs_tol, max_num_steps);
    };
    
    torsten::test::test_grad(f1, f2, pMatrix[0], 2e-5, 1e-6, 1e-3, 1e-3);
  }

  {
    auto f1 = [&] (std::vector<double>& theta) {
      std::vector<std::vector<double> > theta1{theta};
      return torsten::generalOdeModel_bdf(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, theta1, biovar, tlag,
                                           0, rel_tol, abs_tol, max_num_steps);
    };

    auto f2 = [&] (std::vector<stan::math::var>& theta) {
      std::vector<std::vector<stan::math::var> > theta1{theta};
      return torsten::generalOdeModel_bdf(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, theta1, biovar, tlag,
                                           0, rel_tol, abs_tol, max_num_steps);
    };
    
    torsten::test::test_grad(f1, f2, pMatrix[0], 2e-5, 1e-6, 1e-3, 1e-3);
  }
}

TEST_F(TorstenOneCptTest, ode_with_steady_state_zero_rate_biovar_var) {
  // Steady state induced by multiple bolus doses (SS = 1, rate = 0)
  using std::vector;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::Dynamic;

  time[0] = 0.0;
  time[1] = 0.0;
  for(int i = 2; i < 10; i++) time[i] = time[i - 1] + 5;

  amt[0] = 1200;
  addl[0] = 10;
  ss[0] = 1;

  double rel_tol = 1e-8, abs_tol = 1e-8;
  long int max_num_steps = 1e8;

  {
    auto f1 = [&] (std::vector<double>& x) {
      std::vector<std::vector<double> > biovar1{x};
      return torsten::generalOdeModel_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar1, tlag,
                                           0, rel_tol, abs_tol, max_num_steps);
    };

    auto f2 = [&] (std::vector<stan::math::var>& x) {
      std::vector<std::vector<stan::math::var> > biovar1{x};
      return torsten::generalOdeModel_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar1, tlag,
                                           0, rel_tol, abs_tol, max_num_steps);
    };
    
    torsten::test::test_grad(f1, f2, biovar[0], 2e-5, 1e-6, 1e-3, 1e-3);
  }

  {
    auto f1 = [&] (std::vector<double>& x) {
      std::vector<std::vector<double> > biovar1{x};
      return torsten::generalOdeModel_bdf(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar1, tlag,
                                          0, rel_tol, abs_tol, max_num_steps);
    };

    auto f2 = [&] (std::vector<stan::math::var>& x) {
      std::vector<std::vector<stan::math::var> > biovar1{x};
      return torsten::generalOdeModel_bdf(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar1, tlag,
                                          0, rel_tol, abs_tol, max_num_steps);
    };
    
    torsten::test::test_grad(f1, f2, biovar[0], 2e-5, 1e-6, 1e-3, 1e-3);
  }
}

TEST_F(TorstenOneCptTest, single_tlag_event) {
  // Steady state induced by multiple bolus doses (SS = 1, rate = 0)
  using std::vector;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::Dynamic;

  nt = 1;
  time.resize(nt);
  amt.resize(nt);
  rate.resize(nt);
  cmt.resize(nt);
  evid.resize(nt);
  ii.resize(nt);
  addl.resize(nt);
  ss.resize(nt);
  evid[0] = 1;
  cmt[0] = 1;
  ii[0] = 0;
  addl[0] = 0;
  time[0] = 0.0;
  tlag[0][0] = 1.5;

  // double rel_tol = 1e-8, abs_tol = 1e-8;
  // long int max_num_steps = 1e8;

  // auto x = torsten::generalOdeModel_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag,
  //                                        0, rel_tol, abs_tol, max_num_steps);

  // {
  //   auto f1 = [&] (std::vector<double>& x) {
  //     std::vector<std::vector<double> > tlag1(nt, {0, 0});
  //     tlag1[1][0] = x[0];
  //     tlag1[1][1] = x[1];
  //     return torsten::generalOdeModel_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag1,
  //                                          0, rel_tol, abs_tol, max_num_steps);
  //   };

  //   auto f2 = [&] (std::vector<stan::math::var>& x) {
  //     std::vector<std::vector<stan::math::var> > tlag1(nt, {0, 0});
  //     tlag1[1][0] = x[0];
  //     tlag1[1][1] = x[1];
  //     return torsten::generalOdeModel_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag1,
  //                                          0, rel_tol, abs_tol, max_num_steps);
  //   };
    
  //   torsten::test::test_grad(f1, f2, tlag[1], 2e-5, 1e-6, 1e-3, 1e-3);
  // }
}

TEST_F(TorstenOneCptTest, ode_with_single_bolus_tlag) {
  nt = 2;
  time.resize(nt);
  amt.resize(nt);
  rate.resize(nt);
  cmt.resize(nt);
  evid.resize(nt);
  ii.resize(nt);
  addl.resize(nt);
  ss.resize(nt);
  evid[0] = 1;
  cmt[0] = 1;
  ii[0] = 0;
  addl[0] = 0;
  time[0] = 0.0;
  tlag[0][0] = 1.5;

  time[1] = 2.5;

  double rel_tol = 1e-8, abs_tol = 1e-8;
  long int max_num_steps = 1e8;
  std::vector<std::vector<stan::math::var> > tlag_v(1, stan::math::to_var(tlag[0]));
  auto x = torsten::generalOdeModel_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag_v,
                                         0, rel_tol, abs_tol, max_num_steps);
  // std::vector<double> g;
  // stan::math::set_zero_all_adjoints();
  // x(1, 0).grad(tlag_v[0], g);
  // stan::math::set_zero_all_adjoints();
  // std::cout << "taki test: " << g[0] << " " << g[1] << "\n";
  // x(1, 1).grad(tlag_v[0], g);
  // std::cout << "taki test: " << g[0] << " " << g[1] << "\n";
  // stan::math::set_zero_all_adjoints();

  // {
  //   auto f1 = [&] (std::vector<double>& x) {
  //     std::vector<std::vector<double> > tlag1(1, {x[0], x[1]});
  //     return torsten::generalOdeModel_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag1,
  //                                          0, rel_tol, abs_tol, max_num_steps);
  //   };
  //   auto f2 = [&] (std::vector<stan::math::var>& x) {
  //     std::vector<std::vector<stan::math::var> > tlag1(1, {x[0], x[1]});
  //     return torsten::generalOdeModel_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag1,
  //                                          0, rel_tol, abs_tol, max_num_steps);
  //   };
  //   torsten::test::test_grad(f1, f2, tlag[0], 2e-5, 1e-6, 1e-3, 1e-3);
  // }

  // compare with manually generated events history
  nt = 3;
  time.resize(nt);
  amt.resize(nt);
  rate.resize(nt);
  cmt.resize(nt);
  evid.resize(nt);
  ii.resize(nt);
  addl.resize(nt);
  ss.resize(nt);
  
  time[0] = 0.0;
  time[1] = 1.5;
  time[2] = 2.5;

  amt[0] = 1000.0;
  amt[1] = 1000.0;
  amt[2] = 0.0;

  evid[0] = 2;
  evid[1] = 1;
  evid[2] = 0;

  cmt[0] = 1;
  cmt[1] = 1;
  cmt[2] = 2;

  tlag[0][0] = 0.0;

  auto x0 = torsten::generalOdeModel_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag,
                                          0, rel_tol, abs_tol, max_num_steps);

  // the last time step's solution should be identical
  // for (int i = 0; i < nCmt; ++i) {
  //   EXPECT_FLOAT_EQ(x(x.rows() - 1, i), x0(x0.rows() - 1, i));
  // }
}

TEST_F(TorstenOneCptTest, ode_with_steady_state_zero_rate_tlag_var) {
  // Steady state induced by multiple bolus doses (SS = 1, rate = 0)
  using std::vector;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::Dynamic;

  time[0] = 0.0;
  time[1] = 0.0;
  for(int i = 2; i < 10; i++) time[i] = time[i - 1] + 5;

  amt[0] = 0.0;
  evid[0] = 0;
  evid[1] = 1;
  amt[1] = 1200;
  cmt[1] = 1;
  ss[1] = 1;
  tlag.resize(nt);
  for (int i = 0; i < nt; ++i) {
    tlag[i].resize(nCmt);
    tlag[i][0] = 0.0;
    tlag[i][1] = 0.0;
  }
  tlag[1][0] = 1.0;
  tlag[1][1] = 0.0;
  ii[0] = 0.0;
  addl[0] = 0;

  // double rel_tol = 1e-8, abs_tol = 1e-8;
  // long int max_num_steps = 1e8;

  // {
  //   auto f1 = [&] (std::vector<double>& x) {
  //     std::vector<std::vector<double> > tlag1(nt, {0, 0});
  //     tlag1[1][0] = x[0];
  //     tlag1[1][1] = x[1];
  //     return torsten::generalOdeModel_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag1,
  //                                          0, rel_tol, abs_tol, max_num_steps);
  //   };

  //   auto f2 = [&] (std::vector<stan::math::var>& x) {
  //     std::vector<std::vector<stan::math::var> > tlag1(nt, {0, 0});
  //     tlag1[1][0] = x[0];
  //     tlag1[1][1] = x[1];
  //     return torsten::generalOdeModel_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag1,
  //                                          0, rel_tol, abs_tol, max_num_steps);
  //   };
    
  //   torsten::test::test_grad(f1, f2, tlag[1], 2e-5, 1e-6, 1e-3, 1e-3);
  // }

  // {
  //   auto f1 = [&] (std::vector<double>& x) {
  //     std::vector<std::vector<double> > tlag1{x};
  //     return torsten::generalOdeModel_bdf(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag1,
  //                                          0, rel_tol, abs_tol, max_num_steps);
  //   };

  //   auto f2 = [&] (std::vector<stan::math::var>& x) {
  //     std::vector<std::vector<stan::math::var> > tlag1{x};
  //     return torsten::generalOdeModel_bdf(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag1,
  //                                          0, rel_tol, abs_tol, max_num_steps);
  //   };
    
  //   torsten::test::test_grad(f1, f2, tlag[0], 2e-5, 1e-6, 1e-3, 1e-3);
  // }
}

TEST_F(TorstenOneCptTest, ode_with_steady_state_nonzero_rate) {
  // Steady state with constant rate infusion (SS = 1, rate != 0, ii = 0)
  using std::vector;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::Dynamic;

  vector<double> time(10);
  time[0] = 0.0;
  for(int i = 1; i < 10; i++) time[i] = time[i - 1] + 2.5;

  amt[0] = 0.0;
  rate[0] = 150;
  ii[0] = 0.0;
  addl[0] = 0;
  ss[0] = 1;

  double rel_tol = 1e-8, abs_tol = 1e-8;
  long int max_num_steps = 1e8;
  MatrixXd x_rk45 = torsten::generalOdeModel_rk45(f, nCmt,
                                time, amt, rate, ii, evid, cmt, addl, ss,
                                pMatrix, biovar, tlag,
                                0,
                                rel_tol, abs_tol, max_num_steps);

  MatrixXd x_bdf = torsten::generalOdeModel_bdf(f, nCmt,
                              time, amt, rate, ii, evid, cmt, addl, ss,
                              pMatrix, biovar, tlag,
                              0,
                              rel_tol, abs_tol, max_num_steps);

  MatrixXd x = torsten::PKModelOneCpt(time, amt, rate, ii, evid, cmt, addl, ss,
                                      pMatrix, biovar, tlag);

  torsten::test::test_val(x_rk45, x, 1e-4, 1e-4);
  torsten::test::test_val(x_bdf, x, 1e-3, 1e-4);

  // Test AutoDiff against FiniteDiff
  double diff = 1e-8, diff2 = 5e-3;
  test_generalOdeModel2(f, nCmt,
                       time, amt, rate, ii, evid, cmt, addl, ss,
                       pMatrix, biovar, tlag,
                       rel_tol, abs_tol, max_num_steps, diff, diff2, "rk45");
  test_generalOdeModel2(f, nCmt,
                       time, amt, rate, ii, evid, cmt, addl, ss,
                       pMatrix, biovar, tlag,
                       rel_tol, abs_tol, max_num_steps, diff, diff2, "bdf");
}

TEST_F(TorstenOneCptTest, Steady_state_with_multiple_truncated_iv) {
  // (SS = 1, rate != 0, ii > 0)
  using std::vector;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::Dynamic;

  vector<double> time(10);
  time[0] = 0.0;
  for(int i = 1; i < 10; i++) time[i] = time[i - 1] + 2.5;

  amt[0] = 1200;
  rate[0] = 150;
  ii[0] = 16;
  addl[0] = 0;
  ss[0] = 1;

  double rel_tol = 1e-8, abs_tol = 1e-8;
  long int max_num_steps = 1e8;
  MatrixXd x_rk45 = torsten::generalOdeModel_rk45(f, nCmt,
                                time, amt, rate, ii, evid, cmt, addl, ss,
                                pMatrix, biovar, tlag,
                                0,
                                rel_tol, abs_tol, max_num_steps);

  MatrixXd x_bdf = torsten::generalOdeModel_bdf(f, nCmt,
                              time, amt, rate, ii, evid, cmt, addl, ss,
                              pMatrix, biovar, tlag,
                              0,
                              rel_tol, abs_tol, max_num_steps);

  MatrixXd x(10, 2);
  x << 8.465519e-03, 360.2470,
             1.187770e+02, 490.4911,
             1.246902e+02, 676.1759,
             1.249846e+02, 816.5263,
             1.133898e+01, 750.0054,
             5.645344e-01, 557.3459,
             2.810651e-02, 408.1926,
             1.399341e-03, 298.6615,
             6.966908e-05, 218.5065,
             3.468619e-06, 159.8628;

  torsten::test::test_val(x_rk45, x, 1e-4, 1e-4);
  torsten::test::test_val(x_bdf, x, 1e-3, 1e-4);

  // Test AutoDiff against FiniteDiff
  // Currently, torsten does not handle multiple truncated infusions case when
  // amt * F is a parameter (this scenario returns an exception and is not
  // tested here).
  double diff = 1e-8, diff2 = 5e-3;
  test_generalOdeModel2(f, nCmt,
                       time, amt, rate, ii, evid, cmt, addl, ss,
                       pMatrix, biovar, tlag,
                       rel_tol, abs_tol, max_num_steps, diff, diff2, "rk45",
                       2);

  // diff_bdf2 determined empirically
  double diff_bdf2 = 1e-2;
  test_generalOdeModel2(f, nCmt,
                       time, amt, rate, ii, evid, cmt, addl, ss,
                       pMatrix, biovar, tlag,
                       rel_tol, abs_tol, max_num_steps, diff, diff_bdf2,
                       "bdf", 2);
}

TEST_F(TorstenOneCptTest, multiple_dose) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::Dynamic;

  ii[0] = 12;
  addl[0] = 14;

  double rel_tol = 1e-8, abs_tol = 1e-8;
  long int max_num_steps = 1e8;
  MatrixXd x_rk45 = torsten::generalOdeModel_rk45(f, nCmt,
                                time, amt, rate, ii, evid, cmt, addl, ss,
                                pMatrix, biovar, tlag,
                                0,
                                rel_tol, abs_tol, max_num_steps);

  MatrixXd x_bdf = torsten::generalOdeModel_bdf(f, nCmt,
                              time, amt, rate, ii, evid, cmt, addl, ss,
                              pMatrix, biovar, tlag,
                              0,
                              rel_tol, abs_tol, max_num_steps);

  MatrixXd x(10, nCmt);
  x << 1000.0, 0.0,
    740.8182, 254.97490,
    548.8116, 436.02020,
    406.5697, 562.53846,
    301.1942, 648.89603,
    223.1302, 705.72856,
    165.2989, 740.90816,
    122.4564, 760.25988,
    90.71795, 768.09246,
    8.229747, 667.87079;

  torsten::test::test_val(x_rk45, x, 1e-5, 1e-5);
  torsten::test::test_val(x_bdf, x, 1e-5, 1e-5);

  // Test AutoDiff against FiniteDiff
  double diff = 1e-8, diff2 = 5e-3;
  test_generalOdeModel2(f, nCmt,
                       time, amt, rate, ii, evid, cmt, addl, ss,
                       pMatrix, biovar, tlag,
                       rel_tol, abs_tol, max_num_steps, diff, diff2, "rk45");
  test_generalOdeModel2(f, nCmt,
                       time, amt, rate, ii, evid, cmt, addl, ss,
                       pMatrix, biovar, tlag,
                       rel_tol, abs_tol, max_num_steps, diff, diff2, "bdf");
}

TEST_F(TorstenOneCptTest, multiple_dose_overload) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::Dynamic;

  double rel_tol = 1e-8, abs_tol = 1e-8;
  long int max_num_steps = 1e8;

  Matrix<double, Eigen::Dynamic, Eigen::Dynamic> x_rk45_122, x_rk45_112,
    x_rk45_111, x_rk45_121, x_rk45_212, x_rk45_211, x_rk45_221;
  x_rk45_122 = torsten::generalOdeModel_rk45(f, nCmt,
                                time, amt, rate, ii, evid, cmt, addl, ss,
                                pMatrix[0], biovar, tlag,
                                0,
                                rel_tol, abs_tol, max_num_steps);

  x_rk45_112 = torsten::generalOdeModel_rk45(f, nCmt,
                                    time, amt, rate, ii, evid, cmt, addl, ss,
                                    pMatrix[0], biovar[0], tlag,
                                    0,
                                    rel_tol, abs_tol, max_num_steps);

  x_rk45_111 = torsten::generalOdeModel_rk45(f, nCmt,
                                    time, amt, rate, ii, evid, cmt, addl, ss,
                                    pMatrix[0], biovar[0], tlag[0],
                                    0,
                                    rel_tol, abs_tol, max_num_steps);

  x_rk45_121 = torsten::generalOdeModel_rk45(f, nCmt,
                                    time, amt, rate, ii, evid, cmt, addl, ss,
                                    pMatrix[0], biovar, tlag[0],
                                    0,
                                    rel_tol, abs_tol, max_num_steps);

  x_rk45_212 = torsten::generalOdeModel_rk45(f, nCmt,
                                    time, amt, rate, ii, evid, cmt, addl, ss,
                                    pMatrix, biovar[0], tlag,
                                    0,
                                    rel_tol, abs_tol, max_num_steps);

  x_rk45_211 = torsten::generalOdeModel_rk45(f, nCmt,
                                    time, amt, rate, ii, evid, cmt, addl, ss,
                                    pMatrix, biovar[0], tlag[0],
                                    0,
                                    rel_tol, abs_tol, max_num_steps);

  x_rk45_221 = torsten::generalOdeModel_rk45(f, nCmt,
                                    time, amt, rate, ii, evid, cmt, addl, ss,
                                    pMatrix, biovar, tlag[0],
                                    0,
                                    rel_tol, abs_tol, max_num_steps);


  Matrix<double, Eigen::Dynamic, Eigen::Dynamic> x_bdf_122, x_bdf_112,
    x_bdf_111, x_bdf_121, x_bdf_212, x_bdf_211, x_bdf_221;
  x_bdf_122 = torsten::generalOdeModel_bdf(f, nCmt,
                                    time, amt, rate, ii, evid, cmt, addl, ss,
                                    pMatrix[0], biovar, tlag,
                                    0,
                                    rel_tol, abs_tol, max_num_steps);

  x_bdf_112 = torsten::generalOdeModel_bdf(f, nCmt,
                                    time, amt, rate, ii, evid, cmt, addl, ss,
                                    pMatrix[0], biovar[0], tlag,
                                    0,
                                    rel_tol, abs_tol, max_num_steps);

  x_bdf_111 = torsten::generalOdeModel_bdf(f, nCmt,
                                    time, amt, rate, ii, evid, cmt, addl, ss,
                                    pMatrix[0], biovar[0], tlag[0],
                                    0,
                                    rel_tol, abs_tol, max_num_steps);

  x_bdf_121 = torsten::generalOdeModel_bdf(f, nCmt,
                                    time, amt, rate, ii, evid, cmt, addl, ss,
                                    pMatrix[0], biovar, tlag[0],
                                    0,
                                    rel_tol, abs_tol, max_num_steps);

  x_bdf_212 = torsten::generalOdeModel_bdf(f, nCmt,
                                    time, amt, rate, ii, evid, cmt, addl, ss,
                                    pMatrix, biovar[0], tlag,
                                    0,
                                    rel_tol, abs_tol, max_num_steps);

  x_bdf_211 = torsten::generalOdeModel_bdf(f, nCmt,
                                    time, amt, rate, ii, evid, cmt, addl, ss,
                                    pMatrix, biovar[0], tlag[0],
                                    0,
                                    rel_tol, abs_tol, max_num_steps);

  x_bdf_221 = torsten::generalOdeModel_bdf(f, nCmt,
                                    time, amt, rate, ii, evid, cmt, addl, ss,
                                    pMatrix, biovar, tlag[0],
                                    0,
                                    rel_tol, abs_tol, max_num_steps);

  MatrixXd x(10, 2);
  x << 1000.0, 0.0,
             740.8182, 254.97490,
             548.8116, 436.02020,
             406.5697, 562.53846,
             301.1942, 648.89603,
             223.1302, 705.72856,
             165.2989, 740.90816,
             122.4564, 760.25988,
             90.71795, 768.09246,
             8.229747, 667.87079;

  torsten::test::test_val(x, x_rk45_122, 1e-5, 1e-5);
  torsten::test::test_val(x, x_rk45_112, 1e-5, 1e-5);
  torsten::test::test_val(x, x_rk45_111, 1e-5, 1e-5);
  torsten::test::test_val(x, x_rk45_121, 1e-5, 1e-5);
  torsten::test::test_val(x, x_rk45_212, 1e-5, 1e-5);
  torsten::test::test_val(x, x_rk45_211, 1e-5, 1e-5);
  torsten::test::test_val(x, x_rk45_221, 1e-5, 1e-5);

  torsten::test::test_val(x, x_bdf_122, 1e-5, 1e-5);
  torsten::test::test_val(x, x_bdf_112, 1e-5, 1e-5);
  torsten::test::test_val(x, x_bdf_111, 1e-5, 1e-5);
  torsten::test::test_val(x, x_bdf_121, 1e-5, 1e-5);
  torsten::test::test_val(x, x_bdf_212, 1e-5, 1e-5);
  torsten::test::test_val(x, x_bdf_211, 1e-5, 1e-5);
  torsten::test::test_val(x, x_bdf_221, 1e-5, 1e-5);
}

TEST_F(TorstenOneCptTest, generalOdeModel_signature_test) {
  using stan::math::var;
  using std::vector;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::Dynamic;

  double rel_err = 1e-4;

  vector<vector<var> > pMatrix_v(1);
  pMatrix_v[0].resize(3);
  pMatrix_v[0][0] = 10;  // CL
  pMatrix_v[0][1] = 80;  // Vc
  pMatrix_v[0][2] = 1.2;  // ka

  vector<vector<var> > biovar_v(1);
  biovar_v[0].resize(nCmt);
  biovar_v[0][0] = 1;  // F1
  biovar_v[0][1] = 1;  // F2

  vector<vector<var> > tlag_v(1);
  tlag_v[0].resize(nCmt);
  tlag_v[0][0] = 0;  // tlag1
  tlag_v[0][1] = 0;  // tlag2

  double rel_tol = 1e-8, abs_tol = 1e-8;
  long int max_num_steps = 1e8;

  MatrixXd amounts(10, 2);
  amounts << 1000.0, 0.0,
             740.8182, 254.97490,
             548.8116, 436.02020,
             406.5697, 562.53846,
             301.1942, 648.89603,
             223.1302, 705.72856,
             165.2989, 740.90816,
             122.4564, 760.25988,
             90.71795, 768.09246,
             8.229747, 667.87079;

  // RK45
  vector<Matrix<var, Dynamic, Dynamic> > x_rk45_122(7);
  x_rk45_122[0] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar, tlag,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_122[1] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar_v, tlag,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_122[2] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar, tlag_v,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_122[3] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar_v, tlag_v,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_122[4] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar_v, tlag,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_122[5] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar_v, tlag_v,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_122[5] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar, tlag_v,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);

  for (size_t i = 0; i < x_rk45_122.size(); i++)
    for (int j = 0; j < x_rk45_122[i].rows(); j++)
      for (int k = 0; k < x_rk45_122[i].cols(); k++)
        EXPECT_NEAR(amounts(j, k), x_rk45_122[i](j, k).val(),
          std::max(amounts(j, k), x_rk45_122[i](j, k).val()) * rel_err);


  vector<Matrix<var, Dynamic, Dynamic> > x_rk45_112(7);
  x_rk45_112[0] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar[0], tlag,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_112[1] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar_v[0], tlag,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_112[2] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar[0], tlag_v,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_112[3] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar_v[0], tlag_v,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_112[4] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar_v[0], tlag,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_112[5] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar_v[0], tlag_v,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_112[5] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar[0], tlag_v,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);

  for (size_t i = 0; i < x_rk45_112.size(); i++)
    for (int j = 0; j < x_rk45_112[i].rows(); j++)
      for (int k = 0; k < x_rk45_112[i].cols(); k++)
        EXPECT_NEAR(amounts(j, k), x_rk45_112[i](j, k).val(),
                    std::max(amounts(j, k), x_rk45_112[i](j, k).val()) * rel_err);


  vector<Matrix<var, Dynamic, Dynamic> > x_rk45_121(7);
  x_rk45_121[0] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar, tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_121[1] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar_v, tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_121[2] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar, tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_121[3] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar_v, tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_121[4] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar_v, tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_121[5] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar_v, tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_121[5] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar, tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);

  for (size_t i = 0; i < x_rk45_121.size(); i++)
    for (int j = 0; j < x_rk45_121[i].rows(); j++)
      for (int k = 0; k < x_rk45_121[i].cols(); k++)
        EXPECT_NEAR(amounts(j, k), x_rk45_121[i](j, k).val(),
                    std::max(amounts(j, k), x_rk45_121[i](j, k).val()) * rel_err);


  vector<Matrix<var, Dynamic, Dynamic> > x_rk45_111(7);
  x_rk45_111[0] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar[0], tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_111[1] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar_v[0], tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_111[2] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar[0], tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_111[3] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar_v[0], tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_111[4] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar_v[0], tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_111[5] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar_v[0], tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_111[5] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar[0], tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);

  for (size_t i = 0; i < x_rk45_111.size(); i++)
    for (int j = 0; j < x_rk45_111[i].rows(); j++)
      for (int k = 0; k < x_rk45_111[i].cols(); k++)
        EXPECT_NEAR(amounts(j, k), x_rk45_111[i](j, k).val(),
                    std::max(amounts(j, k), x_rk45_111[i](j, k).val()) * rel_err);


  vector<Matrix<var, Dynamic, Dynamic> > x_rk45_211(7);
  x_rk45_211[0] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v, biovar[0], tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_211[1] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v, biovar_v[0], tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_211[2] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v, biovar[0], tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_211[3] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v, biovar_v[0], tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_211[4] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar_v[0], tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_211[5] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar_v[0], tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_211[5] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar[0], tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);

  for (size_t i = 0; i < x_rk45_211.size(); i++)
    for (int j = 0; j < x_rk45_211[i].rows(); j++)
      for (int k = 0; k < x_rk45_211[i].cols(); k++)
        EXPECT_NEAR(amounts(j, k), x_rk45_211[i](j, k).val(),
                    std::max(amounts(j, k), x_rk45_211[i](j, k).val()) * rel_err);


  vector<Matrix<var, Dynamic, Dynamic> > x_rk45_221(7);
  x_rk45_221[0] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v, biovar, tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_221[1] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v, biovar_v, tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_221[2] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v, biovar, tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_221[3] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v, biovar_v, tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_221[4] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar_v, tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_221[5] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar_v, tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_rk45_221[5] = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar, tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);

  for (size_t i = 0; i < x_rk45_221.size(); i++)
    for (int j = 0; j < x_rk45_221[i].rows(); j++)
      for (int k = 0; k < x_rk45_221[i].cols(); k++)
        EXPECT_NEAR(amounts(j, k), x_rk45_221[i](j, k).val(),
                    std::max(amounts(j, k), x_rk45_221[i](j, k).val()) * rel_err);


  // BDF
  vector<Matrix<var, Dynamic, Dynamic> > x_bdf_122(7);
  x_bdf_122[0] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar, tlag,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_122[1] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar_v, tlag,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_122[2] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar, tlag_v,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_122[3] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar_v, tlag_v,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_122[4] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar_v, tlag,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_122[5] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar_v, tlag_v,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_122[5] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar, tlag_v,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);

  for (size_t i = 0; i < x_bdf_122.size(); i++)
    for (int j = 0; j < x_bdf_122[i].rows(); j++)
      for (int k = 0; k < x_bdf_122[i].cols(); k++)
        EXPECT_NEAR(amounts(j, k), x_bdf_122[i](j, k).val(),
                    std::max(amounts(j, k), x_bdf_122[i](j, k).val()) * rel_err);


  vector<Matrix<var, Dynamic, Dynamic> > x_bdf_112(7);
  x_bdf_112[0] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar[0], tlag,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_112[1] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar_v[0], tlag,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_112[2] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar[0], tlag_v,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_112[3] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar_v[0], tlag_v,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_112[4] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar_v[0], tlag,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_112[5] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar_v[0], tlag_v,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_112[5] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar[0], tlag_v,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);

  for (size_t i = 0; i < x_bdf_112.size(); i++)
    for (int j = 0; j < x_bdf_112[i].rows(); j++)
      for (int k = 0; k < x_bdf_112[i].cols(); k++)
        EXPECT_NEAR(amounts(j, k), x_bdf_112[i](j, k).val(),
                    std::max(amounts(j, k), x_bdf_112[i](j, k).val()) * rel_err);


  vector<Matrix<var, Dynamic, Dynamic> > x_bdf_121(7);
  x_bdf_121[0] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar, tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_121[1] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar_v, tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_121[2] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar, tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_121[3] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar_v, tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_121[4] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar_v, tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_121[5] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar_v, tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_121[5] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar, tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);

  for (size_t i = 0; i < x_bdf_121.size(); i++)
    for (int j = 0; j < x_bdf_121[i].rows(); j++)
      for (int k = 0; k < x_bdf_121[i].cols(); k++)
        EXPECT_NEAR(amounts(j, k), x_bdf_121[i](j, k).val(),
                    std::max(amounts(j, k), x_bdf_121[i](j, k).val()) * rel_err);


  vector<Matrix<var, Dynamic, Dynamic> > x_bdf_111(7);
  x_bdf_111[0] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar[0], tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_111[1] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar_v[0], tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_111[2] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar[0], tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_111[3] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v[0], biovar_v[0], tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_111[4] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar_v[0], tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_111[5] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar_v[0], tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_111[5] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix[0], biovar[0], tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);

  for (size_t i = 0; i < x_bdf_111.size(); i++)
    for (int j = 0; j < x_bdf_111[i].rows(); j++)
      for (int k = 0; k < x_bdf_111[i].cols(); k++)
        EXPECT_NEAR(amounts(j, k), x_bdf_111[i](j, k).val(),
                    std::max(amounts(j, k), x_bdf_111[i](j, k).val()) * rel_err);


  vector<Matrix<var, Dynamic, Dynamic> > x_bdf_211(7);
  x_bdf_211[0] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v, biovar[0], tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_211[1] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v, biovar_v[0], tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_211[2] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v, biovar[0], tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_211[3] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v, biovar_v[0], tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_211[4] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar_v[0], tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_211[5] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar_v[0], tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_211[5] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar[0], tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);

  for (size_t i = 0; i < x_bdf_211.size(); i++)
    for (int j = 0; j < x_bdf_211[i].rows(); j++)
      for (int k = 0; k < x_bdf_211[i].cols(); k++)
        EXPECT_NEAR(amounts(j, k), x_bdf_211[i](j, k).val(),
                    std::max(amounts(j, k), x_bdf_211[i](j, k).val()) * rel_err);


  vector<Matrix<var, Dynamic, Dynamic> > x_bdf_221(7);
  x_bdf_221[0] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v, biovar, tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_221[1] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v, biovar_v, tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_221[2] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v, biovar, tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_221[3] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix_v, biovar_v, tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_221[4] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar_v, tlag[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_221[5] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar_v, tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);
  x_bdf_221[5] = torsten::generalOdeModel_bdf(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar, tlag_v[0],
                                  0,
                                  rel_tol, abs_tol, max_num_steps);

  for (size_t i = 0; i < x_bdf_221.size(); i++)
    for (int j = 0; j < x_bdf_221[i].rows(); j++)
      for (int k = 0; k < x_bdf_221[i].cols(); k++)
        EXPECT_NEAR(amounts(j, k), x_bdf_221[i](j, k).val(),
                    std::max(amounts(j, k), x_bdf_221[i](j, k).val()) * rel_err);

  // CHECK - do I need an AD test for every function signature ?
}

template <typename T0, typename T1, typename T2, typename T3>
inline
std::vector<typename boost::math::tools::promote_args<T0, T1, T2, T3>::type>
oneCptModelODE_abstime(const T0& t,
                       const std::vector<T1>& x,
	                   const std::vector<T2>& parms,
	                   const std::vector<T3>& rate,
	                   const std::vector<int>& dummy, std::ostream* pstream__) {
  typedef typename boost::math::tools::promote_args<T0, T1, T2, T3>::type scalar;

  scalar CL0 = parms[0], V1 = parms[1], ka = parms[2], CLSS = parms[3],
    K = parms[4];

  scalar CL = CL0 + (CLSS - CL0) * (1 - stan::math::exp(-K * t));
  scalar k10 = CL / V1;

  std::vector<scalar> y(2, 0);

  y[0] = -ka * x[0];
  y[1] = ka * x[0] - k10 * x[1];

  return y;
}

struct oneCptModelODE_abstime_functor {
  template <typename T0, typename T1, typename T2, typename T3>
  inline
  std::vector<typename boost::math::tools::promote_args<T0, T1, T2, T3>::type>
  operator()(const T0& t,
             const std::vector<T1>& x,
             const std::vector<T2>& parms,
             const std::vector<T3>& rate,
             const std::vector<int>& dummy, std::ostream* pstream__) const {
        return oneCptModelODE_abstime(t, x, parms, rate, dummy, pstream__);
    }
};

TEST(Torsten, genCpt_One_abstime_SingleDose) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::Dynamic;

  double rel_err = 1e-6;

  vector<vector<double> > pMatrix(1);
  pMatrix[0].resize(5);
  pMatrix[0][0] = 10; // CL0
  pMatrix[0][1] = 80; // Vc
  pMatrix[0][2] = 1.2; // ka
  pMatrix[0][3] = 2; // CLSS
  pMatrix[0][4] = 1; // K

  int nCmt = 2;
  vector<vector<double> > biovar(1);
  biovar[0].resize(nCmt);
  biovar[0][0] = 1;  // F1
  biovar[0][1] = 1;  // F2

  vector<vector<double> > tlag(1);
  tlag[0].resize(nCmt);
  tlag[0][0] = 0;  // tlag1
  tlag[0][1] = 0;  // tlag2


  vector<double> time(10);
  time[0] = 0.0;
  for(int i = 1; i < 9; i++) time[i] = time[i - 1] + 0.25;
  time[9] = 4.0;

  vector<double> amt(10, 0);
  amt[0] = 1000;

  vector<double> rate(10, 0);

  vector<int> cmt(10, 2);
  cmt[0] = 1;

  vector<int> evid(10, 0);
  evid[0] = 1;

  vector<double> ii(10, 0);
  ii[0] = 12;

  vector<int> addl(10, 0);
  addl[0] = 14;

  vector<int> ss(10, 0);

  double rel_tol = 1e-8, abs_tol = 1e-8;
  long int max_num_steps = 1e8;

  Matrix<double, Eigen::Dynamic, Eigen::Dynamic> x_rk45;
  x_rk45 = torsten::generalOdeModel_rk45(oneCptModelODE_abstime_functor(), 2,
                                time, amt, rate, ii, evid, cmt, addl, ss,
                                pMatrix, biovar, tlag,
                                0,
                                rel_tol, abs_tol, max_num_steps);

  Matrix<double, Eigen::Dynamic, Eigen::Dynamic> x_bdf;
  x_bdf = torsten::generalOdeModel_bdf(oneCptModelODE_abstime_functor(), 2,
                              time, amt, rate, ii, evid, cmt, addl, ss,
                              pMatrix, biovar, tlag,
                              0,
                              rel_tol, abs_tol, max_num_steps);

  MatrixXd amounts(10, 2);
  amounts << 1000.0, 0.0,
	  	     740.8182, 255.4765,
			 548.8116, 439.2755,
			 406.5697, 571.5435,
			 301.1942, 666.5584,
			 223.1302, 734.5274,
			 165.2989, 782.7979,
			 122.4564, 816.6868,
			 90.71795, 840.0581,
			 8.229747, 869.0283;

  expect_near_matrix_eq(amounts, x_rk45, rel_err);
  expect_near_matrix_eq(amounts, x_bdf, rel_err);

  // Test AutoDiff against FiniteDiff
   double diff = 1e-8, diff2 = .25; // CHECK - diff2 seems pretty high!!
   test_generalOdeModel2(oneCptModelODE_abstime_functor(), nCmt,
                        time, amt, rate, ii, evid, cmt, addl, ss,
                        pMatrix, biovar, tlag,
                        rel_tol, abs_tol, max_num_steps, diff, diff2, "rk45");
   test_generalOdeModel2(oneCptModelODE_abstime_functor(), nCmt,
                        time, amt, rate, ii, evid, cmt, addl, ss,
                        pMatrix, biovar, tlag,
                        rel_tol, abs_tol, max_num_steps, diff, diff2, "bdf");
}

TEST_F(TorstenOneCptTest, multiple_dose_time_dependent_param) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::Dynamic;

  pMatrix.resize(nt);
  for (int i = 0; i < nt; i++) {
    pMatrix[i].resize(5);
    if (i < 6) pMatrix[i][0] = 10; // CL
    else pMatrix[i][0] = 50;
    pMatrix[i][1] = 80; // Vc
    pMatrix[i][2] = 1.2; // ka
  }

  time[0] = 0.0;
  for(int i = 1; i < nt; i++) time[i] = time[i - 1] + 2.5;
  addl[0] = 1;

  double rel_tol = 1e-8, abs_tol = 1e-8;
  long int max_num_steps = 1e8;
  MatrixXd x_rk45 = torsten::generalOdeModel_rk45(f, nCmt,
                                         time, amt, rate, ii, evid, cmt, addl, ss,
                                         pMatrix, biovar, tlag,
                                         0,
                                         rel_tol, abs_tol, max_num_steps);

  MatrixXd x_bdf = torsten::generalOdeModel_bdf(f, nCmt,
                                       time, amt, rate, ii, evid, cmt, addl, ss,
                                       pMatrix, biovar, tlag,
                                       0,
                                       rel_tol, abs_tol, max_num_steps);

  MatrixXd x(nt, 2);
  x << 1000.0, 0.0,
    4.978707e+01, 761.1109513,
    2.478752e+00, 594.7341503,
    1.234098e-01, 437.0034049,
    6.144212e-03, 319.8124495,
    5.488119e+02, 670.0046601,
    2.732374e+01, 323.4948561,
    1.360369e+00, 76.9219400,
    6.772877e-02, 16.5774607,
    3.372017e-03, 3.4974152;

  torsten::test::test_val(x, x_rk45, 1e-6, 1e-5);
  torsten::test::test_val(x, x_bdf, 1e-4, 1e-5);

  // Test AutoDiff against FiniteDiff
  double diff = 1e-8, diff2 = 2e-2;
  test_generalOdeModel2(f, nCmt,
                        time, amt, rate, ii, evid, cmt, addl, ss,
                        pMatrix, biovar, tlag,
                        rel_tol, abs_tol, max_num_steps, diff, diff2, "rk45");
  test_generalOdeModel2(f, nCmt,
                        time, amt, rate, ii, evid, cmt, addl, ss,
                        pMatrix, biovar, tlag,
                        rel_tol, abs_tol, max_num_steps, diff, diff2, "bdf");
}

TEST_F(TorstenOneCptTest, rate_var) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::Dynamic;

  amt[0] = 1200;
  rate[0] = 1200;

  double rel_tol = 1e-6, abs_tol = 1e-6;
  long int max_num_steps = 1e6;

  MatrixXd x_rk45 = torsten::generalOdeModel_rk45(f, nCmt,
                                  time, amt, rate, ii, evid, cmt, addl, ss,
                                  pMatrix, biovar, tlag,
                                  0,
                                  rel_tol, abs_tol, max_num_steps);

  MatrixXd x_bdf = torsten::generalOdeModel_bdf(f, nCmt,
                                time, amt, rate, ii, evid, cmt, addl, ss,
                                pMatrix, biovar, tlag,
                                0,
                                rel_tol, abs_tol, max_num_steps);

  MatrixXd x(nt, 2);
  x << 0.00000,   0.00000,
             259.18178,  40.38605,
             451.18836, 145.61440,
             593.43034, 296.56207,
             698.80579, 479.13371,
             517.68806, 642.57025,
             383.51275, 754.79790,
             284.11323, 829.36134,
             210.47626, 876.28631,
             19.09398, 844.11769;

  torsten::test::test_val(x, x_rk45, 1e-6, 1e-5);
  torsten::test::test_val(x, x_bdf, 1e-5, 1e-5);

  // Test Autodiff
  double diff = 1e-8, diff2 = 2e-2;
  test_generalOdeModel2(f, nCmt,
                       time, amt, rate, ii, evid, cmt, addl, ss,
                       pMatrix, biovar, tlag,
                       rel_tol, abs_tol, max_num_steps, diff, diff2, "rk45");
  test_generalOdeModel2(f, nCmt,
                       time, amt, rate, ii, evid, cmt, addl, ss,
                       pMatrix, biovar, tlag,
                       rel_tol, abs_tol, max_num_steps, diff, diff2, "bdf");
}

TEST_F(TorstenTwoCptTest, rate_par) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::Dynamic;

  pMatrix[0][0] = 5;  // CL
  pMatrix[0][1] = 8;  // Q
  pMatrix[0][2] = 35;  // Vc
  pMatrix[0][3] = 105;  // Vp
  pMatrix[0][4] = 1.2;  // ka

  amt[0] = 1200;
  rate[0] = 1200;

  double rel_tol = 1e-6, abs_tol = 1e-6;
  long int max_num_steps = 1e6;

  MatrixXd x_rk45, x_bdf;
  x_rk45 = torsten::generalOdeModel_rk45(f2, nCmt,
                                         time, amt, rate, ii, evid, cmt, addl, ss,
                                         pMatrix, biovar, tlag,
                                         0,
                                         rel_tol, abs_tol, max_num_steps);
  x_bdf = torsten::generalOdeModel_bdf(f2, nCmt,
                                       time, amt, rate, ii, evid, cmt, addl, ss,
                                       pMatrix, biovar, tlag,
                                       0,
                                       rel_tol, abs_tol, max_num_steps);

  MatrixXd amounts(10, 3);
  amounts << 0.00000,   0.00000,   0.0000000,
    259.18178,  39.55748,   0.7743944,
    451.18836, 139.65573,   5.6130073,
    593.43034, 278.43884,  17.2109885,
    698.80579, 440.32663,  37.1629388,
    517.68806, 574.76950,  65.5141658,
    383.51275, 653.13596,  99.2568509,
    284.11323, 692.06145, 135.6122367,
    210.47626, 703.65965, 172.6607082,
    19.09398, 486.11014, 406.6342765;

  // relative error determined empirically
  double rel_err_rk45 = 1e-6, rel_err_bdf = 1e-4;
  expect_near_matrix_eq(amounts, x_rk45, rel_err_rk45);
  expect_near_matrix_eq(amounts, x_bdf, rel_err_bdf);

  // Test Autodiff
  double diff = 1e-8, diff2 = 2e-2;
  test_generalOdeModel2(f2, nCmt,
                        time, amt, rate, ii, evid, cmt, addl, ss,
                        pMatrix, biovar, tlag,
                        rel_tol, abs_tol, max_num_steps, diff, diff2, "rk45");
  test_generalOdeModel2(f2, nCmt,
                        time, amt, rate, ii, evid, cmt, addl, ss,
                        pMatrix, biovar, tlag,
                        rel_tol, abs_tol, max_num_steps, diff, diff2, "bdf");

  // {
  //   auto f1 = [&] (std::vector<double>& r) {
  //     return torsten::generalOdeModel_rk45(f, nCmt, time, amt, r, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag,
  //                                          0, rel_tol, abs_tol, max_num_steps);
  //   };

  //   auto f2 = [&] (std::vector<stan::math::var>& r) {
  //     return torsten::generalOdeModel_rk45(f, nCmt, time, amt, r, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag,
  //                                          0, rel_tol, abs_tol, max_num_steps);
  //   };
    
  //   torsten::test::test_grad(f1, f2, rate, 2e-5, 1e-6, 1e-3, 1e-3);
  // }
}

TEST_F(FribergKarlssonTest, steady_state) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::Dynamic;

  ss[0] = 1;

  double rel_tol = 1e-6, abs_tol = 1e-6;
  long int max_num_steps = 1e6;

  MatrixXd x_rk45, x_bdf;
  x_rk45 = torsten::generalOdeModel_rk45(f, nCmt,
                                time, amt, rate, ii, evid, cmt, addl, ss,
                                theta, biovar, tlag,
                                0,
                                rel_tol, abs_tol, max_num_steps);

  x_bdf = torsten::generalOdeModel_bdf(f, nCmt,
                              time, amt, rate, ii, evid, cmt, addl, ss,
                              theta, biovar, tlag,
                              0,
                              rel_tol, abs_tol, max_num_steps);

  MatrixXd amounts(10, 8);
  amounts << 8.000000e+04, 11996.63, 55694.35, -3.636308, -3.653620,  -3.653933, -3.653748, -3.653622,
    6.566800e+03, 53123.67, 70649.28, -3.650990, -3.653172, -3.653910, -3.653755, -3.653627,
    5.390358e+02, 34202.00, 80161.15, -3.662446, -3.653349, -3.653883, -3.653761, -3.653632,
    4.424675e+01, 23849.69, 80884.40, -3.665321, -3.653782, -3.653870, -3.653765, -3.653637,
    3.631995e+00, 19166.83, 78031.24, -3.664114, -3.654219, -3.653876, -3.653769, -3.653642,
    2.981323e-01, 16799.55, 74020.00, -3.660988, -3.654550, -3.653896, -3.653774, -3.653647,
    2.447219e-02, 15333.26, 69764.65, -3.656791, -3.654722, -3.653926, -3.653779, -3.653653,
    2.008801e-03, 14233.96, 65591.05, -3.651854, -3.654708, -3.653957, -3.653786, -3.653658,
    1.648918e-04, 13303.26, 61607.92, -3.646317, -3.654488, -3.653983, -3.653793, -3.653663,
    1.353552e-05, 12466.56, 57845.10, -3.640244, -3.654050, -3.653995, -3.653801, -3.653668;

  // relative error determined empirically (12%)
  double rel_err_rk45 = 1.2e-2, rel_err_bdf = 1.2e-2;
  expect_near_matrix_eq(amounts, x_rk45, rel_err_rk45);
  expect_near_matrix_eq(amounts, x_bdf, rel_err_bdf);

  // Test Autodiff
  double diff = 1e-8, diff2 = 2e-2;
  test_generalOdeModel2(f, nCmt,
                       time, amt, rate, ii, evid, cmt, addl, ss,
                       theta, biovar, tlag,
                       rel_tol, abs_tol, max_num_steps, diff, diff2, "rk45");

  std::cout << "WARNING: GRADIENT TESTS FOR GENERAL_ODE_BDF FAILS."
            << " SEE ISSUE 45."
            << std::endl;
  
  // gradients do not get properly evaluated in the bdf case!!!  
  // test_generalOdeModel2(f, nCmt,
  //                      time, amt, rate, ii, evid, cmt, addl, ss,
  //                      theta_v, biovar_v, tlag_v,
  //                      rel_tol, abs_tol, max_num_steps, diff, diff2, "adams");
}

TEST_F(TorstenOdeTest, exception) {
  pMatrix[0][0] = 1.0E-10;
  pMatrix[0][1] = 1.0E-10;
  pMatrix[0][2] = 1.0E-20;
  pMatrix[0][3] = 1.0E+80;
  pMatrix[0][4] = 1.0E+70;

  auto& f = refactor::PKTwoCptModel<double, double, double, double>::f_;
  int ncmt = refactor::PKTwoCptModel<double, double, double, double>::Ncmt;

  EXPECT_THROW(torsten::generalOdeModel_bdf(f, ncmt, time, amt, rate, ii,
                                                   evid, cmt, addl, ss, pMatrix,
                                                   biovar, tlag), std::runtime_error);
}

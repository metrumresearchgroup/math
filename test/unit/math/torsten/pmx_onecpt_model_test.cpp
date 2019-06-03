#include <stan/math.hpp>
#include <stan/math/rev/core.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/torsten/pmx_onecpt_model_test_fixture.hpp>
#include <test/unit/math/torsten/test_util.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

using stan::math::var;
using stan::math::to_var;
using refactor::PMXOneCptModel;
using stan::math::integrate_ode_bdf;
using torsten::pmx_integrate_ode_bdf;
using refactor::PMXOneCptODE;
using refactor::PMXOdeFunctorRateAdaptor;

TEST_F(TorstenOneCptModelTest, rate_dbl) {
  rate[0] = 1200;
  rate[1] = 200;
  using model_t = PMXOneCptModel<double, double, double, double>;
  model_t model(t0, y0, rate, CL, V2, ka);
  std::vector<double> yvec(y0.data(), y0.data() + y0.size());
  PMXOdeFunctorRateAdaptor<PMXOneCptODE, double> f1(model.f());

  std::vector<double> y = f1(t0, yvec, model.par(), rate, x_i, msgs);
  EXPECT_FLOAT_EQ(y[0], rate[0]);
  EXPECT_FLOAT_EQ(y[1], rate[1]);
  EXPECT_FALSE(torsten::has_var_rate<model_t>::value);
}

TEST_F(TorstenOneCptModelTest, rate_var) {
  rate[0] = 1200;
  rate[1] = 200;
  var CLv = to_var(CL);
  var V2v = to_var(V2);
  var kav = to_var(ka);
  std::vector<stan::math::var> rate_var{to_var(rate)};
  using model_t = PMXOneCptModel<double, double, var, var>;
  model_t model(t0, y0, rate_var, CLv, V2v, kav);
  std::vector<double> yvec(y0.data(), y0.data() + y0.size());
  std::vector<stan::math::var> theta(model.par());
  PMXOdeFunctorRateAdaptor<PMXOneCptODE, var> f1(model.f(), theta.size());
  theta.insert(theta.end(), rate_var.begin(), rate_var.end());

  std::vector<var> y = f1(t0, yvec, theta, x_r, x_i, msgs);
  EXPECT_FLOAT_EQ(y[0].val(), rate[0]);
  EXPECT_FLOAT_EQ(y[1].val(), rate[1]);

  EXPECT_TRUE(torsten::has_var_rate<model_t>::value);
}

TEST_F(TorstenOneCptModelTest, rate_var_y0) {
  rate[0] = 1200;
  rate[1] = 200;
  y0[0] = 150;
  y0[1] = 50;
  var CLv = to_var(CL);
  var V2v = to_var(V2);
  var kav = to_var(ka);
  std::vector<stan::math::var> rate_var{to_var(rate)};
  using model_t = PMXOneCptModel<double, double, var, var>;
  model_t model(t0, y0, rate_var, CLv, V2v, kav);
  std::vector<stan::math::var> theta(model.par());
  std::vector<double> yvec(y0.data(), y0.data() + y0.size());
  PMXOdeFunctorRateAdaptor<PMXOneCptODE, var> f1(model.f(), theta.size());
  theta.insert(theta.end(), rate_var.begin(), rate_var.end());

  std::vector<var> y = f1(t0, yvec, theta, x_r, x_i, msgs);

  EXPECT_TRUE(torsten::has_var_rate<model_t>::value);
  EXPECT_FALSE(torsten::has_var_init<model_t>::value);
  EXPECT_TRUE(torsten::has_var_par<model_t>::value);
  EXPECT_TRUE((std::is_same<torsten::f_t<model_t>, refactor::PMXOneCptODE>::value));
}

TEST_F(TorstenOneCptModelTest, onecpt_solver) {
  rate[0] = 1200;
  rate[1] = 200;
  y0[0] = 150;
  y0[1] = 50;
  ts[0] = 10.0;
  ts.resize(1);
  var CLv = to_var(CL);
  var V2v = to_var(V2);
  var kav = to_var(ka);
  std::vector<stan::math::var> rate_var{to_var(rate)};
  using model_t = PMXOneCptModel<double, double, var, var>;
  model_t model(t0, y0, rate_var, CLv, V2v, kav);
  std::vector<stan::math::var> theta(model.par());
  std::vector<double> yvec(y0.data(), y0.data() + y0.size());
  PMXOdeFunctorRateAdaptor<PMXOneCptODE, var> f1(model.f(), theta.size());
  theta.insert(theta.end(), rate_var.begin(), rate_var.end());

  auto y1 = pmx_integrate_ode_bdf(f1, yvec, t0, ts, theta, x_r, x_i, msgs);
  auto y2 = model.solve(ts[0]);

  stan::math::vector_v y1_v = stan::math::to_vector(y1[0]);

  torsten::test::test_grad(theta, y1_v, y2, 1.e-6, 1.e-6);
  torsten::test::test_grad(rate_var, y1_v, y2, 1.e-6, 1.e-8);
}

TEST_F(TorstenOneCptModelTest, ss_bolus_finite_diff) {
  rate[0] = 0;
  rate[1] = 0;
  y0[0] = 150;
  y0[1] = 50;
  ts[0] = 10.0;
  ts.resize(1);
  std::vector<stan::math::var> rate_var{to_var(rate)};
  using model_t = PMXOneCptModel<double, double, double, double>;
  model_t model(t0, y0, rate, CL, V2, ka);

  double ii = 12.0;
  
  int cmt = 1;
  auto f1 = [&](std::vector<double>& amt_vec) {
    return model.solve(amt_vec[0], rate[0], ii, 1);
  };
  auto f2 = [&](std::vector<var>& amt_vec) {
    return model.solve(amt_vec[0], rate[0], ii, 1);
  };
  auto f3 = [&](std::vector<double>& amt_vec) {
    return model.solve(amt_vec[0], rate[1], ii, 2);
  };
  auto f4 = [&](std::vector<var>& amt_vec) {
    return model.solve(amt_vec[0], rate[1], ii, 2);
  };

  std::vector<double> amt_vec{1000.0};
  torsten::test::test_grad(f1, f2, amt_vec, 1.e-3, 1.e-16, 1.e-10, 1.e-12);
  torsten::test::test_grad(f3, f4, amt_vec, 1.e-3, 1.e-16, 1.e-10, 1.e-12);
}

TEST_F(TorstenOneCptModelTest, ss_infusion_finite_diff) {
  y0[0] = 150;
  y0[1] = 50;
  ts[0] = 10.0;
  ts.resize(1);
  std::vector<stan::math::var> rate_var{to_var(rate)};
  using model_t = PMXOneCptModel<double, double, double, double>;
  model_t model(t0, y0, rate, CL, V2, ka);

  double amt = 1800;
  double ii = 12.0;
  
  auto f1 = [&](std::vector<double>& rate_vec) {
    return model.solve(amt, rate_vec[0], ii, 1);
  };
  auto f2 = [&](std::vector<var>& rate_vec) {
    return model.solve(amt, rate_vec[0], ii, 1);
  };
  auto f3 = [&](std::vector<double>& rate_vec) {
    return model.solve(amt, rate_vec[0], ii, 2);
  };
  auto f4 = [&](std::vector<var>& rate_vec) {
    return model.solve(amt, rate_vec[0], ii, 2);
  };

  std::vector<double> rate_vec{500.0};
  torsten::test::test_grad(f1, f2, rate_vec, 1.e-3, 1.e-15, 1.e-10, 1.e-10);
  torsten::test::test_grad(f3, f4, rate_vec, 1.e-3, 1.e-15, 1.e-10, 1.e-10);
}

TEST_F(TorstenOneCptModelTest, ss_bolus_at_cpt_1_run_till_steady) {
  rate[0] = 0;
  rate[1] = 0;
  y0[0] = 150;
  y0[1] = 50;
  ts[0] = 10.0;
  ts.resize(1);
  using model_t = PMXOneCptModel<double, double, double, double>;
  model_t model(t0, y0, rate, CL, V2, ka);

  int cmt = 1;
  double ii = 12.0;
  
  auto f1 = [&](std::vector<double>& amt_vec) {
    double t = t0;
    Eigen::Matrix<double, -1, 1> y = y0;
    for (int i = 0; i < 100; ++i) {
      Eigen::Matrix<double, 1, -1> yt = y.transpose();
      model_t model_i(t, yt, rate, CL, V2, ka);
      double t_next = t + ii;
      Eigen::Matrix<double, -1, 1> ys = model_i.solve(t_next);
      ys(cmt - 1) += amt_vec[0];
      y = ys;
      t = t_next;
    }
    // steady state solution is the end of II dosing before
    // bolus is imposed, to check that we remove the
    // bolus(added in the last iteration) from the results
    y(cmt - 1) -= amt_vec[0];
    return y;
  };
  auto f2 = [&](std::vector<var>& amt_vec) {
    return model.solve(amt_vec[0], rate[cmt - 1], ii, cmt);
  };
  std::vector<double> amt_vec{1000.0};
  torsten::test::test_grad(f1, f2, amt_vec, 1.e-3, 1.e-12, 1.e-8, 1.e-10);
}

TEST_F(TorstenOneCptModelTest, ss_bolus_at_cpt_2_run_till_steady) {
  rate[0] = 0;
  rate[1] = 0;
  y0[0] = 150;
  y0[1] = 50;
  ts[0] = 10.0;
  ts.resize(1);
  using model_t = PMXOneCptModel<double, double, double, double>;
  model_t model(t0, y0, rate, CL, V2, ka);

  int cmt = 2;
  double ii = 12.0;
  
  auto f1 = [&](std::vector<double>& amt_vec) {
    double t = t0;
    Eigen::Matrix<double, -1, 1> y = y0;
    for (int i = 0; i < 100; ++i) {
      Eigen::Matrix<double, 1, -1> yt = y.transpose();
      model_t model_i(t, yt, rate, CL, V2, ka);
      double t_next = t + ii;
      Eigen::Matrix<double, -1, 1> ys = model_i.solve(t_next);
      ys(cmt - 1) += amt_vec[0];
      y = ys;
      t = t_next;
    }
    // steady state solution is the end of II dosing before
    // bolus is imposed, to check that we remove the
    // bolus(added in the last iteration) from the results
    y(cmt - 1) -= amt_vec[0];
    return y;
  };
  auto f2 = [&](std::vector<var>& amt_vec) {
    return model.solve(amt_vec[0], rate[cmt - 1], ii, cmt);
  };
  std::vector<double> amt_vec{1000.0};
  torsten::test::test_grad(f1, f2, amt_vec, 1.e-3, 1.e-12, 1.e-7, 1.e-10);
}

TEST_F(TorstenOneCptModelTest, ss_infusion_at_cpt_2_run_till_steady) {
  y0[0] = 150;
  y0[1] = 50;
  ts[0] = 10.0;
  ts.resize(1);
  using model_t = PMXOneCptModel<double, double, double, double>;
  model_t model(t0, y0, rate, CL, V2, ka);

  int cmt = 2;
  double ii = 6.0;
  double amt = 1000;
  
  auto f1 = [&](std::vector<double>& rate_vec) {
    double t = t0;
    Eigen::Matrix<double, -1, 1> y = y0;
    double t_infus = amt/rate_vec[cmt - 1];
    const std::vector<double> rate_zero{0.0, 0.0};
    for (int i = 0; i < 200; ++i) {
      Eigen::Matrix<double, 1, -1> yt;
      Eigen::Matrix<double, -1, 1> ys;
      yt = y.transpose();
      model_t model_i(t, yt, rate_vec, CL, V2, ka);
      double t_next = t + t_infus;
      ys = model_i.solve(t_next);
      yt = ys.transpose();
      t = t_next;
      model_t model_j(t, yt, rate_zero, CL, V2, ka);
      t_next = t + ii - t_infus;
      ys = model_j.solve(t_next);
      y = ys;
    }
    return y;
  };
  auto f2 = [&](std::vector<var>& rate_vec) {
    return model.solve(amt, rate_vec[cmt - 1], ii, cmt);
  };
  std::vector<double> rate_vec{0.0, 300.0};
  // FIXME: rate gradient is not correct
  // torsten::test::test_grad(f1, f2, rate_vec, 1.e-3, 1.e-12, 1.e-8, 1.e-10);
}

TEST_F(TorstenOneCptModelTest, ss_bolus) {
  rate[0] = 0;
  rate[1] = 0;
  y0[0] = 150;
  y0[1] = 50;
  ts[0] = 10.0;
  ts.resize(1);
  var CLv = to_var(CL);
  var V2v = to_var(V2);
  var kav = to_var(ka);
  std::vector<stan::math::var> rate_var{to_var(rate)};
  using model_t = PMXOneCptModel<double, double, var, var>;
  model_t model(t0, y0, rate_var, CLv, V2v, kav);
  std::vector<stan::math::var> theta(model.par());

  std::cout.precision(12);

  double amt = 1800;
  int cmt = 1;
  double ii = 12.0;
  
  auto y1 = model.solve(amt, rate_var[cmt - 1], ii, cmt);
  EXPECT_FLOAT_EQ(y1(0).val(), 1.00330322392E-3);
  EXPECT_FLOAT_EQ(y1(1).val(), 2.07672937446E+0);

  std::vector<double> g1, g2;
  stan::math::set_zero_all_adjoints();
  y1(0).grad(theta, g1);
  EXPECT_FLOAT_EQ(g1[0], 0.0);
  EXPECT_FLOAT_EQ(g1[1], 0.0);
  EXPECT_FLOAT_EQ(g1[2], -0.0120396453978);
  stan::math::set_zero_all_adjoints();
  y1(1).grad(theta, g1);
  EXPECT_FLOAT_EQ(g1[0], -0.266849753086);
  EXPECT_FLOAT_EQ(g1[1], 0.166781095679);
  EXPECT_FLOAT_EQ(g1[2], -1.8559692314);

  cmt = 2;
  y1 = model.solve(amt, rate_var[cmt - 1], ii, cmt);
  EXPECT_FLOAT_EQ(y1(0).val(), 0);
  EXPECT_FLOAT_EQ(y1(1).val(), 0.996102795153);

  stan::math::set_zero_all_adjoints();
  y1(0).grad(theta, g1);
  EXPECT_FLOAT_EQ(g1[0], 0.0);
  EXPECT_FLOAT_EQ(g1[1], 0.0);
  EXPECT_FLOAT_EQ(g1[2], 0.0);
  stan::math::set_zero_all_adjoints();
  y1(1).grad(theta, g1);
  EXPECT_FLOAT_EQ(g1[0], -0.149498104338);
  EXPECT_FLOAT_EQ(g1[1], 0.0934363152112);
  EXPECT_FLOAT_EQ(g1[2], 0.0);
}

TEST_F(TorstenOneCptModelTest, ss_multi_truncated_infusion) {
  rate[0] = 1100;
  rate[1] = 770;
  y0[0] = 150;
  y0[1] = 50;
  ts[0] = 10.0;
  ts.resize(1);
  var CLv = to_var(CL);
  var V2v = to_var(V2);
  var kav = to_var(ka);
  std::vector<stan::math::var> rate_var{to_var(rate)};
  using model_t = PMXOneCptModel<double, double, var, var>;
  model_t model(t0, y0, rate_var, CLv, V2v, kav);
  std::vector<var> theta{model.par()};

  double amt = 1800;
  int cmt = 1;
  double ii = 12.0;
  
  auto y1 = model.solve(amt, rate_var[cmt - 1], ii, cmt);
  EXPECT_FLOAT_EQ(y1(0).val(), 0.00312961339574);
  EXPECT_FLOAT_EQ(y1(1).val(), 3.61310672484);

  std::vector<double> g1, g2;
  stan::math::set_zero_all_adjoints();
  y1(0).grad(theta, g1);
  EXPECT_FLOAT_EQ(g1[0], 0.0);
  EXPECT_FLOAT_EQ(g1[1], 0.0);
  EXPECT_FLOAT_EQ(g1[2], -0.0342061212685);
  stan::math::set_zero_all_adjoints();
  y1(1).grad(theta, g1);
  EXPECT_FLOAT_EQ(g1[0], -0.421478621857);
  EXPECT_FLOAT_EQ(g1[1], 0.26342413866);
  EXPECT_FLOAT_EQ(g1[2], -3.20135491073);

  cmt = 2;
  y1 = model.solve(amt, rate_var[cmt - 1], ii, cmt);
  EXPECT_FLOAT_EQ(y1(0).val(), 0.0);
  EXPECT_FLOAT_EQ(y1(1).val(), 2.25697891686);

  stan::math::set_zero_all_adjoints();
  y1(0).grad(theta, g1);
  EXPECT_FLOAT_EQ(g1[0], 0.0);
  EXPECT_FLOAT_EQ(g1[1], 0.0);
  EXPECT_FLOAT_EQ(g1[2], 0.0);
  stan::math::set_zero_all_adjoints();
  y1(1).grad(theta, g1);
  EXPECT_FLOAT_EQ(g1[0], -0.298001025911);
  EXPECT_FLOAT_EQ(g1[1], 0.186250641194);
  EXPECT_FLOAT_EQ(g1[2], 0.0);
}

TEST_F(TorstenOneCptModelTest, ss_const_infusion) {
  rate[0] = 1100;
  rate[1] = 770;
  y0[0] = 150;
  y0[1] = 50;
  ts[0] = 10.0;
  ts.resize(1);
  var CLv = to_var(CL);
  var V2v = to_var(V2);
  var kav = to_var(ka);
  std::vector<var> theta{CLv, V2v, kav};
  std::vector<stan::math::var> rate_var{to_var(rate)};
  using model_t = PMXOneCptModel<double, double, var, var>;
  model_t model(t0, y0, rate_var, CLv, V2v, kav);

  std::cout.precision(12);

  double amt = 1800;
  int cmt = 1;
  double ii = 0.0;
  
  auto y1 = model.solve(amt, rate_var[cmt - 1], ii, cmt);
  EXPECT_FLOAT_EQ(y1(0).val(), 916.666666667);
  EXPECT_FLOAT_EQ(y1(1).val(), 1760);

  std::vector<double> g1, g2;
  stan::math::set_zero_all_adjoints();
  y1(0).grad(theta, g1);
  EXPECT_FLOAT_EQ(g1[0], 0.0);
  EXPECT_FLOAT_EQ(g1[1], 0.0);
  EXPECT_FLOAT_EQ(g1[2], -763.888888889);
  stan::math::set_zero_all_adjoints();
  y1(1).grad(theta, g1);
  EXPECT_FLOAT_EQ(g1[0], -35.2);
  EXPECT_FLOAT_EQ(g1[1], 22);
  EXPECT_FLOAT_EQ(g1[2], 0.0);

  cmt = 2;
  y1 = model.solve(amt, rate_var[cmt - 1], ii, cmt);
  EXPECT_FLOAT_EQ(y1(0).val(), 0.0);
  EXPECT_FLOAT_EQ(y1(1).val(), 1232);

  stan::math::set_zero_all_adjoints();
  y1(0).grad(theta, g1);
  EXPECT_FLOAT_EQ(g1[0], 0.0);
  EXPECT_FLOAT_EQ(g1[1], 0.0);
  EXPECT_FLOAT_EQ(g1[2], 0.0);
  stan::math::set_zero_all_adjoints();
  y1(1).grad(theta, g1);
  EXPECT_FLOAT_EQ(g1[0], -24.64);
  EXPECT_FLOAT_EQ(g1[1], 15.4);
  EXPECT_FLOAT_EQ(g1[2], 0.0);
}

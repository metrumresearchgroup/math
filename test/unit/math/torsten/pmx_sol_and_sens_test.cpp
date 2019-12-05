#include <stan/math/rev/arr.hpp>
#include <gtest/gtest.h>
#include <stan/math/torsten/pmx_sol_and_sens.hpp>
#include <stan/math/torsten/dsolve/pmx_integrate_ode_rk45.hpp>
#include <stan/math.hpp>
#include <test/unit/math/torsten/pmx_ode_test_fixture.hpp>
#include <test/unit/util.hpp>
#include <math.h>
#include <iostream>
#include <vector>

using torsten::pmx_integrate_ode_rk45;
using stan::math::var;
using std::vector;

template<typename F>
struct rk45_sol {
  const F& f;
  const std::vector<double>& y0;
  double t0;
  rk45_sol(const F& f0, const std::vector<double>& y00, double t00) :
    f(f0), y0(y00), t0(t00)
  {}

  inline std::vector<std::vector<double> >
  operator()(const std::vector<double>& theta,
             const std::vector<double>& x_r,
             const std::vector<int>& x_i, std::ostream* msgs) const {
    std::vector<double> y(pmx_integrate_ode_rk45(f, y0, t0, x_r, theta, x_r, x_i, msgs).back());
    std::vector<std::vector<double> > res(y.size());
    std::transform(y.begin(), y.end(), res.begin(),
                   [](double y_i) { return std::vector<double>{y_i}; });
    return res;
  }
};

template<typename F>
struct rk45_sen {
  const F& f;
  const std::vector<double>& y0;
  double t0;
  rk45_sen(const F& f0, const std::vector<double>& y00, double t00) :
    f(f0), y0(y00), t0(t00)
  {}

  inline std::vector<std::vector<double> >
  operator()(const std::vector<double>& theta,
             const std::vector<double>& x_r,
             const std::vector<int>& x_i, std::ostream* msgs) const {
    std::vector<stan::math::var> theta_var(stan::math::to_var(theta));
    std::vector<stan::math::var> y(pmx_integrate_ode_rk45(f, y0, t0, x_r, theta_var, x_r, x_i, msgs).back());
    std::vector<std::vector<double> > res(y.size());
    std::transform(y.begin(), y.end(), res.begin(),
                   [&theta_var](stan::math::var& y_i) {
                     std::vector<double> res_i;
                     res_i.push_back(y_i.val());
                     stan::math::set_zero_all_adjoints();
                     std::vector<double> g;
                     y_i.grad(theta_var, g);
                     res_i.insert(res_i.end(), g.begin(), g.end());
                     return res_i; });
    return res;
  }
};

TEST_F(TorstenOdeTest_chem, odeint_rk45_fwd_sensitivity_theta) {
  using stan::math::var;
  
  rk45_sol<chemical_kinetics> f0(f, y0, t0);
  rk45_sen<chemical_kinetics> f1(f, y0, t0);

  {
    std::vector<double> sol_ode = pmx_integrate_ode_rk45(f, y0, t0, ts, theta, x_r, x_i).back();
    std::vector<double> sol = torsten::pmx_sol_and_sens(f0, f1, theta, ts, x_i, nullptr);
    EXPECT_EQ(sol.size(), sol_ode.size());
    for (size_t i = 0; i < sol.size(); ++i) {
      EXPECT_FLOAT_EQ(sol[i], sol_ode[i]);    
    }
  }

  {
    std::vector<var> theta_var(stan::math::to_var(theta));
    std::vector<var> sol_ode = pmx_integrate_ode_rk45(f, y0, t0, ts, theta_var, x_r, x_i).back();
    std::vector<var> sol = torsten::pmx_sol_and_sens(f0, f1, theta_var, ts, x_i, nullptr);  
    EXPECT_EQ(sol.size(), sol_ode.size());
    std::vector<double> g, g1;
    for (size_t i = 0; i < sol.size(); ++i) {
      EXPECT_FLOAT_EQ(sol[i].val(), sol_ode[i].val());
      stan::math::set_zero_all_adjoints();
      sol_ode[i].grad(theta_var, g);
      stan::math::set_zero_all_adjoints();
      sol[i].grad(theta_var, g1);
      torsten::test::test_val(g, g1);
    }
  }
}

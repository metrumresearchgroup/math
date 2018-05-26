#ifndef STAN_MATH_TORSTEN_RATE_ADAPTOR_HPP
#define STAN_MATH_TORSTEN_RATE_ADAPTOR_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>
#include <vector>
#include <iostream>

namespace refactor {

  using boost::math::tools::promote_args;
  using refactor::PKODEModel;

  template<typename...>
  class PKODERateAdaptor;
  template<template<typename...> class T_model, typename... Ts_par>
  // template<typename T_time, typename T_init, typename T_rate, typename T_par, typename F, typename Ti>
  class PKODERateAdaptor<T_model<Ts_par...> > {
    using T_time   = typename T_model<Ts_par...>::time_type;
    using T_init   = typename T_model<Ts_par...>::init_type;
    using T_rate = typename T_model<Ts_par...>::rate_type;
    using T_par = typename T_model<Ts_par...>::par_type;
    using F = typename T_model<Ts_par...>::f_type;
    using par_type = typename promote_args<T_par, T_rate>::type;
    using FA = torsten::ode_rate_var_functor<torsten::general_functor<F> >;

    const std::vector<double> dummy_;
    const FA f_;
    const std::vector<par_type> theta_;
    const PKODEModel<T_time, T_init, double, par_type, FA, int> model_;
    
    std::vector<par_type>
    concat_par_rate(const std::vector<T_par> & par,
                    const std::vector<T_rate> & rate) {
      const size_t n = par.size();
      std::vector<par_type> theta(n + rate.size());
      for (size_t i = 0; i < n; i++) theta[i] = par[i];
      for (size_t i = 0; i < rate.size(); i++) theta[n + i] = rate[i];
      return theta;
    }

  public:

    PKODERateAdaptor(const T_model<Ts_par...> & pkmodel) :
      dummy_{0.0},
      f_(torsten::general_functor<F>(pkmodel.rhs_fun())),
      theta_(concat_par_rate(pkmodel.par(), pkmodel.rate())),
      model_(pkmodel.t0(),
             pkmodel.y0(),
             dummy_,
             theta_, f_,
             pkmodel.ncmt())
    {}

    const PKODEModel<T_time, T_init, double, par_type, FA, int>& model() {
      return model_;
    }
  };
      
  template<typename T_time, typename T_init, typename T_par, typename F>
  class PKODERateAdaptor<PKODEModel<T_time, T_init, double, T_par, F, int>> {
    using FA = torsten::ode_rate_dbl_functor<torsten::general_functor<F> >;
    const torsten::ode_rate_dbl_functor<torsten::general_functor<F> > f_;
    const PKODEModel<T_time, T_init, double, T_par, FA, int> model_;
  public:
    PKODERateAdaptor(const PKODEModel<T_time,
                     T_init, double, T_par, F, int> & pkmodel) :
      f_(torsten::general_functor<F>(pkmodel.rhs_fun())),
      model_(pkmodel.t0(),
             pkmodel.y0(),
             pkmodel.rate(),
             pkmodel.par(), f_,
             pkmodel.ncmt())
    {}

    const PKODEModel<T_time, T_init, double, T_par, FA, int>& model() {
      return model_;
    }
  };
}

#endif

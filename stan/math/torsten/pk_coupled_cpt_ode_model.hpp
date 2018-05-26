#ifndef STAN_MATH_TORSTEN_COUPLED_CPT_ODE_MODEL_HPP
#define STAN_MATH_TORSTEN_COUPLED_CPT_ODE_MODEL_HPP

#include <stan/math/torsten/pk_coupled_model.hpp>

namespace refactor {



  // wrapper: one cpt model coupled with ode model
  template<typename T_time, typename T_init, typename T_rate, typename T_par, typename F, typename Ti>
  struct OneCptODEModel {
    using model_type = PKCoupledModel2<PKOneCptModel<T_time, T_init, T_rate, T_par>,
                                       PKODEModel<T_time,
                                                  T_init,
                                                  T_rate,
                                                  T_par,
                                                  F,
                                                  Ti> >;
    using model_type1 = typename model_type::model_type1;
    using model_type2 = typename model_type::model_type2;
    using scalar_type = typename model_type::scalar_type;
    using time_type = T_time;
    using rate_type = T_rate;
    using par_type = T_par;
    using init_type = T_init;
    using f_type = F;
    const F& f_;
    const model_type model;

    OneCptODEModel(const T_time& t0,
                   const PKRecord<T_init>& y0,
                   const std::vector<T_rate>& rate,
                   const std::vector<T_par> & par,
                   const F& f,
                   const Ti& n2) :
      f_(f),
      model{t0, y0, rate, par,
        PKOneCptModel<T_time, T_init, T_rate, T_par>::Ncmt, 
        f_, n2}
    {}

    template<template<typename...> class T_mp, typename... Ts>
    OneCptODEModel(const T_time& t0,
                   const PKRecord<T_init>& y0,
                   const std::vector<T_rate>& rate,
                   const std::vector<T_par> & par,
                   const T_mp<Ts...> &parameter,
                   const F& f,
                   const Ti& n2) :
      f_(f),
      model{t0, y0, rate, par,
        PKOneCptModel<T_time, T_init, T_rate, T_par>::Ncmt, 
        f_, n2}
    {}
  };

  // wrapper: one cpt model coupled with ode model
  template<typename T_time, typename T_init, typename T_rate, typename T_par, typename F, typename Ti>
  struct TwoCptODEModel {
    using model_type = PKCoupledModel2<PKTwoCptModel<T_time, T_init, T_rate, T_par>,
                                       PKODEModel<T_time,
                                                  T_init,
                                                  T_rate,
                                                  T_par,
                                                  F,
                                                  Ti> >;
    using model_type1 = typename model_type::model_type1;
    using model_type2 = typename model_type::model_type2;
    using scalar_type = typename model_type::scalar_type;
    using rate_type = T_rate;
    using par_type = T_par;
    using init_type = T_init;
    using f_type = F;
    const F& f_;
    const model_type model;

    TwoCptODEModel(const T_time& t0,
                   const PKRecord<T_init>& y0,
                   const std::vector<T_rate>& rate,
                   const std::vector<T_par> & par,
                   const F& f,
                   const Ti& n2) :
      f_(f),
      model{t0, y0, rate, par,
        PKTwoCptModel<T_time, T_init, T_rate, T_par>::Ncmt, 
        f_, n2}
    {}

    template<template<typename...> class T_mp, typename... Ts>
    TwoCptODEModel(const T_time& t0,
                   const PKRecord<T_init>& y0,
                   const std::vector<T_rate>& rate,
                   const std::vector<T_par> & par,
                   const T_mp<Ts...> &parameter,
                   const F& f,
                   const Ti& n2) :
      f_(f),
      model{t0, y0, rate, par,
        PKTwoCptModel<T_time, T_init, T_rate, T_par>::Ncmt, 
        f_, n2}
    {}
  };
}
#endif

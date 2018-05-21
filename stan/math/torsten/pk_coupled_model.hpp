#ifndef PK_COUPLED_MODEL_HPP
#define PK_COUPLED_MODEL_HPP

#include <stan/math/torsten/pk_onecpt_model.hpp>
#include <stan/math/torsten/pk_onecpt_solver.hpp>
#include <stan/math/torsten/pk_twocpt_model.hpp>
#include <stan/math/torsten/pk_twocpt_solver.hpp>
#include <stan/math/torsten/pk_ode_model.hpp>
#include <stan/math/torsten/pk_ode_solver.hpp>

namespace refactor {

  using boost::math::tools::promote_args;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  template <typename...>
  class PKCoupledModel2;
  template <
            template <typename...> class T_model1,
            template <typename...> class T_model2,
            typename...Ts1,
            typename...Ts2>
  class PKCoupledModel2<T_model1<Ts1...>, T_model2<Ts2...> > {
    const refactor::PKRecord<typename T_model1<Ts1...>::init_type> y0_1_;
    const refactor::PKRecord<typename T_model2<Ts2...>::init_type> y0_2_;
    const std::vector<typename promote_args<typename T_model1<Ts1...>::aug_par_type,
                                            typename T_model2<Ts2...>::aug_par_type>::type> theta_;
    const T_model1<Ts1...> model1_;
    const T_model2<Ts2...> model2_;

    template<typename T, typename U, typename V>
    std::vector<typename promote_args<T, U, V>::type>
    augmented_parameter(const std::vector<T>& par,
                        const refactor::PKRecord<U>& other1,
                        const V& other2) const {
      std::vector<typename promote_args<T, U, V>::type> res{par};
      for (int i = 0; i < other1.size(); i++) res.push_back(other1(i));
      res.push_back(other2);
      return res;
    }


  public:
    using model_type1 = T_model1<Ts1...>;
    using model_type2 = T_model2<Ts2...>;
    using scalar_type = typename
      promote_args<typename model_type1::scalar_type,
                   typename model_type2::scalar_type>::type;
    // using time_type   = T_time;

    // constructor
    template<typename T1_time,
             typename T1_init,
             typename T1_rate,
             typename T1_par,
             typename T2_time,
             typename T2_init,
             typename T2_rate,
             typename T2_par,
             typename... T1_extra_par,
             typename... T2_extra_par>
    PKCoupledModel2(const T1_time& t0_1,
                    const refactor::PKRecord<T1_init>& y0_1,
                    const std::vector<T1_rate> &rate_1,
                    const std::vector<T1_par> & par_1,
                    const T2_time& t0_2,
                    const refactor::PKRecord<T2_init>& y0_2,
                    const std::vector<T2_rate> &rate_2,
                    const std::vector<T2_par> & par_2,
                    T1_extra_par... pars1,
                    T2_extra_par... pars2) :
      model1_({t0_1, y0_1, rate_1, par_1, pars1...}),
      model2_({t0_2, y0_2, rate_2, par_2, pars2...})
    {}

    // constructor
    template<typename T1_time,
             typename T1_init,
             typename T1_rate,
             typename T1_par,
             typename... T1_extra_par,
             typename... T2_extra_par>
    PKCoupledModel2(const T1_time& t0,
                    const refactor::PKRecord<T1_init>& y0,
                    const std::vector<T1_rate>& rate,
                    const std::vector<T1_par> & par,
                    const int& n1, 
                    T1_extra_par... pars1,
                    T2_extra_par... pars2) : 
      y0_1_{ y0.head(n1) },
      y0_2_{ y0.segment(n1, y0.size()-n1) },
      model1_({t0, y0_1_, rate, par, pars1...}),
      model2_({t0, y0_2_, rate, par, pars2...})
    {}

   // get
    const T_model1<Ts1...>& model1() const { 
      return model1_;
    }
    const T_model2<Ts2...>& model2() const { 
      return model2_;
    }
  };

  // wrapper: one cpt model coupled with ode model
  template<typename T_time, typename T_init, typename T_rate, typename T_par, typename F, typename Ti>
  struct OneCptODEmodel {
    using model_type = PKCoupledModel2<PKOneCptModel<T_time, T_init, T_rate, T_par>,
                                       PKODEModel<T_time, T_init, T_rate, T_par, F, Ti> >;
    using scalar_type = typename model_type::scalar_type;
    using rate_type = T_rate;
    const model_type model;

    OneCptODEmodel(const T_time& t0,
                   const PKRecord<T_init>& y0,
                   const std::vector<T_rate>& rate,
                   const std::vector<T_par> & par,
                   const F& f,
                   const Ti& n2) :
      model{t0, y0, rate, par,
        PKOneCptModel<T_time, T_init, T_rate, T_par>::Ncmt, 
        f, n2}
    {}

    template<template<typename...> class T_mp, typename... Ts>
    OneCptODEmodel(const T_time& t0,
                   const PKRecord<T_init>& y0,
                   const std::vector<T_rate>& rate,
                   const std::vector<T_par> & par,
                   const T_mp<Ts...> &parameter,
                   const F& f,
                   const Ti& n2) :
      model{t0, y0, rate, par,
        PKOneCptModel<T_time, T_init, T_rate, T_par>::Ncmt, 
        f, n2}
    {}
  };

}




#endif

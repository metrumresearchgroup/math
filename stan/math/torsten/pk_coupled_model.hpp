#ifndef STAN_MATH_TORSTEN_COUPLED_MODEL_HPP
#define STAN_MATH_TORSTEN_COUPLED_MODEL_HPP

#include <stan/math/torsten/pk_onecpt_model.hpp>
#include <stan/math/torsten/pk_twocpt_model.hpp>
#include <stan/math/torsten/pk_ode_model.hpp>

namespace refactor {

  using boost::math::tools::promote_args;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  /**
   * Wrapper of the coupled model for the OneCpt-ODE models
   */
  template <typename...>
  class PKCoupledModel2;

  /**
   * Partial specialization.
   *
   * @tparam T_model1 type of 1st model
   * @tparam T_model2 type of 2nd model
   * @tparam Ts1 parameters of 1st model
   * @tparam Ts2 parameters of 1st model
   */
  template <template <typename...> class T_model1,
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

    /**
     * Coupled model constructor
     *
     * @tparam T1_time model1 time type
     * @tparam T1_init model1 init condition type
     * @tparam T1_rate model1 dosing rate type
     * @tparam T1_par  model1 parameter type
     * @tparam T2_time model2 time type
     * @tparam T2_init model2 init condition type
     * @tparam T2_rate model2 dosing rate type
     * @tparam T2_par  model2 parameter type
     * @tparam T1_extra_par model1 extra parameter type
     * @tparam T2_extra_par model2 extra parameter type
     * @param t0_1   model1 time
     * @param y0_1   model1 init condition
     * @param rate_1 model1 dosing rate
     * @param par_1  model1 parameter
     * @param t0_2   model2 time
     * @param y0_2   model2 init condition
     * @param rate_2 model2 dosing rate
     * @param par_2  model2 parameter
     * @param pars1 model1 extra parameters
     * @param pars2 model2 extra parameters
     */
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

    /**
     * Coupled model constructor
     *
     * @tparam T1_time model1 & model2 time type
     * @tparam T1_init model1 & model2 init condition type
     * @tparam T1_rate model1 & model2 dosing rate type
     * @tparam T1_par  model1 & model2 parameter type
     * @tparam T1_extra_par model1 & model2 extra parameter type
     * @param t0   model1 & 2 time
     * @param y0   model1 & 2 init condition
     * @param rate model1 & 2 dosing rate
     * @param par  model1 & 2 parameter
     * @param n1   model1 initial condition size
     * @param pars1 model1 extra parameters
     * @param pars2 model2 extra parameters
     */
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

    /**
     * get const model1 reference
     *
     * @return const reference to model1
     */
    const T_model1<Ts1...>& model1() const { 
      return model1_;
    }
    /**
     * get const model2 reference
     *
     * @return const reference to model2
     */
    const T_model2<Ts2...>& model2() const { 
      return model2_;
    }
  };



}




#endif

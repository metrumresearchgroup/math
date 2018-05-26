#ifndef STAN_MATH_TORSTEN_AUG_ODE_SOLVER_HPP
#define STAN_MATH_TORSTEN_AUG_ODE_SOLVER_HPP

#include <stan/math/torsten/pk_coupled_cpt_ode_model.hpp>
#include <stan/math/torsten/pk_cptode_adaptor.hpp>

namespace refactor {

  using boost::math::tools::promote_args;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::to_array_1d;
  using std::vector;
  using boost::math::tools::promote_args;

  template<typename...>
  class PKAdaptODESolver;

  /**
   * The ODE solver that applies an adaptor to the
   * user-supplied ODE to fit the Stan ODE solvers.
   *
   * @tparam T_adaptor type of adaptor for the base ode system.
   * @tparam T_model type of ODE-based models to be solved
   * @tparam Ts model parameters
   * @tparam Ts_adapt_extra extra adaptor parameters
   */
  template<template<typename...> class T_adaptor,
           template<typename...> class T_model,
           typename... Ts,
           typename... Ts_adapt_extra>
  class PKAdaptODESolver<T_adaptor<T_model<Ts...>, Ts_adapt_extra...> >{
    const torsten::integrator_structure& integrator_;
  public:
    /**
     * Constructor
     * @param[in] integrator integrator that contains parameters
     */
    PKAdaptODESolver(const torsten::integrator_structure& integrator) :
      integrator_(integrator)
    {}

    /**
     * Solve function solves a model for time span dt, by
     * first applying the adaptor to the model, so that the
     * actual solved model is the adapted one.
     * 
     * @tparam T_time delta t type
     * @param[in] pkmodel ODE-based model to be solved.
     * @param[in] dt time span of the solution
     * @return a column vector of the same size of the
     * adapted ODE.
     */
    template<typename T_time>
    Eigen::Matrix<typename T_model<Ts...>::scalar_type, Eigen::Dynamic, 1>
    solve(const T_model<Ts...> &pkmodel, const T_time& dt) const {
      using stan::math::to_array_1d;
      using std::vector;
      using boost::math::tools::promote_args;

      using scalar = typename T_model<Ts...>::scalar_type;

      T_adaptor<T_model<Ts...>, Ts_adapt_extra... > adapt(pkmodel);
      auto model = adapt.model();
      auto init = model.y0()   ;
      auto rate = model.rate();
      auto f = model.rhs_fun() ;
      auto pars = model.par();

      // assert((size_t) init.cols() == rate.size());

      T_time InitTime = model.t0();
      T_time EventTime = InitTime + dt;

      // Convert time parameters to fixed data for ODE integrator
      // FIX ME - see issue #30
      vector<double> ts(1, torsten::unpromote(EventTime));
      double t0 = torsten::unpromote(InitTime);

      Eigen::Matrix<scalar, Eigen::Dynamic, 1> pred;
      if (ts[0] == t0) {
        pred = init;
      } else {
        vector<scalar> init_vector = to_array_1d(init);
        vector<int> idummy;
        vector<vector<scalar> > pred_V = integrator_(f,
                                                     init_vector,
                                                     t0,
                                                     ts,
                                                     pars,
                                                     rate,
                                                     idummy);
        pred = stan::math::to_vector(pred_V[0]);
      }
      return pred;
    }
  };
}
#endif

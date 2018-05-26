#ifndef STAN_MATH_TORSTEN_CPTODE_ADAPTOR_HPP
#define STAN_MATH_TORSTEN_CPTODE_ADAPTOR_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>
#include <vector>
#include <iostream>
#include <stan/math/torsten/pk_coupled_cpt_ode_model.hpp>

namespace refactor {

  using boost::math::tools::promote_args;
  using refactor::PKODEModel;

  template <typename F0, typename T_pksolver>
  struct CptODEFunctor {
    F0 f0_;

    CptODEFunctor() { }

    explicit CptODEFunctor(const F0& f0) : f0_(f0) { }

    /**
     *  Returns the derivative of the base ODE system. The base 1 PK
     *  component is calculated analytically.
     *
     *  theta stores the ODE parameters, followed by the two initial
     *  PK states.
     *  x_r stores the rates for the FULL system, followed by the initial
     *  time.
     *
     *  Case 1: rate is fixed data and is passed through x_r.
     */
    template <typename T0, typename T1, typename T2, typename T3>
    inline
    std::vector<typename boost::math::tools::promote_args<T0, T1, T2, T3>::type>
    rate_dbl(const T0& t,
             const std::vector<T1>& y,
             const std::vector<T2>& theta,
             const std::vector<T3>& x_r,
             const std::vector<int>& x_i,
             std::ostream* pstream_) const {
      using stan::math::to_array_1d;
      using stan::math::to_vector;
      typedef typename boost::math::tools::promote_args<T0, T1, T2, T3>::type
        scalar;
      typedef typename boost::math::tools::promote_args<T0, T2, T3>::type
        T_pk;  // return object of fOneCpt  doesn't depend on T1

      using T_pkmodel = typename T_pksolver::template default_model<T0, T2, double, T2>;
      int nPK = T_pkmodel::Ncmt;
      int n_pk_par = T_pkmodel::Npar;

      // Get PK parameters
      std::vector<T2> thetaPK(theta.data(), theta.data() + n_pk_par);

      // Get initial PK states

      size_t nTheta = theta.size();
      // The last two components of theta should contain the initial PK states

      // T0 dt = t - stan::math::value_of(theta.back());

      // The last element of x_r contains the absolutime
      T0 dt = t - x_r[x_r.size() - 1];
      // T0 dt = t - stan::math::value_of(theta.back());

      T0 t0 = x_r[x_r.size() - 1];
      refactor::PKRecord<T2> y0_pk(nPK);
      for (int i = 0; i < nPK; ++i) {
        y0_pk(i) = theta[nTheta - nPK + i];
      }

      T_pkmodel pkmodel(t0,
                        y0_pk,
                        x_r,
                        thetaPK);
      auto y_pk0 = T_pksolver().solve(pkmodel, dt);
      std::vector<T_pk> y_pk(y_pk0.data(), y_pk0.data() + y_pk0.size());
      std::vector<scalar> dydt = f0_(dt, y, y_pk, theta, x_r, x_i, pstream_);

      for (size_t i = 0; i < dydt.size(); i++)
        dydt[i] += x_r[nPK + i];

      return dydt;
    }

    /**
     *  Case 2: rate is a parameter, stored in theta.
     *  Theta contains in this order: ODE parameters, rates, and
     *  initial base PK states.
     */
    template <typename T0, typename T1, typename T2, typename T3>
    inline
    std::vector<typename boost::math::tools::promote_args<T0, T1, T2, T3>::type>
    rate_var(const T0& t,
             const std::vector<T1>& y,
             const std::vector<T2>& theta,
             const std::vector<T3>& x_r,
             const std::vector<int>& x_i,
             std::ostream* pstream_) const {
      using std::vector;
      using stan::math::to_array_1d;
      using stan::math::to_vector;
      typedef typename boost::math::tools::promote_args<T0, T1, T2, T3>::type
        scalar;
      typedef typename boost::math::tools::promote_args<T0, T2, T3>::type
        T_pk;  // return object of fTwoCpt  doesn't depend on T1

      size_t nTheta = theta.size();
      using T_pkmodel = typename T_pksolver::template default_model<T0, T2, T2, T2>;
      int nPK = T_pkmodel::Ncmt;
      size_t nPD = y.size();
      size_t nODEparms = nTheta - 2 * nPK - nPD;  // number of ODE parameters
      nODEparms -= 1;

      // Theta first contains the base PK parameters, followed by
      // the other ODE parameters.
      int n_pk_par = T_pkmodel::Npar;

      std::vector<T2> thetaPK(theta.data(), theta.data() + n_pk_par);

      // Next theta contains the rates for the base PK compartments.
      vector<T2> ratePK(nPK);
      for (int i = 0; i < nPK; i++)
        ratePK[i] = theta[nODEparms + nPK + 1 + i];

      // followed by the rates in the other compartments.
      vector<T2> ratePD(nPD);
      for (size_t i = 0; i < nPD; i++)
        ratePD[i] = theta[nODEparms + nPK + 1 + nPK + i];

      // The last elements of theta contain the initial base PK states

      // Last element of x_r contains the initial time
      // T0 dt = t - x_r[x_r.size() - 1];
      T0 dt = t - stan::math::value_of(theta[nODEparms + nPK]);

      T0 t0 = stan::math::value_of(theta[nODEparms + nPK]);
      refactor::PKRecord<T2> y0_pk(nPK);
      for (int i = 0; i < nPK; ++i) {
        y0_pk(i) = theta[nODEparms + nPK - nPK + i];
      }
      T_pkmodel pkmodel(t0,
                                        y0_pk,
                                        ratePK,
                                        thetaPK);
      auto y_pk0 = T_pksolver().solve(pkmodel, dt);
      vector<T_pk> y_pk(y_pk0.data(), y_pk0.data() + y_pk0.size());
      vector<scalar> dydt = f0_(dt, y, y_pk, theta, x_r, x_i, pstream_);

      for (size_t i = 0; i < dydt.size(); i++)
        dydt[i] += ratePD[i];

      return dydt;
    }

    // Dummy operator to shut the compiler up
    // FIX ME - remove / find more elegant solution.
    template <typename T0, typename T1, typename T2, typename T3>
    inline
    std::vector<typename boost::math::tools::promote_args<T0, T1, T2, T3>::type>
    operator()(const T0& t,
               const std::vector<T1>& y,
               const std::vector<T2>& theta,
               const std::vector<T3>& x_r,
               const std::vector<int>& x_i,
               std::ostream* pstream_) const {
      std::cout << "CptODEFunctor: REPORT A BUG IF YOU SEE THIS." << std::endl;
      std::vector<T2> y_pk;
      return f0_(t, y, y_pk, theta, x_r, x_i, pstream_);
    }
  };

  template<typename...>
  class PKCptODEAdaptor;
  template<template<typename...> class T_model,
           typename T_time, typename T_init, typename T_par,
           typename F,  typename T_rate,
           typename T_pksolver>
  class PKCptODEAdaptor<T_model<T_time, T_init, T_rate, T_par, F, int>, 
                        T_pksolver> {
    using par_type = typename promote_args<T_par, T_rate, T_init>::type;
    using FA = torsten::ode_rate_var_functor<CptODEFunctor<F, T_pksolver> >;

    const std::vector<double> dummy_;
    const CptODEFunctor<F, T_pksolver> f0_;
    const FA f_;
    const std::vector<par_type> theta_;
    const PKODEModel<T_time, T_init, double, par_type, FA, int> model_;
    
    std::vector<par_type>
    concat_par_rate(const T_model<T_time, T_init, T_rate, T_par, F, int>& coupled_model) {
      auto par  = coupled_model.model.model2().par();
      auto y0_1 = coupled_model.model.model1().y0();
      auto t0 = stan::math::value_of(coupled_model.model.model1().t0());
      auto rate = coupled_model.model.model2().rate();
      const size_t n = par.size();
      const size_t m = y0_1.size();
      std::vector<par_type> theta(n);
      for (size_t i = 0; i < n; i++) theta[i] = par[i];
      for (size_t i = 0; i < m; i++) theta.push_back(y0_1(i));
      theta.push_back(t0);
      for (size_t i = 0; i < rate.size(); i++) theta.push_back(rate[i]);      
      return theta;
    }

  public:
    PKCptODEAdaptor(const T_model<T_time, T_init, T_rate, T_par, F, int> & coupled_model) :
      dummy_{0.0},
      f0_(coupled_model.model.model2().rhs_fun()),
      f_(f0_),
      theta_{ concat_par_rate(coupled_model) },
      model_{coupled_model.model.model2().t0(),
          coupled_model.model.model2().y0(),
          dummy_,
          theta_,
          f_,
          coupled_model.model.model2().ncmt()}
    {}

    const PKODEModel<T_time, T_init, double, par_type, FA, int>& model() {
      return model_;
    }
  };
      
  // partial
  template<template<typename...> class T_model,
           typename T_time, typename T_init, typename T_par,
           typename F,
           typename T_pksolver>
  class PKCptODEAdaptor<T_model<T_time, T_init, double, T_par, F, int>, 
                        T_pksolver> {
    using par_type = typename promote_args<T_par, T_init>::type;
    using FA = torsten::ode_rate_dbl_functor<
      CptODEFunctor<F, T_pksolver> >;

    const FA f_;
    const std::vector<par_type> theta_;
    const std::vector<double> new_rate_;
    const PKODEModel<T_time, T_init, double, par_type, FA, int> model_;

    std::vector<par_type>
    concat_par_init(const T_model<T_time, T_init, double, T_par, F, int>& coupled_model) {
      auto par  = coupled_model.model.model2().par();
      auto y0_1 = coupled_model.model.model1().y0();
      const size_t n = par.size();
      const size_t m = y0_1.size();
      std::vector<par_type> theta(n);
      for (size_t i = 0; i < n; i++) theta[i] = par[i];
      for (size_t i = 0; i < m; i++) theta.push_back(y0_1(i));
      return theta;
    }

    std::vector<double>
    concat_rate_t0(const T_model<T_time, T_init, double, T_par, F, int>& coupled_model) {
      auto t0 = stan::math::value_of(coupled_model.model.model1().t0());
      auto rate = coupled_model.model.model2().rate();
      std::vector<double> new_rate(rate);
      new_rate.push_back(t0);
      return new_rate;
    }
  public:
    PKCptODEAdaptor(const T_model<T_time,
                    T_init, double, T_par, F, int> & coupled_model) :
      f_(CptODEFunctor<F, T_pksolver>
         (coupled_model.model.model2().rhs_fun()) ),
      theta_(concat_par_init(coupled_model)),
      new_rate_(concat_rate_t0(coupled_model)),
      model_(coupled_model.model.model2().t0(),
             coupled_model.model.model2().y0(),
             new_rate_,
             theta_,
             f_,
             coupled_model.model.model2().ncmt())
    {}

    const PKODEModel<T_time, T_init, double, par_type, FA, int>& model() {
      return model_;
    }
  };
}

#endif

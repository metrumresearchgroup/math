#ifndef STAN_MATH_TORSTEN_COUPLED_CPT_ODE_MODEL_HPP
#define STAN_MATH_TORSTEN_COUPLED_CPT_ODE_MODEL_HPP

#include <stan/math/torsten/pk_coupled_model.hpp>

namespace refactor {

  template <typename F0, typename T_pksolver, template<typename...> class T_pkmodel>
  struct CptODEAugment {
    F0 f0_;

    CptODEAugment() { }

    explicit CptODEAugment(const F0& f0) : f0_(f0) { }

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

      int nPK = T_pkmodel<T0, T2, double, T2>::Ncmt;
      int n_pk_par = T_pkmodel<T0, T2, double, T2>::Npar;

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

      T_pkmodel<T0, T2, double, T2> pkmodel(t0,
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
      int nPK = T_pkmodel<T0, T2, T2, T2>::Ncmt;
      size_t nPD = y.size();
      size_t nODEparms = nTheta - 2 * nPK - nPD;  // number of ODE parameters
      nODEparms -= 1;

      // Theta first contains the base PK parameters, followed by
      // the other ODE parameters.
      int n_pk_par = T_pkmodel<T0, T2, double, T2>::Npar;

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
      T_pkmodel<T0, T2, T2, T2> pkmodel(t0,
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
      std::cout << "CptODEAugment: REPORT A BUG IF YOU SEE THIS." << std::endl;
      std::vector<T2> y_pk;
      return f0_(t, y, y_pk, theta, x_r, x_i, pstream_);
    }
  };

  // wrapper: one cpt model coupled with ode model
  template<typename T_time, typename T_init, typename T_rate, typename T_par, typename F, typename Ti>
  struct OneCptODEModel {
    using model_type = PKCoupledModel2<PKOneCptModel<T_time, T_init, T_rate, T_par>,
                                       PKODEModel<T_time,
                                                  T_init,
                                                  T_rate,
                                                  T_par,
                                                  CptODEAugment<F,refactor::PKOneCptModelSolver,refactor::PKOneCptModel>,
                                                  Ti> >;
    using model_type1 = typename model_type::model_type1;
    using model_type2 = typename model_type::model_type2;
    using scalar_type = typename model_type::scalar_type;
    using rate_type = T_rate;
    using par_type = T_par;
    using f_type = CptODEAugment<F,refactor::PKOneCptModelSolver,refactor::PKOneCptModel>;
    const CptODEAugment<F,refactor::PKOneCptModelSolver,refactor::PKOneCptModel> f_;
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
                                                  CptODEAugment<F,refactor::PKTwoCptModelSolver,refactor::PKTwoCptModel>,
                                                  Ti> >;
    using model_type1 = typename model_type::model_type1;
    using model_type2 = typename model_type::model_type2;
    using scalar_type = typename model_type::scalar_type;
    using rate_type = T_rate;
    using par_type = T_par;
    using f_type = CptODEAugment<F,refactor::PKTwoCptModelSolver,refactor::PKTwoCptModel>;
    const CptODEAugment<F,refactor::PKTwoCptModelSolver,refactor::PKTwoCptModel> f_;
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

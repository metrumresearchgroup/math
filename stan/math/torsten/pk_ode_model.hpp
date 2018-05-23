#ifndef PK_ODE_MODEL_HPP
#define PK_ODE_MODEL_HPP

namespace refactor {

  using boost::math::tools::promote_args;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  // nested class
  template<typename T_time, typename T_init, typename T_rate, typename T_par, typename F, typename Ti>
  class PKODEModel {
    const T_time &t0_;
    const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0_;
    const std::vector<T_rate> &rate_;
    const std::vector<T_par> &par_;
    const F &f_;
    const int ncmt_;
  public:
    // PKODEModel();
    // virtual ~PKODEModel();

    // static const int ncmt = ode_; 
    using scalar_type = typename promote_args<T_time, T_rate, T_par, T_init>::type;
    using aug_par_type = typename promote_args<T_rate, T_par, T_init>::type;
    using init_type   = T_init;
    using time_type   = T_time;
    using par_type    = T_par;
    using rate_type   = T_rate;
    using f_type      = F;

    // constructors
    template<template<typename...> class T_mp, typename... Ts>
    PKODEModel(const T_time& t0,
               const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0,
               const std::vector<T_rate> &rate,
               const std::vector<T_par> &par,
               const T_mp<Ts...> &parameter,
               const F& f,
               const Ti &ncmt) :
      t0_(t0),
      y0_(y0),
      rate_(rate),
      par_(par),
      f_(f),
      ncmt_(ncmt)
    {}

    PKODEModel(const T_time& t0,
               const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0,
               const std::vector<T_rate> &rate,
               const std::vector<T_par> &par,
               const F& f,
               const Ti &ncmt) :
      t0_(t0),
      y0_(y0),
      rate_(rate),
      par_(par),
      f_(f),
      ncmt_(ncmt)
    {}

    // copy constructor
    PKODEModel(const PKODEModel& other) :
      t0_(other.t0_),
      y0_(other.y0_),
      rate_(other.rate_),
      par_(other.par_),
      f_(other.f_),
      ncmt_(other.ncmt_)
    {}

    template<typename T>
    const PKODEModel<T_time, T_init, T_rate, T, F, Ti>
    make_with_new_par(const std::vector<T> &new_par) const {
      return PKODEModel<T_time, T_init, T_rate, T, F, Ti>(t0_, y0_, rate_, new_par, f_, ncmt_);
    }

    template<typename T>
    const PKODEModel<T_time, T_init, T_rate, T, F, Ti>
    make_with_new_par_rate(const std::vector<T> &new_par,
                           const std::vector<T_rate> &new_rate) const {
      return PKODEModel<T_time, T_init, T_rate, T, F, Ti>(t0_, y0_, new_rate, new_par, f_, ncmt_);
    }

    // get
    const T_time              & t0()       const { return t0_; }
    const PKRecord<T_init>    & y0()       const { return y0_; }
    const std::vector<T_rate> & rate()     const { return rate_; }
    const std::vector<T_par>  & par()      const { return par_; }
    const F                   & rhs_fun () const { return f_; }
    const int                 & ncmt ()    const { return ncmt_; }

    // can be solved by linear ode solver
  };

}




#endif

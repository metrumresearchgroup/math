#ifndef PK_ONECPT_MODEL_HPP
#define PK_ONECPT_MODEL_HPP

namespace refactor {

  using boost::math::tools::promote_args;

  // depend on model, we can have arbitrary number of
  // parameters, e.g. biovar, k12, k10, ka. Each parameter
  // can be data or var.
  template<typename T_time, typename T_init, typename T_rate, typename T_par>
  class PKOneCptModel {
    const T_time &t0_;
    const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0_;
    const std::vector<T_rate> &rate_;
    const T_par &CL_;
    const T_par &V2_;
    const T_par &ka_;
    const T_par k10_;
    const std::vector<T_par> alpha_;

  public:
    static const int ncmt = 2; 
    using scalar_type = typename promote_args<T_time, T_rate, T_par, T_init>::type;

    // constructors
    PKOneCptModel(const T_time& t0, const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0,
                  const std::vector<T_rate> &rate,
                  const T_par& CL,
                  const T_par& V2,
                  const T_par& ka) :
      t0_(t0),
      y0_(y0),
      rate_(rate),
      CL_(CL),
      V2_(V2),
      ka_(ka),
      k10_(CL / V2),
      alpha_{k10_, ka_}
    {}

    template<template<typename...> class T_mp, typename... Ts>
    PKOneCptModel(const T_time& t0, const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0,
                  const std::vector<T_rate> &rate,
                  const std::vector<T_par> & par,
                  const T_mp<Ts...> &parameter) :
      PKOneCptModel(t0, y0, rate, par[0], par[1], par[2])
    {}

    // PKOneCptModel(const T_time& t0, const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0,
    //               const Eigen::Matrix<T_par, Eigen::Dynamic, Eigen::Dynamic>& m) :
    //   // TODO
    //   t0_(t0_),
    //   y0_(y0)
    //   // CL_(m.Cl_),
    //   // V2_(m.V2_),
    //   // ka_(m.ka_)
    // {}

    // // copy constructor
    // PKOneCptModel(const PKOneCptModel& m) :
    //   PKOneCptModel(m.t0_, m.y0_, m.rate_, m.CL_, m.V2_, m.ka_)
    // {}

    // get
    const T_time              & t0()   { return t0_; }
    const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0() { return y0_; }
    const std::vector<T_rate> & rate() { return rate_; }
    const T_par               & CL()   { return CL_; }
    const T_par               & V2()   { return V2_; }
    const T_par               & ka()   { return ka_; }
    const T_par               & k10()  { return k10_; }
    const std::vector<T_par>  & alpha(){ return alpha_; }

    // can be solved by linear ode solver
  };

}

#endif

#ifndef PK_LINODE_MODEL_HPP
#define PK_LINODE_MODEL_HPP

#include <stan/math/torsten/torsten_def.hpp>

namespace refactor {

  using boost::math::tools::promote_args;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  /**
   * Linear ODE model.
   *
   * @tparam T_time type of time
   * @tparam T_init type of init condition
   * @tparam T_rate type of dosing rate
   * @tparam T_par  type of parameters.
   */
  template<typename T_time, typename T_init, typename T_rate, typename T_par>
  class PKLinODEModel {
    const T_time &t0_;
    const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0_;
    const std::vector<T_rate> &rate_;
    const Matrix<T_par, Dynamic, Dynamic> ode_; // FIXME: use reference

  public:
    using scalar_type = typename promote_args<T_time, T_rate, T_par, T_init>::type;
    using rate_type = T_rate;
    using par_type = T_par;

  /**
   * Constructor
   * FIXME need to remove parameter as this is for linode only.
   *
   * @tparam T_mp parameters class
   * @tparam Ts parameter types
   * @param t0 initial time
   * @param y0 initial condition
   * @param rate dosing rate
   * @param par model parameters
   * @param parameter ModelParameter type
   */
    template<template<typename...> class T_mp, typename... Ts>
    PKLinODEModel(const T_time& t0,
                  const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0,
                  const std::vector<T_rate> &rate,
                  const std::vector<T_par> & par,
                  const T_mp<Ts...> &parameter) :
      t0_(t0),
      y0_(y0),
      rate_(rate),
      ode_(parameter.get_K())
    {}

  /**
   * Get methods
   *
   */
    const T_time              & t0()          const { return t0_; }
    const PKRecord<T_init>    & y0()          const { return y0_; }
    const std::vector<T_rate> & rate()        const { return rate_; }
    const PKLinSystem<T_par>  & rhs_matrix () const { return ode_; }
    const int                 & ncmt ()       const { return ode_.rows(); }
  };

}

#endif

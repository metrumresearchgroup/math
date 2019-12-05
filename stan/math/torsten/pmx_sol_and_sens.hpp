#ifndef STAN_MATH_TORSTEN_SOL_AND_SENS_HPP
#define STAN_MATH_TORSTEN_SOL_AND_SENS_HPP

#include <Eigen/Dense>
#include <vector>

namespace torsten {

  /**
   * Calculate @var or @double solution of functors that
   * only returns @double results(potentially with graidents)
   *
   * @tparam F0 type of functor that calc solution
   * @tparam F1 type of functor that calc solution & sensitivity/graidents
   * @tparam T  type of parameters
   * @param[in] f0 functor that calculates solution
   * @param[in] f1 functor that calculates solution & sensitivity/graidents
   * @param[in] theta parameters to @f0 and @f1
   * @param[in] x_r @double data to @f0 and @f1
   * @param[in] x_i @int data to @f0 and @f1
   * @param[in] msg output stream
   * @return a solution vector. Here @theta is @double, so the
   * vector type is @double.
   *
   */
  template <typename F0, typename F1>
  std::vector<double> pmx_sol_and_sens(const F0& f0, const F1& f1,
                                       const std::vector<double>& theta,
                                       const std::vector<double>& x_r,
                                       const std::vector<int>& x_i,
                                       std::ostream* msgs) {
    std::vector<std::vector<double> > raw(f0(theta, x_r, x_i, msgs));
    std::vector<double> res(raw.size());
    std::transform(raw.begin(), raw.end(), res.begin(),
                   [](std::vector<double>& qoi_grad) -> double { return qoi_grad[0]; });
    return res;
  }


  /**
   * Calculate @var or @double solution of functors that
   * only returns @double results(potentially with graidents)
   *
   * @tparam F0 type of functor that calc solution
   * @tparam F1 type of functor that calc solution & sensitivity/graidents
   * @tparam T  type of parameters
   * @param[in] f0 functor that calculates solution
   * @param[in] f1 functor that calculates solution & sensitivity/graidents
   * @param[in] theta parameters to @f0 and @f1
   * @param[in] x_r @double data to @f0 and @f1
   * @param[in] x_i @int data to @f0 and @f1
   * @param[in] msg output stream
   * @return a solution vector. Here @theta is @var, so the
   * vector type is @var with gradients to @theta.
   *
   */
  template <typename F0, typename F1, typename T>
  std::vector<T> pmx_sol_and_sens(const F0& f0, const F1& f1,
                                  const std::vector<T>& theta,
                                  const std::vector<double>& x_r,
                                  const std::vector<int>& x_i,
                                  std::ostream* msgs) {
    std::vector<double> theta_d(stan::math::value_of(theta));
    std::vector<std::vector<double> > raw(f1(theta_d, x_r, x_i, msgs));
    std::vector<T> res(raw.size());
    std::transform(raw.begin(), raw.end(), res.begin(),
                   [&theta](std::vector<double>& qoi_grad) {
                     double qoi = qoi_grad[0];
                     std::vector<double> g(qoi_grad.begin() + 1, qoi_grad.end());
                     return stan::math::precomputed_gradients(qoi, theta, g);
                   });
    return res;
  }


}
#endif

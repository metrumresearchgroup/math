#ifndef STAN_MATH_PRIM_FUN_CORR_CONSTRAIN_HPP
#define STAN_MATH_PRIM_FUN_CORR_CONSTRAIN_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/log1m.hpp>
#include <stan/math/prim/fun/square.hpp>
#include <stan/math/prim/fun/sum.hpp>
#include <stan/math/prim/fun/tanh.hpp>
#include <cmath>

namespace stan {
namespace math {

/**
 * Return the result of transforming the specified scalar or container of values
 * to have a valid correlation value between -1 and 1 (inclusive).
 *
 * <p>The transform used is the hyperbolic tangent function,
 *
 * <p>\f$f(x) = \tanh x = \frac{\exp(2x) - 1}{\exp(2x) + 1}\f$.
 *
 * @tparam T type of value or container
 * @param[in] x value or container
 * @return tanh transform
 */
template <typename T>
inline plain_type_t<T> corr_constrain(const T& x) {
  return tanh(x);
}

/**
 * Return the result of transforming the specified scalar or container of values
 * to have a valid correlation value between -1 and 1 (inclusive).
 *
 * <p>The transform used is as specified for
 * <code>corr_constrain(T)</code>.  The log absolute Jacobian
 * determinant is
 *
 * <p>\f$\log | \frac{d}{dx} \tanh x  | = \log (1 - \tanh^2 x)\f$.
 *
 * @tparam T_x Type of scalar or container
 * @param[in] x value or container
 * @param[in,out] lp log density accumulator
 */
template <typename T_x, typename T_lp>
inline auto corr_constrain(const T_x& x, T_lp& lp) {
  plain_type_t<T_x> tanh_x = tanh(x);
  lp += sum(log1m(square(tanh_x)));
  return tanh_x;
}

/**
 * Return the result of transforming the specified scalar or container of values
 * to have a valid correlation value between -1 and 1 (inclusive).
 *
 * <p>The transform used is as specified for
 * <code>corr_constrain(T)</code>.  The log absolute Jacobian
 * determinant is
 *
 * <p>\f$\log | \frac{d}{dx} \tanh x  | = \log (1 - \tanh^2 x)\f$.
 *
 * @tparam Jacobian If true, incremented `lp` with the log Jacobian
 * @tparam T_x Type of scalar or container
 * @tparam T_lp A scalar
 * @param[in] x value or container
 * @param[in,out] lp log density accumulator
 */
template <bool Jacobian, typename T_x, typename T_lp>
inline auto corr_constrain(const T_x& x, T_lp& lp) {
  if (Jacobian) {
    return corr_constrain(x, lp);
  } else {
    return corr_constrain(x);
  }
}

}  // namespace math
}  // namespace stan
#endif

#ifndef STAN_MATH_FWD_SCAL_FUN_LOG_SUM_EXP_HPP
#define STAN_MATH_FWD_SCAL_FUN_LOG_SUM_EXP_HPP

#include <stan/math/fwd/meta.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/log_sum_exp.hpp>
#include <cmath>

namespace stan {
namespace math {

template <typename T>
inline fvar<T> log_sum_exp(const fvar<T>& x1, const fvar<T>& x2) {
  using std::exp;
  return fvar<T>(log_sum_exp(x1.val_, x2.val_),
                 x1.d_ / (1 + exp(x2.val_ - x1.val_))
                     + x2.d_ / (exp(x1.val_ - x2.val_) + 1));
}

template <typename T>
inline fvar<T> log_sum_exp(double x1, const fvar<T>& x2) {
  using std::exp;
  if (x1 == NEGATIVE_INFTY) {
    return fvar<T>(x2.val_, x2.d_);
  }
  return fvar<T>(log_sum_exp(x1, x2.val_), x2.d_ / (exp(x1 - x2.val_) + 1));
}

template <typename T>
inline fvar<T> log_sum_exp(const fvar<T>& x1, double x2) {
  using std::exp;
  if (x2 == NEGATIVE_INFTY) {
    return fvar<T>(x1.val_, x1.d_);
  }
  return fvar<T>(log_sum_exp(x1.val_, x2), x1.d_ / (1 + exp(x2 - x1.val_)));
}

}  // namespace math
}  // namespace stan
#endif
#ifndef STAN_MATH_FWD_ARR_FUN_LOG_SUM_EXP_HPP
#define STAN_MATH_FWD_ARR_FUN_LOG_SUM_EXP_HPP

#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/arr/fun/log_sum_exp.hpp>
#include <vector>
#include <cmath>

namespace stan {
namespace math {

template <typename T>
fvar<T> log_sum_exp(const std::vector<fvar<T> >& v) {
  using std::exp;
  std::vector<T> vals(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    vals[i] = v[i].val_;
  }
  T deriv(0.0);
  T denominator(0.0);
  for (size_t i = 0; i < v.size(); ++i) {
    T exp_vi = exp(vals[i]);
    denominator += exp_vi;
    deriv += v[i].d_ * exp_vi;
  }
  return fvar<T>(log_sum_exp(vals), deriv / denominator);
}

}  // namespace math
}  // namespace stan
#endif

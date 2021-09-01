#ifndef STAN_MATH_PRIM_FUN_CHOLESKY_FACTOR_FREE_HPP
#define STAN_MATH_PRIM_FUN_CHOLESKY_FACTOR_FREE_HPP

#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <cmath>
#include <stdexcept>

namespace stan {
namespace math {

/**
 * Return the unconstrained vector of parameters corresponding to
 * the specified Cholesky factor.  A Cholesky factor must be lower
 * triangular and have positive diagonal elements.
 *
 * @tparam T type of the Cholesky factor (must be derived from \c
 * Eigen::MatrixBase)
 * @param y Cholesky factor.
 * @return Unconstrained parameters for Cholesky factor.
 * @throw std::domain_error If the matrix is not a Cholesky factor.
 */
template <typename T, require_eigen_t<T>* = nullptr>
Eigen::Matrix<value_type_t<T>, Eigen::Dynamic, 1> cholesky_factor_free(
    const T& y) {
  using std::log;

  const auto& y_ref = to_ref(y);
  check_cholesky_factor("cholesky_factor_free", "y", y_ref);
  int M = y.rows();
  int N = y.cols();
  Eigen::Matrix<value_type_t<T>, Eigen::Dynamic, 1> x((N * (N + 1)) / 2
                                                      + (M - N) * N);
  int pos = 0;

  for (int m = 0; m < N; ++m) {
    x.segment(pos, m) = y_ref.row(m).head(m);
    pos += m;
    x.coeffRef(pos++) = log(y_ref.coeff(m, m));
  }

  for (int m = N; m < M; ++m) {
    x.segment(pos, N) = y_ref.row(m);
    pos += N;
  }
  return x;
}

template <typename T, require_std_vector_t<T>* = nullptr>
auto cholesky_factor_free(const T& x) {
  std::vector<decltype(cholesky_factor_free(x[0]))> x_free(x.size());
  std::transform(x.begin(), x.end(), x_free.begin(), [](auto&& xx) {
    return cholesky_factor_free(xx);
  });
  return x_free;
}

}  // namespace math
}  // namespace stan

#endif

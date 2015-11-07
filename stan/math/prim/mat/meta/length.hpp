#ifndef STAN_MATH_PRIM_MAT_META_LENGTH_HPP
#define STAN_MATH_PRIM_MAT_META_LENGTH_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <vector>

namespace stan {

  template <typename T, int R, int C>
  size_t length(const Eigen::Matrix<T, R, C>& m) {
    return m.size();
  }
  template <typename T, int R, int C>
  size_t length(const std::vector<Eigen::Matrix<T, R, C> >& m) {
    size_t accum = 0;
    for (size_t i = 0; i < m.size(); ++i)
      accum += static_cast<size_t>(m[i].size());
    return accum;
  }
}
#endif


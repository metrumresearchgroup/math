#ifndef STAN_MATH_PRIM_MAT_META_CONTAINS_MATRIX_HPP
#define STAN_MATH_PRIM_MAT_META_CONTAINS_MATRIX_HPP

#include<stan/math/prim/mat/fun/Eigen.hpp>
#include<stan/math/prim/scal/meta/contains_matrix.hpp>
#include<vector>

namespace stan {

  template <typename T>
  struct contains_matrix<Eigen::Matrix<T, -1, -1> > {
    enum { value = 1 };
  };

  template <typename T>
  struct contains_matrix<Eigen::Matrix<T, 1, -1> > {
    enum { value = 1 };
  };

  template <typename T>
  struct contains_matrix<Eigen::Matrix<T, -1, 1> > {
    enum { value = 1 };
  };

  template <typename T>
  struct contains_matrix<std::vector<T> > {
    enum { value = contains_matrix<T>::value };
  };
}
#endif



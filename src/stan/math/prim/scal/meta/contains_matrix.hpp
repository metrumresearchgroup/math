#ifndef STAN_MATH_PRIM_SCAL_META_CONTAINS_MATRIX_HPP
#define STAN_MATH_PRIM_SCAL_META_CONTAINS_MATRIX_HPP

namespace stan {

  template <typename T>
  struct contains_matrix {
    enum { value = false };
  };


}
#endif



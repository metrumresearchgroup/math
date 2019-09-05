#ifndef STAN_MATH_MIX_MAT_FUN_TYPEDEFS_HPP
#define STAN_MATH_MIX_MAT_FUN_TYPEDEFS_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>

namespace stan {
namespace math {

using matrix_fv = Eigen::Matrix<fvar<var>, Eigen::Dynamic, Eigen::Dynamic>;

typedef Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, Eigen::Dynamic>
    matrix_ffv;

using vector_fv = Eigen::Matrix<fvar<var>, Eigen::Dynamic, 1>;

using vector_ffv = Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, 1>;

using row_vector_fv = Eigen::Matrix<fvar<var>, 1, Eigen::Dynamic>;

using row_vector_ffv = Eigen::Matrix<fvar<fvar<var> >, 1, Eigen::Dynamic>;

}  // namespace math
}  // namespace stan
#endif

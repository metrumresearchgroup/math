#ifndef STAN__MATH__MATRIX__EIGENVECTORS_SYM_HPP
#define STAN__MATH__MATRIX__EIGENVECTORS_SYM_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/matrix/check_nonzero_size.hpp>
#include <stan/math/error_handling/matrix/check_symmetric.hpp>
#include <stan/math/error_handling/check_not_nan.hpp>

namespace stan {
  namespace math {

    /**
     * Return the eigenvectors of the specified symmetric matrix
     * in descending order of magnitude.
     *
     * <p>See <code>eigen_decompose()</code> for more information.
     * @param m Specified matrix.
     * @return Eigenvectors of matrix.
     * @throws if m is size 0 or if it is not symmetric or 
     *    if any elements are nan
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    eigenvectors_sym(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      stan::math::check_nonzero_size("eigenvectors_sym(%1%)",m,
                                     "m",(double*)0);
      stan::math::check_not_nan("eigenvalues_sym(%1%)",m,"m",(double*)0);
      stan::math::check_symmetric("eigenvalues_sym(%1%)",m,"m",(double*)0);

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >
        solver(m);
      return solver.eigenvectors(); 
    }

  }
}
#endif

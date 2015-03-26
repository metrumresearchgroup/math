#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/rev/mat/fun/Eigen_NumTraits.hpp>
#include <gtest/gtest.h>

#include <stan/math/prim/mat/prob/matrix_normal_prec_log.hpp>
#include <stan/math/rev/mat/fun/to_var.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/mat/fun/trace_gen_inv_quad_form_ldlt.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/mat/fun/log_determinant_ldlt.hpp>

// UTILITY FUNCTIONS FOR TESTING
#include <vector>
#include <test/unit/math/rev/mat/prob/expect_eq_diffs.hpp>
#include <test/unit/math/rev/mat/prob/test_gradients.hpp>
#include <test/unit/math/rev/mat/prob/test_gradients_matrix_normal.hpp>
#include <iostream>

void fill_vec_w_mat(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& filler,
                    std::vector<double>& vessel) {
  for (int i = 0; i < filler.cols(); ++i) {
    for (int j = 0; j < filler.rows(); ++j) {
      vessel.push_back(filler(j,i));
    }
  }
}

void fill_vec_w_sym_mat(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& filler,
                        std::vector<double>& vessel) {
  for (int i = 0; i < filler.cols(); ++i) {
    for (int j = 0; j <= i; ++j) {
      vessel.push_back(filler(j,i));
    }
  }
}


TEST(MatrixNormal, TestGradFunctionalVectorized) {

  std::vector<double> y_, mu_, sigma_, d_;

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> mu(3,5);
  mu.setZero();

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y(3,5);
  y << 2.0, -2.0, 11.0, 4.0, -2.0,
       11.0, 2.0, -5.0, 11.0, 0.0,
       -2.0, 11.0, 2.0, -2.0, -11.0;

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Sigma(5,5);
  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> D(3,3);
  D << 1.0, 0.5, 0.1,
       0.5, 1.0, 0.2,
       0.1, 0.2, 1.0;
  fill_vec_w_mat(y, y_);
  fill_vec_w_mat(mu, mu_);
  fill_vec_w_sym_mat(Sigma, sigma_);
  fill_vec_w_sym_mat(D, d_);

  test_grad_matrix_normal(matrix_normal_fun(3,5),y_, mu_, sigma_, d_);
}

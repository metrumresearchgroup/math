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
#include <stan/math/rev/mat/fun/trace_gen_quad_form.hpp>
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

template <typename T>
std::vector<T> fill_vec_w_mat(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& filler) {
  std::vector<T> vessel;
  for (int i = 0; i < filler.cols(); ++i) {
    for (int j = 0; j < filler.rows(); ++j) {
      vessel.push_back(filler(j,i));
    }
  }
  return vessel;
}

template <typename T>
std::vector<T> fill_vec_w_sym_mat(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& filler) {
  std::vector<T> vessel;
  for (int i = 0; i < filler.cols(); ++i) {
    for (int j = 0; j <= i; ++j) {
      vessel.push_back(filler(j,i));
    }
  }
  return vessel;
}

template <typename T_y, typename T_mu, typename T_sigma, typename T_D>
void test_types_matrix_normal(){
  Eigen::Matrix<T_y, -1, -1> y_use(3,5);
  Eigen::Matrix<T_sigma, -1, -1> sigma_use(5,5);
  Eigen::Matrix<T_mu, -1, -1> mu_use(3,5);
  Eigen::Matrix<T_D, -1, -1> D_use(3,3);
  mu_use.setZero();

  y_use << 2.0, -2.0, 11.0, 4.0, -2.0,
       11.0, 2.0, -5.0, 11.0, 0.0,
       -2.0, 11.0, 2.0, -2.0, -11.0;

  sigma_use << 9.0, -3.0, 0.0,  0.0, 0.0,
          -3.0,  4.0, 0.0,  0.0, 0.0,
           0.0,  0.0, 5.0,  1.0, 0.0,
           0.0,  0.0, 1.0, 10.0, 0.0,
           0.0,  0.0, 0.0,  0.0, 2.0;

  D_use << 1.0, 0.5, 0.1,
       0.5, 1.0, 0.2,
       0.1, 0.2, 1.0;

  std::vector<T_y> y_ = fill_vec_w_mat(y_use);
  std::vector<T_mu> mu_ = fill_vec_w_mat(mu_use);
  std::vector<T_sigma> sigma_ = fill_vec_w_sym_mat(sigma_use);
  std::vector<T_D> d_ = fill_vec_w_sym_mat(D_use);

  test_grad_matrix_normal(matrix_normal_fun(3,5),y_, mu_, sigma_, d_);
}

TEST(MatrixNormal, TestGradFunctionalVectorized) {
  using stan::agrad::var;
  test_types_matrix_normal<double,double,double,double>();
  test_types_matrix_normal<var,double,double,double>();
  test_types_matrix_normal<double,var,double,double>();
//  test_types_matrix_normal<var,var,double,double>();
  test_types_matrix_normal<double,double,var,double>();
  test_types_matrix_normal<var,double,var,double>();
//  test_types_matrix_normal<double,var,var,double>();
//  test_types_matrix_normal<var,var,var,double>();
  test_types_matrix_normal<double,double,double,var>();
  test_types_matrix_normal<var,double,double,var>();
//  test_types_matrix_normal<double,var,double,var>();
//  test_types_matrix_normal<var,var,double,var>();
  test_types_matrix_normal<double,double,var,var>();
  test_types_matrix_normal<var,double,var,var>();
//  test_types_matrix_normal<double,var,var,var>();
//  test_types_matrix_normal<var,var,var,var>();
}

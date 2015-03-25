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

//void fill_vec_w_mat(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& filler,
//                    std::vector<double>& vessel) {
//  for (int i = 0; i < filler.cols(); ++i) {
//    for (int j = 0; j < filler.rows(); ++j) {
//      vessel.push_back(filler(j,i));
//    }
//  }
//}

struct matrix_normal_fun {
  template <typename T_y, typename T_mu, typename T_sigma, typename T_D>
  typename boost::math::tools::promote_args<T_y, T_mu, T_sigma, T_D>::type
  operator() (const std::vector<T_y>& y_vec,
              const std::vector<T_mu>& mu_vec,
              const std::vector<T_sigma>& sigma_vec,
              const std::vector<T_D>& D_vec,
              int rows, int cols) {
    Eigen::Matrix<T_y,-1,-1> y(rows,cols);
    Eigen::Matrix<T_mu,-1,-1> mu(rows,cols);
    Eigen::Matrix<T_sigma,-1,-1> Sigma(cols, cols);
    Eigen::Matrix<T_D,-1,-1> D(rows, rows);

    int pos = 0;
    for (int k = 0; k < cols; ++k) 
      for (int l = 0; l < rows; ++l)
        y(l,k) = y_vec[pos++];
    std::cout << y << std::endl;

    pos = 0;        
    for (int k = 0; k < cols; ++k) 
      for (int l = 0; l < rows; ++l)
        mu(l,k) = mu_vec[pos++];
    
    pos = 0;
    for (int j = 0; j < cols; ++j) {
      for (int i = 0; i <= j; ++i) {
        Sigma(i,j) = sigma_vec[pos++];
        Sigma(j,i) = Sigma(i,j);
      }
    }

    pos = 0;
    for (int j = 0; j < rows; ++j) {
      for (int i = 0; i <= j; ++i) {
        D(i,j) = D_vec[pos++];
        D(j,i) = D(i,j);
      }
    }
    
  return stan::prob::matrix_normal_prec_log<false>(y, mu, Sigma, D);
};

TEST(MatrixNormal, TestGradFunctionalVectorized) {

//  std::vector<double> y_, mu_, sigma_, d_;
//
//  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> mu(3,5);
//  mu.setZero();
//
//  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y(3,5);
//  y << 2.0, -2.0, 11.0, 4.0, -2.0,
//       11.0, 2.0, -5.0, 11.0, 0.0,
//       -2.0, 11.0, 2.0, -2.0, -11.0;
//
//  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Sigma(5,5);
//  Sigma << 9.0, -3.0, 0.0,  0.0, 0.0,
//          -3.0,  4.0, 0.0,  0.0, 0.0,
//           0.0,  0.0, 5.0,  1.0, 0.0,
//           0.0,  0.0, 1.0, 10.0, 0.0,
//           0.0,  0.0, 0.0,  0.0, 2.0;
//
//  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> D(3,3);
//  D << 1.0, 0.5, 0.1,
//       0.5, 1.0, 0.2,
//       0.1, 0.2, 1.0;
//  fill_vec_w_mat(y, y_);
//  fill_vec_w_mat(mu, mu_);
//  fill_vec_w_mat(Sigma, sigma_);
//  fill_vec_w_mat(D, d_);
//
//  matrix_normal_fun f;
//  test_grad_matrix_normal(f,y_, mu_, sigma_, d_,3,5);
  std::cout << "Test" << std::endl;
}

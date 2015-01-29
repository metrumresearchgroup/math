#include <gtest/gtest.h>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/math/matrix/value_of_rec.hpp>
#include <test/unit/agrad/util.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <stan/agrad/finite_diff.hpp>

struct fun0 {
  template <typename T>
  inline
  T operator()(const T& x) const {
    return 5.0 * x * x * x;
  }
};

// fun1(x,y) = (x^2 * y) + (3 * y^2)
struct fun1 {
  template <typename T>
  inline
  T operator()(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) const {
    return x(0) * x(0) * x(0) * x(1) * x(1)
      + x(1) * x(1) * x(1) * x(0) + x(2) * x(2) * x(2) * x(1) * x(0); 
  }
};

std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >
grad_hess_fun1(const Eigen::Matrix<double,Eigen::Dynamic,1>& inp_vec){
  std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >grad_hess_ret;
  for(int i = 0; i < inp_vec.size(); ++i)
    grad_hess_ret.push_back(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>(3,3));

  double x = inp_vec(0);
  double y = inp_vec(1);
  double z = inp_vec(2);
  double x_sq = x * x;
  double y_sq = y * y;
  double z_sq = z * z;
  double zy = z * y;
  double zx = z * x;
  double yx = x * y;
  double xy = yx;
  
  grad_hess_ret[0] << 6 * y_sq, 12 * xy, 0,
                      12 * xy, 6 * x_sq + 6 * y, 3 * z_sq,
                      0, 3 * z_sq, 6 * zy;
  grad_hess_ret[1] << 12 * xy, 6 * x_sq + 6 * y, 3 * z_sq,
                      6 * x_sq + 6 * y, 6 * x, 0,
                      3 * z_sq, 0, 6 * zx;
  grad_hess_ret[2] << 0, 3 * z_sq, 6 * zy,
                      3 * z_sq, 0, 6 * zx,
                      6 * zy, 6 * zx, 6 * yx;
  return grad_hess_ret;
}

struct norm1 {
  template <typename T>
  inline
  T operator()(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) const {
    return stan::prob::normal_log(x(0), x(1), x(2)); 
  }
};

Eigen::Matrix<double,3,3>
norm_hess(const Eigen::Matrix<double,Eigen::Dynamic,1>& x){
  Eigen::Matrix<double,3,3> hess;
  double inv_sigma_sq = 1 / (x(2) * x(2));
  double y_m_mu = x(0) - x(1);
  double part_1_3 = 2 * y_m_mu * inv_sigma_sq / x(2);
  double part_3_3 = inv_sigma_sq - 3 * inv_sigma_sq * inv_sigma_sq * y_m_mu * y_m_mu;
  hess << -inv_sigma_sq, inv_sigma_sq, part_1_3,
       inv_sigma_sq, -inv_sigma_sq, -part_1_3,
       part_1_3, -part_1_3, part_3_3;
  return hess;
}


//TEST(AgradFiniteDiff,gradient) {
//  using Eigen::Matrix;
//  using Eigen::Dynamic;
//
//  fun1 f;
//  norm1 n;
//  Matrix<double,Dynamic,1> x(2);
//  Matrix<double,Dynamic,1> norm_vec(3);
//
//  x << 0.5, 0.3;
//
//
//  norm_vec << 0.5, 0.3, 0.7;
//  double fx;
//  double fin_diff_fx;
//
//  Matrix<double,Dynamic,1> grad_fx;
//  Matrix<double,Dynamic,1> finite_diff_fx;
//
//  Matrix<double,Dynamic,1> grad_norm_fx;
//  Matrix<double,Dynamic,1> finite_diff_norm_fx;
//
//  stan::agrad::gradient(f,x,fx,grad_fx);
//  stan::agrad::finite_diff_gradient(f,x,fin_diff_fx,finite_diff_fx);
//  EXPECT_NEAR(finite_diff_fx(0), grad_fx(0), 1e-06);
//  EXPECT_NEAR(finite_diff_fx(1), grad_fx(1), 1e-06);
//
//  stan::agrad::gradient(n,norm_vec,fx,grad_norm_fx);
//  stan::agrad::finite_diff_gradient(n,norm_vec,fx,finite_diff_norm_fx);
//
//  for (size_type i = 0; i < 3; ++i)
//    EXPECT_NEAR(grad_norm_fx(i), finite_diff_norm_fx(i), 1e-06);
//
//}
//
//TEST(AgradFiniteDiff,hessian) {
//  using Eigen::Matrix;  
//  using Eigen::Dynamic;
//  fun1 f;
//  norm1 n;
//  Matrix<double,Dynamic,1> x(2);
//  Matrix<double,Dynamic,1> norm_vec(3);
//
//  Matrix<double,2,2> fun1_hess;
//
//  x << 0.5, 0.7;
//
//  fun1_hess << 2 * x(1), 2 * x(0),
//               2 * x(0), 6;
//
//  norm_vec << 0.5, 0.3, 0.7;
//
//  double fx(0);
//  double finite_diff_x(0);
//  Matrix<double,Dynamic,1> grad;
//  Matrix<double,Dynamic,Dynamic> H;
//  Matrix<double,Dynamic,Dynamic> finite_diff_H;
//
//  Matrix<double,Dynamic,1> norm_grad;
//  Matrix<double,Dynamic,Dynamic> norm_H;
//  Matrix<double,Dynamic,Dynamic> finite_diff_norm_H;
//  stan::agrad::hessian(f,x,fx,grad,H);
//  stan::agrad::finite_diff_hessian(f,x,finite_diff_x,finite_diff_H);
//
//  stan::agrad::hessian(n,norm_vec,fx,norm_grad,norm_H);
//  stan::agrad::finite_diff_hessian(n,norm_vec,finite_diff_x,finite_diff_norm_H);
//
//  Matrix<double,3,3> norm_hess_an = norm_hess(norm_vec);
//
//  for (size_type i = 0; i < 2; ++i)
//    for (size_type j = 0; j < 2; ++j){
//      EXPECT_NEAR(H(i,j), finite_diff_H(i,j), 3e-04);
//      EXPECT_NEAR(fun1_hess(i,j), finite_diff_H(i,j),3e-04);
//    }
//
//  for (size_type i = 0; i < 3; ++i)
//    for (size_type j = 0; j < 3; ++j){
//      EXPECT_NEAR(norm_H(i,j), finite_diff_norm_H(i,j), 3e-04) << "i: " << i << " j: " << j;
//      EXPECT_NEAR(norm_hess_an(i,j), finite_diff_norm_H(i,j), 3e-04) << "i: " << i << " j: " << j;
//    }
//
//}

TEST(AgradFiniteDiff,grad_hessian) {
  using Eigen::Matrix;  
  using Eigen::Dynamic;
  norm1 n;
  fun1 fun;
  Matrix<double,Dynamic,1> norm_vec(3);
  norm_vec << 0.5, 0.3, 0.7;

  Matrix<double,Dynamic,1> fun_vec(3);
  fun_vec << 1.6666667e-01, 0.5, 1;

  double fx(0);
  double finite_diff_x(0);

  Matrix<double,Dynamic,Dynamic> norm_H_ad;
  std::vector<Matrix<double,Dynamic,Dynamic> > ad_grad_H;

  Matrix<double,Dynamic,1> norm_grad;
  Matrix<double,Dynamic,Dynamic> norm_H;
  std::vector<Matrix<double,Dynamic,Dynamic> > finite_diff_norm_H;
  std::vector<Matrix<double,Dynamic,Dynamic> > an_norm_H = 
    grad_hess_fun1(fun_vec);

  stan::agrad::grad_hessian(fun,fun_vec,fx,norm_H_ad,ad_grad_H);
  stan::agrad::finite_diff_grad_hessian(fun,fun_vec,finite_diff_x,finite_diff_norm_H);


  for (size_t i = 0; i < 3; ++i){
    for (size_type j = 0; j < 3; ++j){
      for (size_type k = 0; k < 3; ++k){
        EXPECT_NEAR(ad_grad_H[i](j,k),finite_diff_norm_H[i](j,k),1e-03) << " i: " << i << " j: " << j << " k: " << k; 
        EXPECT_NEAR(ad_grad_H[i](j,k),an_norm_H[i](j,k),1e-07) << " i: " << i << " j: " << j << " k: " << k; 
      }
    }
  }

//  for (size_t i = 0; i < 3; ++i){
//    for (size_type j = 0; j < 3; ++j){
//      for (size_type k = 0; k < 3; ++k)
//        std::cout << ad_grad_H[i](j,k) << " "; 
//      std::cout << std::endl;
//    }
//    std::cout << std::endl;
//  }

}

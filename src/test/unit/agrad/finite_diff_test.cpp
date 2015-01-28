#include <gtest/gtest.h>
#include <stan/agrad/rev/matrix/value_of_rec.hpp>
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
    return x(0) * x(0) * x(1)
      + 3.0 * x(1) * x(1); 
  }
};

struct norm1 {
  template <typename T>
  inline
  T operator()(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) const {
    return stan::prob::normal_log(x(0), x(1), x(2)); 
  }
};

TEST(AgradFiniteDiff,gradient) {
  using Eigen::Matrix;
  using Eigen::Dynamic;

  fun1 f;
  norm1 n;
  Matrix<double,Dynamic,1> x(2);
  Matrix<double,Dynamic,1> norm_vec(3);
  x << 0.5, 0.3;
  norm_vec << 0.5, 0.3, 0.7;
  double fx;
  double fin_diff_fx;

  Matrix<double,Dynamic,1> grad_fx;
  Matrix<double,Dynamic,1> finite_diff_fx;

  Matrix<double,Dynamic,1> grad_norm_fx;
  Matrix<double,Dynamic,1> finite_diff_norm_fx;

  stan::agrad::gradient(f,x,fx,grad_fx);
  stan::agrad::finite_diff_gradient(f,x,fin_diff_fx,finite_diff_fx);
  EXPECT_NEAR(finite_diff_fx(0), grad_fx(0), 1e-06);
  EXPECT_NEAR(finite_diff_fx(1), grad_fx(1), 1e-06);

  stan::agrad::gradient(n,norm_vec,fx,grad_norm_fx);
  stan::agrad::finite_diff_gradient(n,norm_vec,fx,finite_diff_norm_fx);

  for (size_type i = 0; i < 3; ++i)
    EXPECT_NEAR(grad_norm_fx(i), finite_diff_norm_fx(i), 1e-06);

}

TEST(AgradFiniteDiff,hessian) {
  using Eigen::Matrix;  
  using Eigen::Dynamic;
  fun1 f;
  norm1 n;
  Matrix<double,Dynamic,1> x(2);
  Matrix<double,Dynamic,1> norm_vec(3);
  x << 5, 7;
  norm_vec << 0.5, 0.3, 0.7;

  double fx(0);
  double finite_diff_x(0);
  Matrix<double,Dynamic,1> grad;
  Matrix<double,Dynamic,Dynamic> H;
  Matrix<double,Dynamic,Dynamic> finite_diff_H;
  Matrix<double,Dynamic,1> norm_grad;
  Matrix<double,Dynamic,Dynamic> norm_H;
  Matrix<double,Dynamic,Dynamic> finite_diff_norm_H;
  stan::agrad::hessian(f,x,fx,grad,H);
  stan::agrad::finite_diff_hessian(f,x,finite_diff_x,finite_diff_H);

  stan::agrad::hessian(n,norm_vec,fx,norm_grad,norm_H);
  stan::agrad::finite_diff_hessian(n,norm_vec,finite_diff_x,finite_diff_norm_H);

  for (size_type i = 0; i < 2; ++i)
    for (size_type j = 0; j < 2; ++j)
      EXPECT_NEAR(H(i,j), finite_diff_H(i,j), 1e-06);

  for (size_type i = 0; i < 3; ++i)
    for (size_type j = 0; j < 3; ++j)
      EXPECT_NEAR(norm_H(i,j), finite_diff_norm_H(i,j), 1e-06);

}

TEST(AgradFiniteDiff,grad_hessian) {
  using Eigen::Matrix;  
  using Eigen::Dynamic;
  norm1 n;
  Matrix<double,Dynamic,1> norm_vec(3);
  norm_vec << 0.5, 0.3, 0.7;

  double fx(0);
  double finite_diff_x(0);
  Matrix<double,Dynamic,1> norm_grad;
  Matrix<double,Dynamic,Dynamic> norm_H;
  std::vector<Matrix<double,Dynamic,Dynamic> > finite_diff_norm_H;

//  stan::agrad::hessian(n,norm_vec,fx,norm_grad,norm_H);
  stan::agrad::finite_diff_grad_hessian(n,norm_vec,finite_diff_x,finite_diff_norm_H);


  for (size_t i = 0; i < 3; ++i){
    for (size_type j = 0; j < 3; ++j){
      for (size_type k = 0; k < 3; ++k)
        std::cout << finite_diff_norm_H[i](j,k) << " "; 
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

}

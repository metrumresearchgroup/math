#include <stan/math/matrix/rows_dot_product.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(MathMatrix,rows_dot_product_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(3, 2);
  m1 << 14, nan,
        3, 4.1,
        10, 4.1;
        
  Eigen::MatrixXd m2(3, 2);
  m2 << 10.1, 1,
        nan, 1,
        5.4, 1.5;
  
  Eigen::VectorXd vr;
  
  using stan::math::rows_dot_product;
  using boost::math::isnan;

  vr = rows_dot_product(m1, m2);
  EXPECT_PRED1(isnan<double>, vr(0));
  EXPECT_PRED1(isnan<double>, vr(1));
  EXPECT_DOUBLE_EQ(10*5.4 + 4.1*1.5, vr(2));
}

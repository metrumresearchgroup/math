#include <stan/math/matrix/columns_dot_product.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(MathMatrix,columns_dot_product_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(2,3);
  m1 << 14, nan, 3,
        nan, 10, 4.1;
        
  Eigen::MatrixXd m2(2,3);
  m2 << 10.1, 1, 1.3,
        1, 5.4, 1.5;
  
  Eigen::RowVectorXd vr;
  
  using stan::math::columns_dot_product;
  using boost::math::isnan;

  vr = columns_dot_product(m1, m2);
  EXPECT_PRED1(isnan<double>, vr(0));
  EXPECT_PRED1(isnan<double>, vr(1));
  EXPECT_DOUBLE_EQ(3*1.3 + 4.1*1.5, vr(2));
}

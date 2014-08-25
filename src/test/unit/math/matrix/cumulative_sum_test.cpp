#include <stan/math/matrix/cumulative_sum.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

template <typename T>
void test_cumulative_sum() {
  using stan::math::cumulative_sum;

  T c(1);
  c[0] = 1.7;
  T d = cumulative_sum(c);
  EXPECT_EQ(c.size(), d.size());
  EXPECT_FLOAT_EQ(c[0],d[0]);

  T e(2);
  e[0] = 5.9;  e[1] = -1.2;
  T f = cumulative_sum(e);
  EXPECT_EQ(e.size(), f.size());
  EXPECT_FLOAT_EQ(e[0],f[0]);
  EXPECT_FLOAT_EQ(e[0] + e[1], f[1]);

  T g(3);
  g[0] = 5.9;  g[1] = -1.2;   g[2] = 192.13456;
  T h = cumulative_sum(g);
  EXPECT_EQ(g.size(), h.size());
  EXPECT_FLOAT_EQ(g[0],h[0]);
  EXPECT_FLOAT_EQ(g[0] + g[1], h[1]);
  EXPECT_FLOAT_EQ(g[0] + g[1] + g[2], h[2]);
}

TEST(MathMatrix, cumulative_sum) {
  using stan::math::cumulative_sum;

  EXPECT_FLOAT_EQ(0, cumulative_sum(std::vector<double>(0)).size());

  Eigen::Matrix<double,Eigen::Dynamic,1> a;
  EXPECT_FLOAT_EQ(0,cumulative_sum(a).size());

  Eigen::Matrix<double,1,Eigen::Dynamic> b;
  EXPECT_FLOAT_EQ(0,cumulative_sum(b).size());

  test_cumulative_sum<std::vector<double> >();
  test_cumulative_sum<Eigen::Matrix<double,Eigen::Dynamic,1> >();
  test_cumulative_sum<Eigen::Matrix<double,1,Eigen::Dynamic> >();
}

TEST(MathMatrix, cumulative_sum_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(3,2);
  m1 << 1, nan,
        3.1, 4.1,
        nan, 6;
  Eigen::MatrixXd m2(3,2);
  m2 << 10.1, 100,
        nan, 0,
        -10, -12;
        
  Eigen::VectorXd v1(3);
  v1 << 100.1, nan, 1.1;
        
  std::vector<double> v2(14, 1.1);
  v2[2] = nan;
  
  Eigen::MatrixXd mr;
  std::vector<double> vr;
        
  using stan::math::cumulative_sum;
  using boost::math::isnan;
    
  mr = cumulative_sum(m1);
  EXPECT_DOUBLE_EQ(1, mr(0));
  EXPECT_DOUBLE_EQ(4.1, mr(1));
  for (int i = 2; i < 6; i++)
    EXPECT_PRED1(isnan<double>, mr(i));
    
  mr = cumulative_sum(m2);
  EXPECT_DOUBLE_EQ(10.1, mr(0));
  for (int i = 1; i < 6; i++)
    EXPECT_PRED1(isnan<double>, mr(i));
    
  mr = cumulative_sum(v1);
  EXPECT_DOUBLE_EQ(100.1, mr(0));
  for (int i = 1; i < 3; i++)
    EXPECT_PRED1(isnan<double>, mr(i));
    
  vr = cumulative_sum(v2);
  EXPECT_DOUBLE_EQ(1.1, vr[0]);
  EXPECT_DOUBLE_EQ(2.2, vr[1]);
  for (int i = 2; i < 14; i++)
    EXPECT_PRED1(isnan<double>, vr[i]);

}

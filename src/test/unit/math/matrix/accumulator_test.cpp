#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/accumulator.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

// test sum of first n numbers for sum of a
template <typename T>
void test_sum(stan::math::accumulator<T>& a,
              int n) {
  EXPECT_FLOAT_EQ((n * (n + 1)) / 2, a.sum());
}

TEST(MathMatrix,accumulateDouble) {
  using stan::math::accumulator;
  
  accumulator<double> a;
  test_sum(a, 0);

  a.add(1.0);
  test_sum(a, 1);

  for (int i = 2; i <= 1000; ++i)
    a.add(i);
  test_sum(a, 1000);
  
}
TEST(MathMatrix,accumulateCollection) {
  // tests int, double, vector<double>, vector<int>
  // MatrixXd, VectorXd, and recursions of vector<T>

  using stan::math::accumulator;
  using std::vector;
  using Eigen::VectorXd;
  using Eigen::MatrixXd;

  accumulator<double> a;

  int pos = 0;
  test_sum(a, 0);

  vector<double> v(10);
  for (size_t i = 0; i < 10; ++i)
    v[i] = pos++;
  a.add(v);                                         
  test_sum(a, pos-1);

  a.add(pos++);                    
  test_sum(a, pos-1);

  double x = pos++;
  a.add(x);                        
  test_sum(a, pos-1);

  vector<int> u(10);         
  for (size_t i = 0; i < 10; ++i)
    a.add(pos++);
  test_sum(a, pos-1);

  vector<vector<int> > ww(10);
  for (size_t i = 0; i < 10; ++i) {
    vector<int> w(5);
    for (size_t n = 0; n < 5; ++n)
      w[n] = pos++;
    ww[i] = w;
  }
  a.add(ww);
  test_sum(a, pos-1);

  MatrixXd m(5,6);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 6; ++j)
      m(i,j) = pos++;
  a.add(m);
  test_sum(a, pos-1);

  VectorXd mv(7);
  for (int i = 0; i < 7; ++i)
    mv(i) = pos++;
  a.add(mv);
  test_sum(a, pos-1);

  vector<VectorXd> vvx(8);
  for (size_t i = 0; i < 8; ++i) {
    VectorXd vx(3);
    for (int j = 0; j < 3; ++j)
      vx(j) = pos++;
    vvx[i] = vx;
  }
  a.add(vvx);
  test_sum(a, pos-1);
}
TEST(MathMatrix, accumulator_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  using stan::math::accumulator;
  using boost::math::isnan;
  
  accumulator<double> a;
  a.add(nan);

  EXPECT_PRED1(isnan<double>, a.sum());
  
  accumulator<double> b;
  b.add(1);
  b.add(3);
  
  stan::math::matrix_d m1(2, 1);
  m1 << 1, nan;
  
  b.add(m1);

  EXPECT_PRED1(isnan<double>, b.sum());
  
  accumulator<double> c;
  for (size_t i = 0; i < 14; i++)
    c.add(i);
  
  std::vector<double> v2(14, 1.1);
  v2[7] = nan;
  
  c.add(v2);

  EXPECT_PRED1(isnan<double>, c.sum());
}




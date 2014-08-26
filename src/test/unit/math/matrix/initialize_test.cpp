#include <stan/agrad/rev.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/initialize.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>

TEST(MathMatrix,initialize) {
  // 2nd template
  using stan::math::initialize;
  double x;
  double y = 10;
  initialize(x,y);         // template 2
  EXPECT_FLOAT_EQ(y,x);

  int z = 5;
  initialize(y,z);        // template 2
  EXPECT_FLOAT_EQ(z,y);
}

TEST(MathMatrix,initializeVar) {
  using stan::agrad::var;
  using stan::math::initialize;
  var a;
  var b = 10;
  initialize(a,b);           // template 1
  EXPECT_FLOAT_EQ(10, a.val());

  initialize(a, 5);          // template 2
  EXPECT_FLOAT_EQ(5, a.val());

  initialize(a, 13.2);      // template 2
  EXPECT_FLOAT_EQ(13.2, a.val());

}
  
TEST(MathMatrix, initMatrix) {
  using stan::agrad::var;
  using stan::math::initialize;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  Matrix<double,Dynamic,Dynamic> m(3,2);
  initialize(m, 13.2);      // template 3, 2
  for (int i = 0; i < m.size(); ++i)
    EXPECT_FLOAT_EQ(m(i), 13.2);

  Matrix<double,Dynamic,1> v(3);
  initialize(v,2);           // template 3, 2
  for (int i = 0; i < v.size(); ++i)
    EXPECT_FLOAT_EQ(v(i), 2);

  Matrix<double,1,Dynamic> rv(3);
  initialize(rv,12);         // template 3, 2
  for (int i = 0; i < v.size(); ++i)
    EXPECT_FLOAT_EQ(rv(i), 12);

  Matrix<var, Dynamic, Dynamic> mvar(2,3);
  initialize(mvar, 2.3);     // template 3, 1
  for (int i = 0; i < mvar.size(); ++i)
    EXPECT_FLOAT_EQ(mvar(i).val(), 2.3);
  
}
TEST(MathMatrix, initStdVector) {
  using std::vector;
  using stan::agrad::var;
  using stan::math::initialize;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  vector<double> x(3);
  initialize(x,2.2);
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(2.2, x[i]);  // template 4,2

  vector<Matrix<double,Dynamic,Dynamic> > z(4, Matrix<double,Dynamic,Dynamic>(3,2));
  initialize(z, 3.7);
  for (size_t i = 0; i < 4; ++i)
    for (int m = 0; m < 3; ++m)
      for (int n = 0; n < 2; ++n)
        EXPECT_FLOAT_EQ(3.7, z[i](m,n));
}

TEST(MathMatrix, initialize_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(3,2);
  Eigen::MatrixXd m2(3,2);
  Eigen::VectorXd v1(3);
  Eigen::RowVectorXd v2(3);
  std::vector<double> v3(14, 1.1);
        
  using stan::math::initialize;
  using boost::math::isnan;
  
  initialize(m1, nan);
  initialize(m2, nan);
  initialize(v1, nan);
  initialize(v2, nan);
  initialize(v3, nan);
    
  expect_matrix_is_nan(m1);
  expect_matrix_is_nan(m2);
  expect_matrix_is_nan(v1);
  expect_matrix_is_nan(v2);
  expect_matrix_is_nan(v3);
}

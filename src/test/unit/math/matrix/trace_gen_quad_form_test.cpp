#include <stan/math/matrix/trace_gen_quad_form.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, trace_gen_quad_form_mat) {
  using stan::math::trace_gen_quad_form;
  using stan::math::matrix_d;
  
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  matrix_d cd(2,2);
  double res;
  
  
  bd << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  cd.setIdentity(2,2);
  
  // double-double-double
  res = trace_gen_quad_form(cd,ad,bd);
  EXPECT_FLOAT_EQ(26758, res);
}

TEST(MathMatrix, trace_gen_quad_form_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(3,3);
  m1 << 1, nan, 1.1,
        nan, 4.1, 4.5,
        nan, nan, 5.1;
        
  Eigen::MatrixXd m2(3,3);
  m2 << 10.1, 100, 2,
        1.0, 1, nan,
        -10, -12, 6;

  Eigen::MatrixXd m3(3,3);
  m3 << 1, nan, 5.1,
        3, 4.1, 1.1,
        nan, 6, nan;
        
  Eigen::MatrixXd m4(3,3);
  m4 << 10.1, 100, 4,
        0.1, nan, 5,
        -10, -12, 7;
        
  Eigen::MatrixXd m1c(3,3);
  m1c << 1, 1.1, 4,
        1.3, 4.1, 3,
        4.1, 1.6, 1;
        
  Eigen::MatrixXd m2c(3,3);
  m2c << 10.1, 100, 2,
        1.0, 1, 7,
        -10, -12, 6;

  Eigen::MatrixXd m3c(3,3);
  m3c << 1, 0.1, 5.1,
        3, 4.1, 1.1,
        4, 6, .4;
        
  Eigen::MatrixXd m4c(3,3);
  m4c << 10.1, 100, 1,
        0.1, 6, 6,
        -10, -12, 1.1;

  using stan::math::trace_gen_quad_form;
  using boost::math::isnan;
  
  std::vector <Eigen::MatrixXd> vec(8);
  vec[0] = m1;
  vec[1] = m2;
  vec[2] = m3;
  vec[3] = m4;
  vec[4] = m1c;
  vec[5] = m2c;
  vec[6] = m3c;
  vec[7] = m4c;
  
  for (size_t i = 0; i < 8; i++)
    for (size_t j = 0; j < 8; j++)
      for (size_t k = 0; k < 8; k++)
         if (i < 4 || j < 4 || k < 4)
           EXPECT_PRED1(isnan<double>, trace_gen_quad_form(vec[i], vec[j], vec[k]));
}

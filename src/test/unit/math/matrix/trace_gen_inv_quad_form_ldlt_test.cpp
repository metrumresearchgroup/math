#include <stan/math/matrix/trace_gen_inv_quad_form_ldlt.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/trace.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>


    /*
     * Compute the trace of an inverse quadratic form.  I.E., this computes
     *       trace(D B^T A^-1 B)
     * where D is a square matrix and the LDLT_factor of A is provided.
     */

TEST(MathMatrix, trace_gen_inv_quad_form_ldlt) {
  using stan::math::matrix_d;
  matrix_d D(2,2), A(4,4), B(4,2), gen_inv_quad_form;
  
  D << 1, 2, 3, 4;
  A << 9.0,  3.0, 3.0,   3.0, 
       3.0, 10.0, 2.0,   2.0,
       3.0,  2.0, 7.0,   1.0,
       3.0,  2.0, 1.0, 112.0;
  B << 100, 10,
    0,  1,
    -3, -3,
    5,  2;
  
  gen_inv_quad_form = D * B.transpose() * A.inverse() * B;
  

  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());
  
  EXPECT_FLOAT_EQ(stan::math::trace(gen_inv_quad_form),
                  stan::math::trace_gen_inv_quad_form_ldlt(D, ldlt_A, B));
}

TEST(MathMatrix, trace_gen_inv_quad_form_ldlt_nan) {
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
        
  stan::math::LDLT_factor<double,-1,-1> ldlt_m1;
  ldlt_m1.compute(m1);
  stan::math::LDLT_factor<double,-1,-1> ldlt_m2;
  ldlt_m2.compute(m2);
  stan::math::LDLT_factor<double,-1,-1> ldlt_m3;
  ldlt_m3.compute(m3);
  stan::math::LDLT_factor<double,-1,-1> ldlt_m4;
  ldlt_m4.compute(m4);
  stan::math::LDLT_factor<double,-1,-1> ldlt_m1c;
  ldlt_m1c.compute(m1c);
  stan::math::LDLT_factor<double,-1,-1> ldlt_m2c;
  ldlt_m2c.compute(m2c);
  stan::math::LDLT_factor<double,-1,-1> ldlt_m3c;
  ldlt_m3c.compute(m3c);
  stan::math::LDLT_factor<double,-1,-1> ldlt_m4c;
  ldlt_m4c.compute(m4c);

  using stan::math::trace_gen_inv_quad_form_ldlt;
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
  
  std::vector <stan::math::LDLT_factor<double,-1,-1> > ldlt_vec(8);
  ldlt_vec[0] = ldlt_m1;
  ldlt_vec[1] = ldlt_m2;
  ldlt_vec[2] = ldlt_m3;
  ldlt_vec[3] = ldlt_m4;
  ldlt_vec[4] = ldlt_m1c;
  ldlt_vec[5] = ldlt_m2c;
  ldlt_vec[6] = ldlt_m3c;
  ldlt_vec[7] = ldlt_m4c;
  
  for (size_t i = 0; i < 8; i++)
    for (size_t j = 0; j < 8; j++)
      for (size_t k = 0; k < 8; k++)
         if (i < 4 || k < 4)
           EXPECT_PRED1(isnan<double>, trace_gen_inv_quad_form_ldlt(vec[i], ldlt_vec[j], vec[k]));
}

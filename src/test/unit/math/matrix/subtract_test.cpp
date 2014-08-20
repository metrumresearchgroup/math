#include <stan/math/matrix/subtract.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>

TEST(MathMatrix,subtract_v_exception) {
  stan::math::vector_d d1, d2;

  d1.resize(3);
  d2.resize(3);
  EXPECT_NO_THROW(stan::math::subtract(d1, d2));

  d1.resize(0);
  d2.resize(0);
  EXPECT_NO_THROW(stan::math::subtract(d1, d2));

  d1.resize(2);
  d2.resize(3);
  EXPECT_THROW(stan::math::subtract(d1, d2), std::domain_error);
}
TEST(MathMatrix,subtract_rv_exception) {
  stan::math::row_vector_d d1, d2;

  d1.resize(3);
  d2.resize(3);
  EXPECT_NO_THROW(stan::math::subtract(d1, d2));

  d1.resize(0);
  d2.resize(0);
  EXPECT_NO_THROW(stan::math::subtract(d1, d2));

  d1.resize(2);
  d2.resize(3);
  EXPECT_THROW(stan::math::subtract(d1, d2), std::domain_error);
}
TEST(MathMatrix,subtract_m_exception) {
  stan::math::matrix_d d1, d2;

  d1.resize(2,3);
  d2.resize(2,3);
  EXPECT_NO_THROW(stan::math::subtract(d1, d2));

  d1.resize(0,0);
  d2.resize(0,0);
  EXPECT_NO_THROW(stan::math::subtract(d1, d2));

  d1.resize(2,3);
  d2.resize(3,3);
  EXPECT_THROW(stan::math::subtract(d1, d2), std::domain_error);
}

TEST(MathMatrix,subtract_c_m) {
  stan::math::matrix_d v(2,2);
  v << 1, 2, 3, 4;
  stan::math::matrix_d result;

  result = stan::math::subtract(2.0,v);
  EXPECT_FLOAT_EQ(1.0,result(0,0));
  EXPECT_FLOAT_EQ(0.0,result(0,1));
  EXPECT_FLOAT_EQ(-1.0,result(1,0));
  EXPECT_FLOAT_EQ(-2.0,result(1,1));

  result = stan::math::subtract(v,2.0);
  EXPECT_FLOAT_EQ(-1.0,result(0,0));
  EXPECT_FLOAT_EQ(0.0,result(0,1));
  EXPECT_FLOAT_EQ(1.0,result(1,0));
  EXPECT_FLOAT_EQ(2.0,result(1,1));
}

TEST(MathMatrix,subtract_c_rv) {
  stan::math::row_vector_d v(3);
  v << 1, 2, 3;
  stan::math::row_vector_d result;

  result = stan::math::subtract(2.0,v);
  EXPECT_FLOAT_EQ(1.0,result(0));
  EXPECT_FLOAT_EQ(0.0,result(1));
  EXPECT_FLOAT_EQ(-1.0,result(2));

  result = stan::math::subtract(v,2.0);
  EXPECT_FLOAT_EQ(-1.0,result(0));
  EXPECT_FLOAT_EQ(0.0,result(1));
  EXPECT_FLOAT_EQ(1.0,result(2));
}


TEST(MathMatrix,subtract_c_v) {
  stan::math::vector_d v(3);
  v << 1, 2, 3;
  stan::math::vector_d result;

  result = stan::math::subtract(2.0,v);
  EXPECT_FLOAT_EQ(1.0,result(0));
  EXPECT_FLOAT_EQ(0.0,result(1));
  EXPECT_FLOAT_EQ(-1.0,result(2));

  result = stan::math::subtract(v,2.0);
  EXPECT_FLOAT_EQ(-1.0,result(0));
  EXPECT_FLOAT_EQ(0.0,result(1));
  EXPECT_FLOAT_EQ(1.0,result(2));
}

TEST(MathMatrix, subtract_exception) {
  stan::math::vector_d v1(2);
  v1 << 1, 2;
  stan::math::vector_d v2(3);
  v2 << 10, 100, 1000;

  stan::math::row_vector_d rv1(2);
  v1 << 1, 2;
  stan::math::row_vector_d rv2(3);
  v2 << 10, 100, 1000;

  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::math::matrix_d m2(3,2);
  m2 << 10, 100, 1000, 0, -10, -12;

  using stan::math::subtract;
  EXPECT_THROW(subtract(v1,v2),std::domain_error);
  EXPECT_THROW(subtract(rv1,rv2),std::domain_error);
  EXPECT_THROW(subtract(m1,m2),std::domain_error);
}


TEST(MathMatrix, subtract_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  stan::math::matrix_d m1(2,2);
  m1 << 1, nan,
        3, 6;
        
  stan::math::matrix_d m2(2,2);
  m2 << 1, 100,
        nan, 4.9;
        
  stan::math::matrix_d m3(2,2);
  m3 << 10.1, 100,
        1, 0;
        
  stan::math::matrix_d mr;

  using stan::math::subtract;
  using boost::math::isnan;
  
  mr = subtract(m1, m2);
  
  EXPECT_DOUBLE_EQ(0, mr(0, 0));
  EXPECT_PRED1(isnan<double>, mr(0, 1));
  EXPECT_PRED1(isnan<double>, mr(1, 0));
  EXPECT_DOUBLE_EQ(1.1, mr(1, 1));
  
  mr = subtract(m1, 0.1);
  
  EXPECT_DOUBLE_EQ(0.9, mr(0, 0));
  EXPECT_PRED1(isnan<double>, mr(0, 1));
  EXPECT_DOUBLE_EQ(2.9, mr(1, 0));
  EXPECT_DOUBLE_EQ(5.9, mr(1, 1));
   
  mr = subtract(0.1, m1);
  
  EXPECT_DOUBLE_EQ(-0.9, mr(0, 0));
  EXPECT_PRED1(isnan<double>, mr(0, 1));
  EXPECT_DOUBLE_EQ(-2.9, mr(1, 0));
  EXPECT_DOUBLE_EQ(-5.9, mr(1, 1)); 
  
  expect_matrix_is_nan(subtract(m2, nan));
  expect_matrix_is_nan(subtract(m3, nan));
  expect_matrix_is_nan(subtract(nan, m1));
  expect_matrix_is_nan(subtract(nan, m3));
}

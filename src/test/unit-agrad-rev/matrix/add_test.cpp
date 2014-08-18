#include <stan/math/matrix/add.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/agrad/rev.hpp>

TEST(AgradRevMatrix,add_scalar) {
  using stan::agrad::matrix_v;
  using stan::math::add;

  matrix_v v(2,2);
  v << 1, 2, 3, 4;
  matrix_v result;

  result = add(2.0,v);
  EXPECT_FLOAT_EQ(3.0,result(0,0).val());
  EXPECT_FLOAT_EQ(4.0,result(0,1).val());
  EXPECT_FLOAT_EQ(5.0,result(1,0).val());
  EXPECT_FLOAT_EQ(6.0,result(1,1).val());

  result = add(v,2.0);
  EXPECT_FLOAT_EQ(3.0,result(0,0).val());
  EXPECT_FLOAT_EQ(4.0,result(0,1).val());
  EXPECT_FLOAT_EQ(5.0,result(1,0).val());
  EXPECT_FLOAT_EQ(6.0,result(1,1).val());
}

TEST(AgradRevMatrix, add_vector_vector) {
  using stan::math::add;
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  vector_d vd_1(5);
  vector_d vd_2(5);
  vector_v vv_1(5);
  vector_v vv_2(5);
  
  vd_1 << 1, 2, 3, 4, 5;
  vv_1 << 1, 2, 3, 4, 5;
  vd_2 << 2, 3, 4, 5, 6;
  vv_2 << 2, 3, 4, 5, 6;
  
  vector_d expected_output(5);
  expected_output << 3, 5, 7, 9, 11;
  
  vector_d output_d;
  output_d = add(vd_1, vd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_d(0));
  EXPECT_FLOAT_EQ(expected_output(1), output_d(1));
  EXPECT_FLOAT_EQ(expected_output(2), output_d(2));
  EXPECT_FLOAT_EQ(expected_output(3), output_d(3));
  EXPECT_FLOAT_EQ(expected_output(4), output_d(4));  

  vector_v output_v = add(vv_1, vd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val());  

  output_v = add(vd_1, vv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val());  

  output_v = add(vv_1, vv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val());  
}
TEST(AgradRevMatrix, add_vector_vector_exception) {
  using stan::math::add;
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  vector_d d1(5), d2(1);
  vector_v v1(5), v2(1);
  
  EXPECT_THROW(add(d1, d2), std::domain_error);
  EXPECT_THROW(add(v1, d2), std::domain_error);
  EXPECT_THROW(add(d1, v2), std::domain_error);
  EXPECT_THROW(add(v1, v2), std::domain_error);
}
TEST(AgradRevMatrix, add_rowvector_rowvector) {
  using stan::math::add;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  row_vector_d rvd_1(5), rvd_2(5);
  row_vector_v rvv_1(5), rvv_2(5);

  rvd_1 << 1, 2, 3, 4, 5;
  rvv_1 << 1, 2, 3, 4, 5;
  rvd_2 << 2, 3, 4, 5, 6;
  rvv_2 << 2, 3, 4, 5, 6;
  
  row_vector_d expected_output(5);
  expected_output << 3, 5, 7, 9, 11;
  
  row_vector_d output_d = add(rvd_1, rvd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_d(0));
  EXPECT_FLOAT_EQ(expected_output(1), output_d(1));
  EXPECT_FLOAT_EQ(expected_output(2), output_d(2));
  EXPECT_FLOAT_EQ(expected_output(3), output_d(3));
  EXPECT_FLOAT_EQ(expected_output(4), output_d(4));  

  row_vector_v output_v = add(rvv_1, rvd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val());  

  output_v = add(rvd_1, rvv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val());  

  output_v = add(rvv_1, rvv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_v(0).val());
  EXPECT_FLOAT_EQ(expected_output(1), output_v(1).val());
  EXPECT_FLOAT_EQ(expected_output(2), output_v(2).val());
  EXPECT_FLOAT_EQ(expected_output(3), output_v(3).val());
  EXPECT_FLOAT_EQ(expected_output(4), output_v(4).val());  
}
TEST(AgradRevMatrix, add_rowvector_rowvector_exception) {
  using stan::math::add;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  row_vector_d d1(5), d2(2);
  row_vector_v v1(5), v2(2);

  row_vector_v output;
  EXPECT_THROW( add(d1, d2), std::domain_error);
  EXPECT_THROW( add(d1, v2), std::domain_error);
  EXPECT_THROW( add(v1, d2), std::domain_error);
  EXPECT_THROW( add(v1, v2), std::domain_error);
}
TEST(AgradRevMatrix, add_matrix_matrix) {
  using stan::math::add;
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;

  matrix_d md_1(2,2), md_2(2,2);
  matrix_v mv_1(2,2), mv_2(2,2);

  md_1 << -10, 1, 10, 0;
  mv_1 << -10, 1, 10, 0;
  md_2 << 10, -10, 1, 2;
  mv_2 << 10, -10, 1, 2;
  
  matrix_d expected_output(2,2);
  expected_output << 0, -9, 11, 2;
  
  matrix_d output_d = add(md_1, md_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_d(0,0));
  EXPECT_FLOAT_EQ(expected_output(0,1), output_d(0,1));
  EXPECT_FLOAT_EQ(expected_output(1,0), output_d(1,0));
  EXPECT_FLOAT_EQ(expected_output(1,1), output_d(1,1));

  matrix_v output_v = add(mv_1, md_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_v(0,0).val());
  EXPECT_FLOAT_EQ(expected_output(0,1), output_v(0,1).val());
  EXPECT_FLOAT_EQ(expected_output(1,0), output_v(1,0).val());
  EXPECT_FLOAT_EQ(expected_output(1,1), output_v(1,1).val());

  output_v = add(md_1, mv_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_v(0,0).val());
  EXPECT_FLOAT_EQ(expected_output(0,1), output_v(0,1).val());
  EXPECT_FLOAT_EQ(expected_output(1,0), output_v(1,0).val());
  EXPECT_FLOAT_EQ(expected_output(1,1), output_v(1,1).val());

  output_v = add(mv_1, mv_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_v(0,0).val());
  EXPECT_FLOAT_EQ(expected_output(0,1), output_v(0,1).val());
  EXPECT_FLOAT_EQ(expected_output(1,0), output_v(1,0).val());
  EXPECT_FLOAT_EQ(expected_output(1,1), output_v(1,1).val());
}
TEST(AgradRevMatrix, add_matrix_matrix_exception) {
  using stan::math::add;
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  
  matrix_d d1(2,2), d2(1,2);
  matrix_v v1(2,2), v2(1,2);

  EXPECT_THROW(add(d1, d2), std::domain_error);
  EXPECT_THROW(add(d1, v2), std::domain_error);
  EXPECT_THROW(add(v1, d2), std::domain_error);
  EXPECT_THROW(add(v1, v2), std::domain_error);
}

TEST(AgradRevMatrix, add_nan_mat) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::math::add;

  matrix_d mat_a(2,2);
  matrix_d mat_b(2,2);
  matrix_v mat_a_v(2,2);
  matrix_v mat_b_v(2,2);
  double nan = std::numeric_limits<double>::quiet_NaN();

  mat_a << nan,2,3,4;
  mat_b << 5,6,nan,8;
  mat_a_v << nan,2,3,4;
  mat_b_v << 5,6,nan,8;

  matrix_v res1 = add(mat_a, mat_b_v);
  matrix_v res2 = add(mat_a_v, mat_b);
  matrix_v res3 = add(mat_a_v, mat_b_v);

  EXPECT_PRED1(isnan<double>, res1(0,0).val());
  EXPECT_FLOAT_EQ(8.0, res1(0,1).val());
  EXPECT_PRED1(isnan<double>, res1(1,0).val());
  EXPECT_FLOAT_EQ(12.0, res1(1,1).val());

  EXPECT_PRED1(isnan<double>, res2(0,0).val());
  EXPECT_FLOAT_EQ(8.0, res2(0,1).val());
  EXPECT_PRED1(isnan<double>, res2(1,0).val());
  EXPECT_FLOAT_EQ(12.0, res2(1,1).val());

  EXPECT_PRED1(isnan<double>, res3(0,0).val());
  EXPECT_FLOAT_EQ(8.0, res3(0,1).val());
  EXPECT_PRED1(isnan<double>, res3(1,0).val());
  EXPECT_FLOAT_EQ(12.0, res3(1,1).val());
}

TEST(AgradRevMatrix, add_nan_vec) {
  using stan::math::vector_d;
  using stan::agrad::vector_v;
  using stan::math::add;

  vector_d vec_a(2);
  vector_d vec_b(2);
  vector_v vec_a_v(2);
  vector_v vec_b_v(2);
  double nan = std::numeric_limits<double>::quiet_NaN();

  vec_a << nan,2;
  vec_b << 5,6;
  vec_a_v << nan,2;
  vec_b_v << 5,6;

  vector_v res1 = add(vec_a, vec_b_v);
  vector_v res2 = add(vec_a_v, vec_b);
  vector_v res3 = add(vec_a_v, vec_b_v);

  EXPECT_PRED1(isnan<double>, res1(0).val());
  EXPECT_FLOAT_EQ(8.0, res1(1).val());

  EXPECT_PRED1(isnan<double>, res2(0).val());
  EXPECT_FLOAT_EQ(8.0, res2(1).val());

  EXPECT_PRED1(isnan<double>, res3(0).val());
  EXPECT_FLOAT_EQ(8.0, res3(1).val());
}

TEST(AgradRevMatrix, add_nan_row_vec) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;
  using stan::math::add;

  row_vector_d row_veca(2);
  row_vector_d row_vecb(2);
  row_vector_v row_veca_v(2);
  row_vector_v row_vecb_v(2);
  double nan = std::numeric_limits<double>::quiet_NaN();

  row_veca << nan,2;
  row_vecb << 5,6;
  row_veca_v << nan,2;
  row_vecb_v << 5,6;

  row_vector_v res1 = add(row_veca, row_vecb_v);
  row_vector_v res2 = add(row_veca_v, row_vecb);
  row_vector_v res3 = add(row_veca_v, row_vecb_v);

  EXPECT_PRED1(isnan<double>, res1(0).val());
  EXPECT_FLOAT_EQ(8.0, res1(1).val());

  EXPECT_PRED1(isnan<double>, res2(0).val());
  EXPECT_FLOAT_EQ(8.0, res2(1).val());

  EXPECT_PRED1(isnan<double>, res3(0).val());
  EXPECT_FLOAT_EQ(8.0, res3(1).val());
}


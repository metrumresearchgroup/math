#include <stan/math/prim/mat.hpp>
#include <gtest/gtest.h>

TEST(MetaTraits, VectorBuilder_false_false) {
  using stan::VectorBuilder;
  using stan::length;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  Matrix<double,Dynamic,1> a_vector(4);
  Matrix<double,1,Dynamic> a_row_vector(5);

  VectorBuilder<false,double,double> dvv3(length(a_vector));
  EXPECT_THROW(dvv3[0], std::logic_error);
  
  VectorBuilder<false,double,double> dvv4(length(a_row_vector));
  EXPECT_THROW(dvv4[0], std::logic_error);
}

TEST(MetaTraits, VectorBuilder_true_false) {
  using stan::VectorBuilder;
  using stan::length;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  Matrix<double,Dynamic,1> a_vector(4);
  Matrix<double,1,Dynamic> a_row_vector(5);

  VectorBuilder<true,double,double> dvv3(length(a_vector));
  EXPECT_FLOAT_EQ(0.0, dvv3[0]);
  EXPECT_FLOAT_EQ(0.0, dvv3[1]);
  EXPECT_FLOAT_EQ(0.0, dvv3[2]);  
  
  VectorBuilder<true,double,double> dvv4(length(a_row_vector));
  EXPECT_FLOAT_EQ(0.0, dvv4[0]);
  EXPECT_FLOAT_EQ(0.0, dvv4[1]);
  EXPECT_FLOAT_EQ(0.0, dvv4[2]);
}

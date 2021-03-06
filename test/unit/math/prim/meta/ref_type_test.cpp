#include <stan/math/prim/meta.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

TEST(MathMetaPrim, ref_type_non_eigen) {
  using stan::ref_type_t;
  std::vector<int> a{1, 2, 3};
  ref_type_t<std::vector<int>> a_ref1 = a;
  ref_type_t<std::vector<int>&> a_ref2 = a;
  ref_type_t<std::vector<int>&&> a_ref3 = std::vector<int>{1, 2, 3};

  double b = 3;
  ref_type_t<double> b_ref1 = b;
  ref_type_t<double&> b_ref2 = b;
  ref_type_t<double&&> b_ref3 = 3;

  const std::vector<double> c{0.5, 4, 0.7};
  ref_type_t<const std::vector<double>> c_ref1 = c;
  ref_type_t<const std::vector<double>&> c_ref2 = c;

  EXPECT_STD_VECTOR_FLOAT_EQ(a_ref1, a);
  EXPECT_STD_VECTOR_FLOAT_EQ(a_ref2, a);
  EXPECT_STD_VECTOR_FLOAT_EQ(a_ref3, a);
  EXPECT_EQ(b_ref1, b);
  EXPECT_EQ(b_ref2, b);
  EXPECT_EQ(b_ref3, b);
  EXPECT_STD_VECTOR_FLOAT_EQ(c_ref1, c);
  EXPECT_STD_VECTOR_FLOAT_EQ(c_ref2, c);
  EXPECT_TRUE(std::is_lvalue_reference<ref_type_t<double>>::value);
  EXPECT_TRUE(std::is_lvalue_reference<ref_type_t<double&>>::value);
  EXPECT_FALSE(std::is_reference<ref_type_t<double&&>>::value);
  EXPECT_TRUE(
      std::is_lvalue_reference<ref_type_t<const std::vector<double>>>::value);
  EXPECT_TRUE(
      std::is_lvalue_reference<ref_type_t<const std::vector<double>&>>::value);
  EXPECT_FALSE(
      std::is_reference<ref_type_t<const std::vector<double>&&>>::value);
}

TEST(MathMetaPrim, ref_type_eigen_directly_accessible) {
  using stan::ref_type_t;
  Eigen::MatrixXd a(3, 3);
  a << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  Eigen::MatrixXd a2 = a;
  ref_type_t<Eigen::MatrixXd> a_ref1 = a;
  ref_type_t<Eigen::MatrixXd&> a_ref2 = a;
  ref_type_t<Eigen::MatrixXd&&> a_ref3 = std::move(a2);

  auto b = a.block(1, 0, 2, 2);
  ref_type_t<decltype(b)> b_ref1 = b;
  ref_type_t<decltype(b)&> b_ref2 = b;
  ref_type_t<decltype(b)&&> b_ref3 = a.block(1, 0, 2, 2);

  Eigen::Ref<Eigen::MatrixXd> c = a;
  Eigen::Ref<Eigen::MatrixXd> c2 = a;
  ref_type_t<Eigen::Ref<Eigen::MatrixXd>> c_ref1 = c;
  ref_type_t<Eigen::Ref<Eigen::MatrixXd>&> c_ref2 = c;
  ref_type_t<Eigen::Ref<Eigen::MatrixXd>&&> c_ref3 = std::move(c2);

  EXPECT_MATRIX_EQ(a_ref1, a);
  EXPECT_MATRIX_EQ(a_ref2, a);
  EXPECT_MATRIX_EQ(a_ref3, a);

  EXPECT_MATRIX_EQ(b_ref1, b);
  EXPECT_MATRIX_EQ(b_ref2, b);
  EXPECT_MATRIX_EQ(b_ref3, b);

  EXPECT_MATRIX_EQ(c_ref1, c);
  EXPECT_MATRIX_EQ(c_ref2, c);
  EXPECT_MATRIX_EQ(c_ref3, c);
  EXPECT_TRUE((std::is_same<decltype(a), ref_type_t<decltype(a)&&>>::value));
  EXPECT_TRUE((std::is_same<decltype(b), ref_type_t<decltype(b)&&>>::value));
  EXPECT_TRUE((std::is_same<decltype(c), ref_type_t<decltype(c)&&>>::value));
  EXPECT_TRUE(std::is_lvalue_reference<ref_type_t<Eigen::MatrixXd>>::value);
  EXPECT_TRUE(std::is_lvalue_reference<ref_type_t<Eigen::MatrixXd&>>::value);
  EXPECT_FALSE(std::is_reference<ref_type_t<Eigen::MatrixXd&&>>::value);
}

TEST(MathMetaPrim, ref_type_eigen_expression) {
  using stan::plain_type_t;
  using stan::ref_type_t;
  Eigen::MatrixXd m(3, 3);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  auto a = m * 3;
  ref_type_t<decltype(a)> a_ref1 = a;
  ref_type_t<decltype(a)&> a_ref2 = a;
  ref_type_t<decltype(a)&&> a_ref3 = m * 3;

  Eigen::MatrixXd a_eval = a;
  EXPECT_MATRIX_EQ(a_ref1, a_eval);
  EXPECT_MATRIX_EQ(a_ref2, a_eval);
  EXPECT_MATRIX_EQ(a_ref3, a_eval);

  EXPECT_TRUE((
      std::is_same<plain_type_t<decltype(a)>, ref_type_t<decltype(a)>>::value));
  EXPECT_TRUE((std::is_same<plain_type_t<decltype(a)>,
                            ref_type_t<decltype(a)&>>::value));
  EXPECT_TRUE((std::is_same<plain_type_t<decltype(a)>,
                            ref_type_t<decltype(a)&&>>::value));
}

TEST(MathMetaPrim, ref_type_if_test) {
  using stan::plain_type_t;
  using stan::ref_type_if_t;
  Eigen::MatrixXd m(3, 3);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  auto a = m * 3;

  EXPECT_TRUE(
      (std::is_same<plain_type_t<decltype(a)>,
                    std::decay_t<ref_type_if_t<true, decltype(a)&&>>>::value));
  EXPECT_TRUE(
      (std::is_same<decltype(a),
                    std::decay_t<ref_type_if_t<false, decltype(a)&&>>>::value));
}

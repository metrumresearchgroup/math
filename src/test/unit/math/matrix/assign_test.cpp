#include <stdexcept>
#include <vector>
#include <stan/math/matrix/assign.hpp>
#include <stan/math/matrix/get_base1_lhs.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>

TEST(MathMatrixAssign,int) {
  using stan::math::assign;
  int a;
  int b = 5;
  assign(a,b);
  EXPECT_EQ(5,a);
  EXPECT_EQ(5,b);

  assign(a,12);
  EXPECT_EQ(a,12);
}
TEST(MathMatrixAssign,double) {
  using stan::math::assign;

  double a;
  int b = 5;
  double c = 5.0;
  assign(a,b);
  EXPECT_FLOAT_EQ(5.0,a);
  EXPECT_FLOAT_EQ(5.0,b);

  assign(a,c);
  EXPECT_FLOAT_EQ(5.0,a);
  EXPECT_FLOAT_EQ(5.0,b);
  
  assign(a,5.2);
  EXPECT_FLOAT_EQ(5.2,a);
}
TEST(MathMatrixAssign,vectorDouble) {
  using stan::math::assign;
  using std::vector;
  
  vector<double> y(3);
  y[0] = 1.2;
  y[1] = 100;
  y[2] = -5.1;

  vector<double> x(3);
  assign(x,y);
  EXPECT_EQ(3U,x.size());
  EXPECT_EQ(3U,y.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(y[i],x[i]);

  vector<double> z(2);
  EXPECT_THROW(assign(x,z), std::domain_error);

  vector<int> ns(3);
  ns[0] = 1;
  ns[1] = -10;
  ns[2] = 500;

  assign(x,ns);
  EXPECT_EQ(3U,x.size());
  EXPECT_EQ(3U,ns.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(ns[i], x[i]);
}



TEST(MathMatrixAssign,eigenRowVectorDoubleToDouble) {
  using stan::math::assign;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<double,1,Dynamic> y(3);
  y[0] = 1.2;
  y[1] = 100;
  y[2] = -5.1;

  Matrix<double,1,Dynamic> x(3);
  assign(x,y);
  EXPECT_EQ(3,x.size());
  EXPECT_EQ(3,y.size());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(y[i],x[i]);
}
TEST(MathMatrixAssign,eigenRowVectorIntToDouble) {
  using stan::math::assign;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<double,1,Dynamic> x(3);
  x[0] = 1.2;
  x[1] = 100;
  x[2] = -5.1;

  Matrix<int,1,Dynamic> ns(3);
  ns[0] = 1;
  ns[1] = -10;
  ns[2] = 500;

  assign(x,ns);
  EXPECT_EQ(3,x.size());
  EXPECT_EQ(3,ns.size());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(ns[i], x[i]);
}
TEST(MathMatrixAssign,eigenRowVectorShapeMismatch) {
  using stan::math::assign;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<double,1,Dynamic> x(3);
  x[0] = 1.2;
  x[1] = 100;
  x[2] = -5.1;

  Matrix<double,1,Dynamic> z(2);
  EXPECT_THROW(assign(x,z), std::domain_error);

  Matrix<double,Dynamic,1> zz(3);
  zz << 1, 2, 3;
  EXPECT_THROW(assign(x,zz),std::domain_error);
  
  Matrix<double,Dynamic,Dynamic> zzz(3,1);
  zzz << 1, 2, 3;
  EXPECT_THROW(assign(x,zzz),std::domain_error);

  Matrix<double,Dynamic,Dynamic> zzzz(1,3);
  EXPECT_THROW(assign(x,zzzz),std::domain_error);
}


TEST(MathMatrixAssign,eigenMatrixDoubleToDouble) {
  using stan::math::assign;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<double,Dynamic,Dynamic> y(3,2);
  y << 1.2, 100, -5.1, 12, 1000, -5100;

  Matrix<double,Dynamic,Dynamic> x(3,2);
  assign(x,y);
  EXPECT_EQ(6,x.size());
  EXPECT_EQ(6,y.size());
  EXPECT_EQ(3,x.rows());
  EXPECT_EQ(3,y.rows());
  EXPECT_EQ(2,x.cols());
  EXPECT_EQ(2,y.cols());
  for (size_t i = 0; i < 6; ++i)
    EXPECT_FLOAT_EQ(y(i),x(i));
}
TEST(MathMatrixAssign,eigenMatrixIntToDouble) {
  using stan::math::assign;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<int,Dynamic,Dynamic> y(3,2);
  y << 1, 2, 3, 4, 5, 6;

  Matrix<double,Dynamic,Dynamic> x(3,2);
  assign(x,y);
  EXPECT_EQ(6,x.size());
  EXPECT_EQ(6,y.size());
  EXPECT_EQ(3,x.rows());
  EXPECT_EQ(3,y.rows());
  EXPECT_EQ(2,x.cols());
  EXPECT_EQ(2,y.cols());
  for (size_t i = 0; i < 6; ++i)
    EXPECT_FLOAT_EQ(y(i),x(i));
}
TEST(MathMatrixAssign,eigenMatrixShapeMismatch) {
  using stan::math::assign;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<double,Dynamic,Dynamic> x(2,3);
  x << 1, 2, 3, 4, 5, 6;

  Matrix<double,1,Dynamic> z(6);
  z << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(assign(x,z), std::domain_error);
  EXPECT_THROW(assign(z,x), std::domain_error);

  Matrix<double,Dynamic,1> zz(6);
  zz << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(assign(x,zz),std::domain_error);
  EXPECT_THROW(assign(zz,x),std::domain_error);
  
  Matrix<double,Dynamic,Dynamic> zzz(6,1);
  zzz << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(assign(x,zzz),std::domain_error);
  EXPECT_THROW(assign(zzz,x),std::domain_error);

}

TEST(MathMatrix,block) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1_lhs;
  using stan::math::assign;

  Matrix<double,Dynamic,Dynamic> m(2,3);
  m << 1, 2, 3, 4, 5, 6;
  
  Matrix<double,1,Dynamic> rv(3);
  rv << 10, 100, 1000;
  
  assign(get_base1_lhs(m,1,"m",1),rv);  
  EXPECT_FLOAT_EQ(10.0, m(0,0));
  EXPECT_FLOAT_EQ(100.0, m(0,1));
  EXPECT_FLOAT_EQ(1000.0, m(0,2));
}


TEST(MathMatrix,vectorVector) {
  using std::vector;
  using stan::math::assign;
  vector<vector<double> > x(3,vector<double>(2));
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 2; ++j)
      x[i][j] = (i + 1) * (j - 10);
  
  vector<vector<double> > y(3,vector<double>(2));

  assign(y,x);
  EXPECT_EQ(3U,y.size());
  for (size_t i = 0; i < 3U; ++i) {
    EXPECT_EQ(2U,y[i].size());
    for (size_t j = 0; j < 2U; ++j) {
      EXPECT_FLOAT_EQ(x[i][j],y[i][j]);
    }
  }
}


TEST(MathMatrix,vectorVectorVector) {
  using std::vector;
  using stan::math::assign;
  vector<vector<vector<double> > > 
    x(4,vector<vector<double> >(3,vector<double>(2)));
  for (size_t k = 0; k < 4; ++k)
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 2; ++j)
        x[k][i][j] = (i + 1) * (j - 10) * (20 * k + 100);
  
  vector<vector<vector<double> > > 
    y(4,vector<vector<double> >(3,vector<double>(2)));

  assign(y,x);
  EXPECT_EQ(4U,y.size());
  for (size_t k = 0; k < 4U; ++k) {
    EXPECT_EQ(3U,y[k].size());
    for (size_t i = 0; i < 3U; ++i) {
      EXPECT_EQ(2U,y[k][i].size());
      for (size_t j = 0; j < 2U; ++j) {
        EXPECT_FLOAT_EQ(x[k][i][j],y[k][i][j]);
      }
    }
  }
}

TEST(MathMatrix,vectorEigenVector) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::assign;

  vector<Matrix<double,Dynamic,1> > x(2, Matrix<double,Dynamic,1>(3));
  for (size_t i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      x[i](j) = (i + 1) * (10 * j + 2);
  vector<Matrix<double,Dynamic,1> > y(2, Matrix<double,Dynamic,1>(3));

  assign(y,x);

  EXPECT_EQ(2U,y.size());
  for (size_t i = 0; i < 2U; ++i) {
    EXPECT_EQ(3U,y[i].size());
    for (size_t j = 0; j < 3U; ++j) {
      EXPECT_FLOAT_EQ(x[i](j), y[i](j));
    }
  }
}

TEST(MathMatrix,getAssignRow) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::get_base1_lhs;
  using stan::math::assign;

  Matrix<double,Dynamic,Dynamic> m(2,3);
  m << 1, 2, 3, 4, 5, 6;
  
  Matrix<double,1,Dynamic> rv(3);
  rv << 10, 100, 1000;
  
  assign(get_base1_lhs(m,1,"m",1),rv);  
  EXPECT_FLOAT_EQ(10.0, m(0,0));
  EXPECT_FLOAT_EQ(100.0, m(0,1));
  EXPECT_FLOAT_EQ(1000.0, m(0,2));
}


TEST(MathMatrix, assign_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(2,2);
  m1 << 1, nan,
        3, 6;
        
  Eigen::MatrixXd m2(2,2);
  m2 << 1, 100,
        nan, 4.9;
        
  Eigen::MatrixXd m3(2,2);
  m3 << 10.1, 100,
        1, 0;
        
  Eigen::VectorXd v1(3);
  v1 << 10.1, nan, 1.1;
        
  Eigen::RowVectorXd v2(3);
  v2 << 0.1, 4.6, nan;
        
  Eigen::MatrixXd mr;
  Eigen::VectorXd vr;
  Eigen::RowVectorXd rvr;

  using stan::math::assign;
  using boost::math::isnan;
  
  mr = Eigen::MatrixXd::Zero(2, 2);
  assign(mr, m1);
  expect_matrix_eq_or_both_nan(mr, m1);
  
  mr = Eigen::MatrixXd::Zero(2, 2);
  assign(mr, m2);
  expect_matrix_eq_or_both_nan(mr, m2);

  mr = Eigen::MatrixXd::Zero(2, 2);  
  assign(mr, m3);
  expect_matrix_eq_or_both_nan(mr, m3);

  vr = Eigen::VectorXd::Zero(3);  
  assign(vr, v1);
  expect_matrix_eq_or_both_nan(vr, v1);
  
  rvr = Eigen::RowVectorXd::Zero(3);  
  assign(rvr, v2);
  expect_matrix_eq_or_both_nan(rvr, v2);

  std::vector<double> stdvec(14, 1.1);
  v2[7] = nan;
  
  std::vector<double> stdvecr(14);
  assign(stdvecr, stdvec);
  expect_matrix_eq_or_both_nan(stdvecr, stdvec);
}

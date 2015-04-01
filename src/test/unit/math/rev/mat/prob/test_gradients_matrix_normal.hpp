#include <cmath>
#include <vector>
#include <iomanip>
#include <stdexcept>

std::vector<stan::agrad::var> get_vvar(std::vector<double> vd) {
  size_t size = vd.size();
  std::vector<stan::agrad::var> vv;
  vv.reserve(size);
  for (size_t i = 0; i < size; i++)
    vv.push_back(vd[i]);
  return vv;
}

std::vector<double> vdouble_from_vvar(std::vector<stan::agrad::var> vv) {
  size_t size = vv.size();
  std::vector<double> vd;
  vd.reserve(size);
  for (size_t i = 0; i < size; i++)
    vd.push_back(vv[i].val());
  return vd;
}

std::vector<double> vdouble_from_vvar(std::vector<double> vv) {
  return vv;
}

struct matrix_normal_fun {
  const int rows_;
  const int cols_;

  matrix_normal_fun(int rows, int cols) : rows_(rows), cols_(cols) { }

  template <typename T_y, typename T_mu, typename T_sigma, typename T_D>
  typename boost::math::tools::promote_args<T_y, T_mu, T_sigma, T_D>::type
  operator() (const std::vector<T_y>& y_vec,
              const std::vector<T_mu>& mu_vec,
              const std::vector<T_sigma>& sigma_vec,
              const std::vector<T_D>& D_vec) const {
    Eigen::Matrix<T_y,-1,-1> y(rows_,cols_);
    Eigen::Matrix<T_mu,-1,-1> mu(rows_,cols_);
    Eigen::Matrix<T_sigma,-1,-1> Sigma(cols_, cols_);
    Eigen::Matrix<T_D,-1,-1> D(rows_, rows_);

    size_t pos = 0;
    for (int k = 0; k < cols_; ++k) 
      for (int l = 0; l < rows_; ++l)
        y(l,k) = y_vec[pos++];

    pos = 0;        
    for (int k = 0; k < cols_; ++k) 
      for (int l = 0; l < rows_; ++l)
        mu(l,k) = mu_vec[pos++];
    
    pos = 0;
    for (int i = 0; i < cols_; ++i) {
      for (int j = 0; j <= i; ++j) {
        Sigma(j,i) = sigma_vec[pos++];
        Sigma(i,j) = Sigma(j,i);
      }
    }

    pos = 0;
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j <= i; ++j) {
        D(j,i) = D_vec[pos++];
        D(i,j) = D(j,i);
      }
    }
    
    return stan::prob::matrix_normal_prec_log<false>(y, mu, D, Sigma);
  }
};
  

template <typename F, typename T_y, typename T_mu, typename T_sigma, typename T_D>
std::vector<double> 
finite_diffs_matrix_normal(const F& fun,
             const std::vector<T_y>& vec_y,
             const std::vector<T_mu>& vec_mu,
             const std::vector<T_sigma>& vec_sigma,
             const std::vector<T_D>& vec_D,
             double epsilon = 1e-6) {
  std::vector<double> diffs;
  diffs.reserve(vec_y.size() + vec_mu.size() + vec_sigma.size() + vec_D.size());

  std::vector<double> vec_y_plus = vdouble_from_vvar(vec_y);
  std::vector<double> vec_y_minus = vec_y_plus;
  std::vector<double> vec_mu_plus = vdouble_from_vvar(vec_mu);
  std::vector<double> vec_mu_minus = vec_mu_plus;  
  std::vector<double> vec_sigma_plus = vdouble_from_vvar(vec_sigma);
  std::vector<double> vec_sigma_minus = vec_sigma_plus;  
  std::vector<double> vec_D_plus = vdouble_from_vvar(vec_D);
  std::vector<double> vec_D_minus = vec_D_plus;  
    
  if (!stan::is_constant<T_y>::value) {
    for (size_t i = 0; i < vec_y.size(); ++i) {
      double recover_vec_y_plus = vec_y_plus[i];
      double recover_vec_y_minus = vec_y_minus[i];
      vec_y_plus[i] += epsilon;
      vec_y_minus[i] -= epsilon;
      diffs.push_back((fun(vec_y_plus,vec_mu_plus,vec_sigma_plus,vec_D_plus) -
                      fun(vec_y_minus,vec_mu_minus,vec_sigma_minus,vec_D_minus)) /
                      (2 * epsilon));
      vec_y_plus[i] = recover_vec_y_plus;
      vec_y_minus[i] = recover_vec_y_minus;
    }
  }
  if (!stan::is_constant<T_mu>::value) {
    for (size_t i = 0; i < vec_mu.size(); ++i) {
      double recover_vec_mu_plus = vec_mu_plus[i];
      double recover_vec_mu_minus = vec_mu_minus[i];
      vec_mu_plus[i] += epsilon;
      vec_mu_minus[i] -= epsilon;
      diffs.push_back((fun(vec_y_plus,vec_mu_plus,vec_sigma_plus,vec_D_plus) -
                      fun(vec_y_minus,vec_mu_minus,vec_sigma_minus,vec_D_minus)) /
                      (2 * epsilon));
      vec_mu_plus[i] = recover_vec_mu_plus;
      vec_mu_minus[i] = recover_vec_mu_minus;
    }
  }
  if (!stan::is_constant<T_sigma>::value) {
    for (size_t i = 0; i < vec_sigma.size(); ++i) {
      double recover_vec_sigma_plus = vec_sigma_plus[i];
      double recover_vec_sigma_minus = vec_sigma_minus[i];
      vec_sigma_plus[i] += epsilon;
      vec_sigma_minus[i] -= epsilon;
      diffs.push_back((fun(vec_y_plus,vec_mu_plus,vec_sigma_plus,vec_D_plus) -
                      fun(vec_y_minus,vec_mu_minus,vec_sigma_minus,vec_D_minus)) /
                      (2 * epsilon));
      vec_sigma_plus[i] = recover_vec_sigma_plus;
      vec_sigma_minus[i] = recover_vec_sigma_minus;
    }
  }
  if (!stan::is_constant<T_D>::value) {
    for (size_t i = 0; i < vec_D.size(); ++i) {
      double recover_vec_D_plus = vec_D_plus[i];
      double recover_vec_D_minus = vec_D_minus[i];
      vec_D_plus[i] += epsilon;
      vec_D_minus[i] -= epsilon;
      diffs.push_back((fun(vec_y_plus,vec_mu_plus,vec_sigma_plus,vec_D_plus) -
                      fun(vec_y_minus,vec_mu_minus,vec_sigma_minus,vec_D_minus)) /
                      (2 * epsilon));
      vec_D_plus[i] = recover_vec_D_plus;
      vec_D_minus[i] = recover_vec_D_minus;
    }
  }
  return diffs;
}

template <typename F, typename T_y, typename T_mu, typename T_sigma, typename T_D>
std::vector<double>
grad_matrix_normal(const F& fun,
     const std::vector<T_y>& vec_y,
     const std::vector<T_mu>& vec_mu,
     const std::vector<T_sigma>& vec_sigma,
     const std::vector<T_D>& vec_D) {

  stan::agrad::start_nested();
  stan::agrad::var fx = fun(vec_y, vec_mu, vec_sigma, vec_D);
  std::vector<double> grad;
  std::vector<stan::agrad::var> vec_vars;
  if (!stan::is_constant<T_y>::value) {
    for (size_t i = 0; i < vec_y.size(); i++){
      vec_vars.push_back(vec_y[i]);
    }
  }
  if (!stan::is_constant<T_mu>::value) {
    for (size_t i = 0; i < vec_mu.size(); i++)
      vec_vars.push_back(vec_mu[i]);
  }
  if (!stan::is_constant<T_sigma>::value) {
    for (size_t i = 0; i < vec_sigma.size(); i++)
      vec_vars.push_back(vec_sigma[i]);
  }
  if (!stan::is_constant<T_D>::value) {
    for (size_t i = 0; i < vec_D.size(); i++)
      vec_vars.push_back(vec_D[i]);
  }
  fx.grad(vec_vars,grad);
  stan::agrad::recover_memory_nested();
  return grad;
}

template <typename F, typename T_y, typename T_mu, typename T_sigma, typename T_D>
void test_grad_matrix_normal(const F& fun,
                             const std::vector<T_y>& vec_y,
                             const std::vector<T_mu>& vec_mu,
                             const std::vector<T_sigma>& vec_sigma,
                             const std::vector<T_D>& vec_D) {
  using std::fabs;
  std::vector<double> diffs_finite = finite_diffs_matrix_normal(fun,vec_y,vec_mu,vec_sigma,vec_D);
  std::vector<double> diffs_var = grad_matrix_normal(fun,vec_y,vec_mu,vec_sigma,vec_D);
  EXPECT_EQ(diffs_finite.size(), diffs_var.size());
  for (size_t i = 0; i < diffs_finite.size(); ++i) {
    double tolerance = 1e-6 * fmax(fabs(diffs_finite[i]), fabs(diffs_var[i])) + 1e-14;
    EXPECT_NEAR(diffs_finite[i], diffs_var[i], tolerance);
  }
}

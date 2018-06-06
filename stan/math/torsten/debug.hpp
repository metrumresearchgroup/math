#ifndef STAN_MATH_TORSTEN_DEBUG_HPP
#define STAN_MATH_TORSTEN_DEBUG_HPP

// initialize some templates for debugging

#ifdef DEBUG
template class std::vector<double>;
template class std::vector<stan::math::var>;
// template class Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
// template class Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>;

template class std::vector<std::vector<double>> ;
template class std::vector<std::vector<stan::math::var>> ;
// template class std::vector<
//   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>;
// template class std::vector<
//   Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>>;

namespace torsten {
  template<typename T, typename D>
  void debug_grad(T& f, std::vector<D> & x, std::vector<double>& g) {}
  template<typename T, typename D>
  void debug_grad(T& f, D & x, std::vector<double>& g) {}

  void debug_grad(stan::math::var& f,
                  std::vector<stan::math::var>& x,
                  std::vector<double>& g) {
    f.grad(x, g);
    stan::math::set_zero_all_adjoints();
  }

  void debug_grad(stan::math::var& f,
                  stan::math::var& x,
                  std::vector<double>& g) {
    std::vector<stan::math::var> xv{x};
    f.grad(xv, g);
    stan::math::set_zero_all_adjoints();
  }  
}

#endif

#endif

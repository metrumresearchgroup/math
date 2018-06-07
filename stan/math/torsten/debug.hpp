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
  void debug_grad(const T& f, const std::vector<D> & x, std::vector<double>& g) {
    std::cout << "N/A" << "\n";
  }
  template<typename T, typename D>
  void debug_grad(const T& f, const D& x, std::vector<double>& g) {
    std::cout << "N/A" << "\n";
  }

  void debug_grad(const stan::math::var& f,
                  const std::vector<stan::math::var>& x,
                  std::vector<double>& g) {
    stan::math::var ff = f;
    std::vector<stan::math::var> xx {x};
    ff.grad(xx, g);
    stan::math::set_zero_all_adjoints();
    Eigen::Matrix<double, 1, Eigen::Dynamic> gm =
      stan::math::to_matrix(g, 1, g.size());
    std::cout << gm << "\n";
  }

  void debug_grad(const stan::math::var& f,
                  const stan::math::var& x,
                  std::vector<double>& g) {
    stan::math::var ff = f;
    std::vector<stan::math::var> xv{x};
    ff.grad(xv, g);
    stan::math::set_zero_all_adjoints();
    std::cout << g[0] << "\n";
  }  
}

#ifndef TORSTEN_DEBUG_DIFF
#define TORSTEN_DEBUG_DIFF(F, X, G)             \
  std::cout << "TORSTEN DEBUG: ";               \
  std::cout << std::setw(35);                   \
  std::cout << "d[ "#F" ] / d[ "#X" ] ";        \
  std::cout << std::setw(5);                    \
  std::cout << " = ";                           \
  debug_grad(F, X, G);
#endif

#endif

#endif

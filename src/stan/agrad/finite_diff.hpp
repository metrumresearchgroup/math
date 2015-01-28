#ifndef STAN__AGRAD__FINITE_DIFF_HPP
#define STAN__AGRAD__FINITE_DIFF_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/math/functions.hpp>
#include <stan/agrad/autodiff.hpp>

namespace stan {
  
  namespace agrad {

    /**
     * Calculate the value and the gradient of the specified function
     * at the specified argument using finite difference.  
     *
     * <p>The functor must implement 
     * 
     * <code>
     * T
     * operator()(const
     * Eigen::Matrix<T,Eigen::Dynamic,1>&)
     * </code>
     *
     * Where <code>T</code> is of type
     * stan::agrad::var, stan::agrad::fvar<double>,
     * stan::agrad::fvar<stan::agard::var> or 
     * stan::agrad::fvar<stan::agrad::fvar<stan::agrad::var> >
     * 
     * @tparam F Type of function
     * @param[in] f Function
     * @param[in] x Argument to function
     * @param[out] fx Function applied to argument
     * @param[out] grad_fx Gradient of function at argument
     */
    template <typename F>
    void
    finite_diff_gradient(const F& f,
                         const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
                         double& fx,
                         Eigen::Matrix<double,Eigen::Dynamic,1>& grad_fx, 
                         const double epsilon = 1e-6) {
      Eigen::Matrix<double,Eigen::Dynamic,1> x_temp(x);

      size_type d = x.size();
      grad_fx.resize(d);

      fx = f(x);
      
      for (size_type i = 0; i < d; ++i){
        double delta_f = 0.0;

        x_temp(i) += epsilon;
        delta_f = f(x_temp);

        x_temp(i) = x(i) - epsilon;
        delta_f -= f(x_temp);
        
        x_temp(i) = x(i);

        delta_f /= 2.0 * epsilon;
        grad_fx(i) = delta_f;
      }

    }

    /**
     * Calculate the value and the hessian of the specified function
     * at the specified argument using finite difference.  
     *
     * <p>The functor must implement 
     * 
     * <code>
     * stan::agrad::var
     * operator()(const
     * std::vector<T>&)
     * </code>
     *
     * Where <code>T</code> is of type
     * stan::agrad::var, stan::agrad::fvar<double>,
     * stan::agrad::fvar<stan::agard::var> or 
     * stan::agrad::fvar<stan::agrad::fvar<stan::agrad::var> >
     * 
     * @tparam F Type of function
     * @param[in] f Function
     * @param[in] x Argument to function
     * @param[out] fx Function applied to argument
     * @param[out] grad_fx Gradient of function at argument
     */
    template <typename F>
    void
    finite_diff_hessian(const F& f,
                        const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
                        double& fx,
                        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& hess_fx, 
                        const double epsilon = 1e-6) {

      size_type d = x.size();

      Eigen::Matrix<double,Eigen::Dynamic,1> x_temp(x);
      Eigen::Matrix<double,Eigen::Dynamic,1> g_auto(d);
      hess_fx.resize(d, d);

      fx = f(x);
      
      for (size_type i = 0; i < d; ++i){
        Eigen::VectorXd g_diff = Eigen::VectorXd::Zero(d);
        x_temp(i) += epsilon;
        gradient(f, x_temp, fx, g_auto);
        g_diff = g_auto;

        x_temp(i) = x(i) - epsilon;
        gradient(f, x_temp, fx, g_auto);
        g_diff -= g_auto;

        x_temp(i) = x(i);
        g_diff /= 2.0 * epsilon;
        
        hess_fx.col(i) = g_diff;
      }

    }
    
    template <typename F>
    void
    finite_diff_grad_hessian(const F& f,
                             const Eigen::Matrix<double,Eigen::Dynamic,1>& x,
                             double& fx,
                             std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >& grad_hess_fx, 
                             const double epsilon = 1e-6) {

      size_type d = x.size();

      Eigen::Matrix<double,Eigen::Dynamic,1> x_temp(x);
      Eigen::Matrix<double,Eigen::Dynamic,1> grad_auto(d);
      Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> H_auto(d,d);
      Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> H_diff(d,d);

      fx = f(x);
      
      for (size_type i = 0; i < d; ++i){
        H_diff.setZero();

        x_temp(i) += epsilon;
        hessian(f, x_temp, fx, grad_auto, H_auto);
        H_diff = H_auto;

        x_temp(i) = x(i) - epsilon;
        hessian(f, x_temp, fx, grad_auto, H_auto);
        H_diff -= H_auto;

        x_temp(i) = x(i);
        H_diff /= 2.0 * epsilon;
        
        grad_hess_fx.push_back(H_diff);
      }

    }

  }
}
#endif

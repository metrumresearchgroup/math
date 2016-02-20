#ifndef STAN_MATH_PRIM_MAT_PROB_ORDERED_LOGISTIC_LOG_HPP
#define STAN_MATH_PRIM_MAT_PROB_ORDERED_LOGISTIC_LOG_HPP

#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/fun/inv_logit.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/fun/log1m_exp.hpp>
#include <stan/math/prim/scal/fun/log1p_exp.hpp>
#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <stan/math/prim/scal/err/check_less_or_equal.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/mat/prob/categorical_rng.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>

namespace stan {

  namespace math {

    template <typename T1, typename T2, typename T3>
    size_t max_size_mixed(const T1& x1, const T2& x2, const T3& x3) {
      size_t result = length(x1);
      result = result > length(x2) ? result : length(x2);
      result = result > length_mvt(x3) ? result : length_mvt(x3);
      return result;
    }

    // y in 0, ..., K-1;   c.size()==K-2, c increasing,  lambda finite
    /**
     * Returns the (natural) log probability of the specified integer
     * outcome given the continuous location and specified cutpoints
     * in an ordered logistic model.
     *
     * <p>Typically the continous location
     * will be the dot product of a vector of regression coefficients
     * and a vector of predictors for the outcome.
     *
     * @tparam propto True if calculating up to a proportion.
     * @tparam T_loc Location type.
     * @tparam T_cut Cut-point type.
     * @param y Outcome.
     * @param lambda Location.
     * @param c Positive increasing vector of cutpoints.
     * @return Log probability of outcome given location and
     * cutpoints.

     * @throw std::domain_error If the outcome is not between 1 and
     * the number of cutpoints plus 2; if the cutpoint vector is
     * empty; if the cutpoint vector contains a non-positive,
     * non-finite value; or if the cutpoint vector is not sorted in
     * ascending order.
     */
    template <bool propto, typename T_lambda, typename T_cut>
    typename boost::math::tools::promote_args<T_lambda, T_cut>::type
    ordered_logistic_log(const T_y& y, const T_lambda& lambda,
                         const T_cut& c) {
      typedef typename return_type<T_lambda, T_cut>::type lp_type;
      typedef typename stan::partials_return_type<T_lambda,
                                                  T_cut>::type
        T_partials_return;
      lp_type lp(0.0);
      using std::exp;
      using std::log;
      using stan::math::inv_logit;
      using stan::math::log1m;
      using stan::math::log1p_exp;

      static const char* function("stan::math::ordered_logistic");

      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_nonnegative;
      using stan::math::check_less;
      using stan::math::check_less_or_equal;
      using stan::math::check_greater;
      using stan::math::check_bounded;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_lambda> lambda_vec(lambda);
      VectorViewMvt<const T_loc> c_vec(c);
      // size of std::vector of Eigen vectors
      size_t size_vec = max_size_mixed(y, lambda, c);

      // Check if every vector of the array has the same size
      int size_c = c_vec[0].size();
      if (size_vec > 1) {
        int size_c_old = size_c;
        int size_c_new;
        for (size_t i = 1, size_ = length_mvt(c); i < size_; i++) {
          int size_c_new = c_vec[i].size();
          check_size_match(function,
                           "Size of one of the vectors of "
                           "the cuts", size_c_new,
                           "Size of another vector of the "
                           "cuts", size_c_old);
          size_c_old = size_c_new;
        }
        (void) size_c_old;
        (void) size_c_new;
      }

      for (size_t i = 0; i < size_vec; i++) {
        check_bounded(function, "Random variable", y_vec[i], 1, K);
        check_finite(function, "Location parameter", lambda_vec[i]);
        check_greater(function, "Size of cut points parameter", c_vec[i].size(), 0);
        check_finite(function, "Cut points parameter", c_vec[i](c.size()-1));
        check_finite(function, "Cut points parameter", c_vec[i](0));
      }

      int K = size_c + 1;

      for (size_t i = 0; i < length_mvt(c); ++i) 
        for (int j = 1; j < size_c; ++j)
          check_greater(function, "Cut points parameter", c_vec[i](j), c(j - 1));

      operands_paritals_container<typename T_lambda, double,
                                  typename T_loc, Matrix<double, Dynamic, 1> > 
                                    opc(lambda, c);

      for (size_t i = 0; i < size_vec; ++i) {
        int y_el = y_vec[i];
        const T_partials_return d_lam = value_of(lambda_vec[i]);
        Matrix<double,Dynamic,1> cs(size_c,0.0);
        cs.setZero();

        // log(1 - inv_logit(lambda))
        if (y_el == 1) {
          d_c = value_of(c_vec[i](0));
          lp += -log1p_exp(d_lam - d_c);
          inv_log = 1./(1+exp(d_c-d_lam));
          opc.dx1[i] += -inv_log;
          cs(0) = inv_log;
          opc.dx2[i] += cs;
        }
        else if (y_el == K) {
          d_c = value_of(c_vec[i](K-2));
          inv_log = 1./(1+exp(d_lam-d_c));
          opc.dx1[i] += inv_log;
          cs(K-2) = -inv_log
          opc.dx2[i] += cs;
          lp += -log1p_exp(value_of(c_vec[i](K-2)) - d_lam);
        }
        else {
          d_cym2 = value_of(c_vec[i](y_el-2));
          d_cym1 = value_of(c_vec[i](y_el-1));
          lp += d_cym1 - lam_el
            -log1m_exp(d_cym2-d_cym1)-log1p_exp(d_cym2-d_lam)
            -log1p_exp(d_cym1-d_lam);
        }
      }
    }

    template <typename T_lambda, typename T_cut>
    typename boost::math::tools::promote_args<T_lambda, T_cut>::type
    ordered_logistic_log(int y, const T_lambda& lambda,
                         const Eigen::Matrix<T_cut, Eigen::Dynamic, 1>& c) {
      return ordered_logistic_log<false>(y, lambda, c);
    }
  }
}

#endif

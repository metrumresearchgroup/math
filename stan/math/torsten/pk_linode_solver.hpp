#ifndef PK_LINODE_SOLVER_HPP
#define PK_LINODE_SOLVER_HPP

namespace refactor {

  using boost::math::tools::promote_args;
  using Eigen::Matrix;
  using Eigen::Dynamic;


  class PKLinODEModelSolver {
  public:
    PKLinODEModelSolver() {}

    template<typename T_time, template <class, class... > class T_model, class... Ts_par>    
    Eigen::Matrix<typename T_model<Ts_par...>::scalar_type, Eigen::Dynamic, 1> 
    solve(const T_model<Ts_par...> &pkmodel, const T_time& dt) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::matrix_exp;
      using stan::math::mdivide_left;
      using stan::math::multiply;
      using scalar_type = typename T_model<Ts_par...>::scalar_type;

      auto init = pkmodel.y0()   ;
      auto rate = pkmodel.rate() ;

      if (dt == 0) { return init;
      } else {
        auto system = pkmodel.rhs_matrix();

        bool rate_zeros = true;
        for (size_t i = 0; i < rate.size(); i++)
          if (rate[i] != 0) rate_zeros = false;

        // trick to promote dt, and dt_system
        scalar_type dt_s = dt;

        if (rate_zeros) {
          Matrix<scalar_type, Dynamic, Dynamic> dt_system = multiply(dt_s, system);
          Matrix<scalar_type, Dynamic, 1> pred = matrix_exp(dt_system)
            * init.transpose();
          return pred.transpose();
        } else {
          int nCmt = system.cols();
          Matrix<scalar_type, Dynamic, 1> rate_vec(rate.size()), x(nCmt), x2(nCmt);
          for (size_t i = 0; i < rate.size(); i++) rate_vec(i) = rate[i];
          x = mdivide_left(system, rate_vec);
          x2 = x + init.transpose();
          Matrix<scalar_type, Dynamic, Dynamic> dt_system = multiply(dt_s, system);
          Matrix<scalar_type, Dynamic, 1> pred = matrix_exp(dt_system) * x2;
          pred -= x;
          return pred.transpose();
        }
      }
    }
  };

}

#endif

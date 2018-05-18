#ifndef PK_ONECPT_SOLVER_HPP
#define PK_ONECPT_SOLVER_HPP

namespace refactor {

  using boost::math::tools::promote_args;

  class PKOneCptModelSolver {
  public:
    PKOneCptModelSolver() {}

    template<typename T_time, template <class, class... > class T_model, class... Ts_par>    
    Eigen::Matrix<typename T_model<Ts_par...>::scalar_type, Eigen::Dynamic, 1> 
    solve(const T_model<Ts_par...> &pkmodel, const T_time& dt) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using scalar_type = typename T_model<Ts_par...>::scalar_type;

      auto init = pkmodel.y0()   ;
      auto rate = pkmodel.rate() ;
      auto ka   = pkmodel.ka()   ;
      auto alpha= pkmodel.alpha();

      std::vector<scalar_type> a(2, 0);
      Matrix<scalar_type, 1, Dynamic> pred = Matrix<scalar_type, 1, Dynamic>::Zero(2);

      if ((init[0] != 0) || (rate[0] != 0)) {
        pred(0, 0) = init[0] * exp(-ka * dt) + rate[0] * (1 - exp(-ka * dt)) / ka;
        a[0] = ka / (ka - alpha[0]);
        a[1] = -a[0];
        pred(0, 1) += torsten::PolyExp(dt, init[0], 0, 0, 0, false, a, alpha, 2) +
          torsten::PolyExp(dt, 0, rate[0], dt, 0, false, a, alpha, 2);
      }

      if ((init[1] != 0) || (rate[1] != 0)) {
        a[0] = 1;
        pred(0, 1) += torsten::PolyExp(dt, init[1], 0, 0, 0, false, a, alpha, 1) +
          torsten::PolyExp(dt, 0, rate[1], dt, 0, false, a, alpha, 1);
      }
      return pred;
    }
  };

}

#endif

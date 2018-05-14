#ifndef PK_LINODE_SOLVER_SS_HPP
#define PK_LINODE_SOLVER_SS_HPP

namespace refactor {

  using boost::math::tools::promote_args;
  using Eigen::Matrix;
  using Eigen::Dynamic;


  class PKLinODEModelSolverSS {
  public:
    PKLinODEModelSolverSS() {}

    template<typename T_amt, typename T_rate, 
             typename T_ii,
             template <class, class... > class T_model, class... Ts_par>
    Eigen::Matrix<typename T_model<Ts_par...>::scalar_type, Eigen::Dynamic, 1> 
    solve(T_model<Ts_par...> pkmodel,
          const T_amt& amt,
          const T_rate& rate,
          const T_ii& ii,
          const int& cmt) const {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::matrix_exp;
      using stan::math::mdivide_left;
      using stan::math::multiply;
      using std::vector;
      // using scalar_type = typename T_model<Ts_par...>::scalar_type;
      using par_type = typename T_model<Ts_par...>::par_type;

      typedef typename promote_args<T_ii, par_type>::type T0;
      typedef typename promote_args<T_amt, T_rate, T_ii,
                                    par_type>::type scalar;

      Matrix<par_type, Dynamic, Dynamic> system = pkmodel.rhs_matrix();
      int nCmt = system.rows();
      Matrix<T0, Dynamic, Dynamic> workMatrix;
      Matrix<T0, Dynamic, Dynamic> ii_system = multiply(ii, system);
      Matrix<scalar, 1, Dynamic> pred(nCmt);
      pred.setZero();
      Matrix<scalar, Dynamic, 1> amounts(nCmt);
      amounts.setZero();

      if (rate == 0) {  // bolus dose
        amounts(cmt - 1) = amt;
        workMatrix = - matrix_exp(ii_system);
        for (int i = 0; i < nCmt; i++) workMatrix(i, i) += 1;
        amounts = mdivide_left(workMatrix, amounts);
        pred = multiply(matrix_exp(ii_system), amounts);

      } else if (ii > 0) {  // multiple truncated infusions
        scalar delta = amt / rate;
        static const char* function("Steady State Event");
        torsten::check_mti(amt, delta, ii, function);

        amounts(cmt - 1) = rate;
        scalar t = delta;
        amounts = mdivide_left(system, amounts);
        Matrix<scalar, Dynamic, Dynamic> t_system = multiply(delta, system);
        pred = matrix_exp(t_system) * amounts;
        pred -= amounts;

        workMatrix = - matrix_exp(ii_system);
        for (int i = 0; i < nCmt; i++) workMatrix(i, i) += 1;

        Matrix<scalar, Dynamic, 1> pred_t = pred.transpose();
        pred_t = mdivide_left(workMatrix, pred_t);
        t = ii - t;
        t_system = multiply(t, system);
        pred_t = matrix_exp(t_system) * pred_t;
        pred = pred_t.transpose();

      } else {  // constant infusion
        amounts(cmt - 1) -= rate;
        pred = mdivide_left(system, amounts);
      }
      return pred;
    }
  };

}

#endif

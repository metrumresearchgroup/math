#ifndef PK_COUPLED_SOLVER_HPP
#define PK_COUPLED_SOLVER_HPP

#include <stan/math/torsten/PKModel/functors/general_functor.hpp>
#include <stan/math/torsten/pk_cptode_adaptor.hpp>
#include <stan/math/torsten/pk_onecpt_solver.hpp>
#include <stan/math/torsten/pk_twocpt_solver.hpp>

namespace refactor {

  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::to_array_1d;
  using std::vector;
  using boost::math::tools::promote_args;

  template<typename T_s1, typename T_s2>
  class PKCoupledModelSolver {
    const T_s1 &sol1_;
    const T_s2 &sol2_;
    
  public:
    // constructor
    PKCoupledModelSolver(const T_s1 &sol1, const T_s2 &sol2) :
      sol1_(sol1), sol2_(sol2)
    {}

    template<typename T_time,
             template <typename...> class T_model1,
             template <typename...> class T_model2,
             typename... Ts1,
             typename... Ts2>
    Eigen::Matrix<typename PKCoupledModel2<T_model1<Ts1...>, T_model2<Ts2...> >::scalar_type, 1,Eigen::Dynamic> 
    solve(const PKCoupledModel2<T_model1<Ts1...>, T_model2<Ts2...> > &coupled_model, const T_time& dt) const
    {
      using model_type = PKCoupledModel2<T_model1<Ts1...>, T_model2<Ts2...> >;
      using scalar = typename model_type::scalar_type;

      Eigen::Matrix<scalar, 1, Eigen::Dynamic> pred;

      const T_model1<Ts1...> & model1 = coupled_model.model1();
      const T_model2<Ts2...> & model2 = coupled_model.model2();
      auto x1 = sol1_.solve(model1, dt);
      auto x2 = sol2_.solve(model2, dt);

      pred.resize(x1.size() + x2.size());
      pred << x1.transpose(), x2.transpose();

      return pred;
    }
    
    // for PK-ODE coupled models
    template<typename T0,
             template<typename...> class T_model,
             typename... Ts>
    Matrix<typename T_model<Ts...>::scalar_type, Dynamic, 1>
    solve(const T_model<Ts...> &model, const T0& dt) const {
      torsten::integrator_structure integrator(sol2_.integrator());
      using Adaptor = PKCptODEAdaptor<T_model<Ts...>, T_s1>;
      PKAdaptODESolver<Adaptor> adapted_sol(integrator);
      auto model1 = model.model.model1();
      auto model2 = model.model.model2();

      using scalar = typename T_model<Ts...>::scalar_type;
      Eigen::Matrix<scalar, 1, Eigen::Dynamic> pred;
      if (stan::math::value_of(dt) < 1.e-9) {
        pred.resize(model1.y0().size() + model2.y0().size());
        pred << model1.y0(), model2.y0();
      } else {
        auto x1 = sol1_.solve(model1, dt);
        auto x2 = adapted_sol.solve(model, dt);
        pred.resize(x1.size() + x2.size());
        pred << x1.transpose(), x2.transpose();
      }
      return pred;      
    }

  };

}




#endif

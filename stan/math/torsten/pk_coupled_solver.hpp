#ifndef PK_COUPLED_SOLVER_HPP
#define PK_COUPLED_SOLVER_HPP

#include <stan/math/torsten/PKModel/functors/general_functor.hpp>

namespace refactor {

  using boost::math::tools::promote_args;
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
    
    template<typename T_time,
             template <typename...> class T_model1,
             template <typename...> class T_model2,
             typename... Ts1,
             typename... Ts2>
    Eigen::Matrix<typename PKCoupledModel2<T_model1<Ts1...>, T_model2<Ts2...> >::scalar_type, 1,Eigen::Dynamic> 
    solve_rate_var(const PKCoupledModel2<T_model1<Ts1...>, T_model2<Ts2...> > &coupled_model, 
           const T_time& dt) const
    {
      using model_type = PKCoupledModel2<T_model1<Ts1...>, T_model2<Ts2...> >;
      using scalar = typename model_type::scalar_type;
      const T_model1<Ts1...> & model1 = coupled_model.model1();
      const T_model2<Ts2...> & model2 = coupled_model.model2();

      Eigen::Matrix<scalar, 1, Eigen::Dynamic> pred;

      if (stan::math::value_of(dt) < 1.e-9) {
        pred.resize(model1.y0().size() + model2.y0().size());
        pred << model1.y0(), model2.y0();
      } else {
        auto x1 = sol1_.solve(model1, dt);

        auto par = model2.par();
        std::vector<typename T_model2<Ts2...>::aug_par_type> new_par(par.size());
        for(size_t i = 0; i < par.size(); ++i) new_par[i] = par[i];
        for (int i = 0; i < model1.y0().size(); i++) new_par.push_back(model1.y0()(i));
        new_par.push_back(stan::math::value_of(model1.t0()));
        auto aug_model2 = model2.make_with_new_par(new_par);
        auto x2 = sol2_.solve(aug_model2, dt);

        pred.resize(x1.size() + x2.size());
        pred << x1.transpose(), x2.transpose();
      }
      return pred;
    }

    template<typename T_time,
             template <typename...> class T_model1,
             template <typename...> class T_model2,
             typename... Ts1,
             typename... Ts2>
    Eigen::Matrix<typename PKCoupledModel2<T_model1<Ts1...>, T_model2<Ts2...> >::scalar_type, 1,Eigen::Dynamic> 
    solve_rate_dbl(const PKCoupledModel2<T_model1<Ts1...>, T_model2<Ts2...> > &coupled_model, 
           const T_time& dt) const {
      using model_type = PKCoupledModel2<T_model1<Ts1...>, T_model2<Ts2...> >;
      using scalar = typename model_type::scalar_type;

      const T_model1<Ts1...> & model1 = coupled_model.model1();
      const T_model2<Ts2...> & model2 = coupled_model.model2();

      Eigen::Matrix<scalar, 1, Eigen::Dynamic> pred;

      if (stan::math::value_of(dt) < 1.e-9) {
        pred.resize(model1.y0().size() + model2.y0().size());
        pred << model1.y0(), model2.y0();
      } else {
        auto x1 = sol1_.solve(model1, dt);

        auto par = model2.par();
        std::vector<typename T_model2<Ts2...>::aug_par_type> new_par(par.size());
        for(size_t i = 0; i < par.size(); ++i) new_par[i] = par[i];
        std::vector<typename T_model2<Ts2...>::rate_type> new_rate{model2.rate()};
        for (int i = 0; i < model1.y0().size(); i++) new_par.push_back(model1.y0()(i));
        new_rate.push_back(stan::math::value_of(model1.t0()));
        auto aug_model2 = model2.make_with_new_par_rate(new_par, new_rate);
        auto x2 = sol2_.solve(aug_model2, dt);

        pred.resize(x1.size() + x2.size());
        pred << x1.transpose(), x2.transpose();
      }
      return pred;
    }
    
    template<typename T0, typename T_time, typename T_init, typename T_rate, typename T_par, typename F, typename Ti>
    Matrix<typename OneCptODEmodel<T_time,
                                   T_init,
                                   T_rate,
                                   T_par,
                                   F,
                                   Ti>::scalar_type, Dynamic, 1>
    solve(const OneCptODEmodel<T_time,
          T_init,
          T_rate,
          T_par,
          F,
          Ti> &model, const T0& dt) const {
      return solve_rate_var(model.model, dt);
    }

    template<typename T0, typename T_time, typename T_init, typename T_par, typename F, typename Ti>
    Matrix<typename OneCptODEmodel<T_time,
                                   T_init,
                                   double,
                                   T_par,
                                   F,
                                   Ti>::scalar_type, Dynamic, 1>
    solve(const OneCptODEmodel<T_time,
          T_init,
          double,
          T_par,
          F,
          Ti> &model, const T0& dt) const {
      return solve_rate_dbl(model.model, dt);
    }

  };

}




#endif

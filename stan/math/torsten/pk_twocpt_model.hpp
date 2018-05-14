#ifndef PK_TWOCPT_MODEL_HPP
#define PK_TWOCPT_MODEL_HPP

namespace refactor {

  using boost::math::tools::promote_args;

  // depend on model, we can have arbitrary number of
  // parameters, e.g. biovar, k12, k10, ka. Each parameter
  // can be data or var.
  template<typename T_time, typename T_init, typename T_rate, typename T_par>
  class PKTwoCptModel {
    const T_time &t0_;
    const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0_;
    const std::vector<T_rate> &rate_;
    const T_par &CL_;
    const T_par &Q_;
    const T_par &V2_;
    const T_par &V3_;
    const T_par &ka_;
    const T_par k10_;
    const T_par k12_;
    const T_par k21_;
    const T_par ksum_;
    const std::vector<T_par> alpha_;

  public:
    static const int ncmt = 3;
    using scalar_type = typename promote_args<T_time, T_rate, T_par, T_init>::type;

    // constructors
    PKTwoCptModel(const T_time& t0, const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0,
                  const std::vector<T_rate> &rate,
                  const T_par& CL,
                  const T_par& Q,
                  const T_par& V2,
                  const T_par& V3,
                  const T_par& ka) :
      t0_(t0),
      y0_(y0),
      rate_(rate),
      CL_(CL),
      Q_(Q),
      V2_(V2),
      V3_(V3),
      ka_(ka),
      k10_(CL_ / V2_),
      k12_(Q_ / V2_),
      k21_(Q_ / V3_),
      ksum_(k10_ + k12_ + k21_),
      alpha_{0.5 * (ksum_ + sqrt(ksum_ * ksum_ - 4 * k10_ * k21_)),
        0.5 * (ksum_ - sqrt(ksum_ * ksum_ - 4 * k10_ * k21_)),
        ka_}
    {}

    PKTwoCptModel(const T_time& t0, const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0,
                  const std::vector<T_rate> &rate,
                  const std::vector<T_par> & par) :
      PKTwoCptModel(t0, y0, rate, par[0], par[1], par[2], par[3], par[4])
    {}

    // PKTwoCptModel(const T_time& t0, const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0,
    //               const Eigen::Matrix<T_par, Eigen::Dynamic, Eigen::Dynamic>& m) :
    //   // TODO
    //   t0_(t0_),
    //   y0_(y0)
    //   // CL_(m.Cl_),
    //   // V2_(m.V2_),
    //   // ka_(m.ka_)
    // {}

    // // copy constructor
    // PKTwoCptModel(const PKTwoCptModel& m) :
    //   PKTwoCptModel(m.t0_, m.y0_, m.rate_, m.CL_, m.Q_, m.V2_, m.V3_, m.ka_)
    // {}

    // get
    const T_time              & t0()   { return t0_; }
    const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0() { return y0_; }
    const std::vector<T_rate> & rate()  { return rate_; }
    const T_par               & CL()    { return CL_;    }
    const T_par               & Q()     { return Q_;     }
    const T_par               & V2()    { return V2_;    }
    const T_par               & V3()    { return V3_;    }
    const T_par               & ka()    { return ka_;    }
    const T_par               & k10()   { return k10_;   }
    const T_par               & k12()   { return k12_;   }
    const T_par               & k21()   { return k21_;   }
    const std::vector<T_par>  & alpha() { return alpha_; }

    // can be solved by linear ode solver
  };

}

#endif

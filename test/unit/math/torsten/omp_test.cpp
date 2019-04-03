#include <test/unit/math/torsten/test_util.hpp>
#include <Eigen/Dense>
#include <Eigen/src/Core/NumTraits.h>
#include <gtest/gtest.h>
#include <vector>

using std::vector;
using Eigen::Matrix;
using Eigen::Dynamic;
using stan::math::var;
using refactor::PKRec;
using torsten::EventsManager;
using torsten::NONMENEventsRecord;

struct binary_functor {
  template<typename T1, typename T2>
  auto operator()(const T1& t1, const T2& t2) {
    return t1 + t2;
  }
};

struct binary_functor2 {
  template<typename T1, typename T2>
  auto operator()(const T1& t1, const T2& t2, double t3) {
    return t3 * (t1 + t2) * stan::math::pow(t2, 2);
  }
};

// requires -DSTAN_THREADS
template<typename F, typename T1, typename T2, typename... Ts>
auto pmx_omp_binary(const F& f, const std::vector<T1>& v1, const std::vector<T2>& v2, const Ts&... other) {
  std::vector<decltype(F()(v1[0], v2[0], other...))> res(v1.size());

#pragma omp parallel for
  for (size_t i = 0; i < v1.size(); ++i) {
    res[i] = f(v1[i], v2[i], other...);
  }

  return res;
}

TEST(pmx_omp, binary) {
  using stan::math::var;

  binary_functor f;
  binary_functor2 f2;
  const int n = 1000000;
  std::vector<double> v1(n, 1);
  std::vector<stan::math::var> v2(n, 4.4);

  pmx_omp_binary(f, v1, v2);

  pmx_omp_binary(f2, v1, v2, 3);
}

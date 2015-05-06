#ifndef STAN_MATH_PRIM_SCAL_META_VECTORVIEWMAT_HPP
#define STAN_MATH_PRIM_SCAL_META_VECTORVIEWMAT_HPP

#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/scal/meta/is_vector_like.hpp>
#include <stan/math/prim/scal/meta/scalar_type_pre.hpp>
#include <stan/math/prim/scal/meta/scalar_type.hpp>
#include <stdexcept>
#include <vector>

namespace stan {

  template <typename T1, typename T2, bool is_array
            = stan::is_vector_like
            <typename stan::math::value_type<T1>::type>::value,
            bool throw_if_accessed = false>
  class VectorViewMat {
  public:
    typedef typename scalar_type<T1>::type matrix_t;
    typedef typename scalar_type<T2>::type scalar_t;

    explicit VectorViewMat(matrix_t& m, scalar_t* scals) : scal_(&scals[0]) {
      rows_.push_back(m.rows());
      accumsizes_.push_back(0);
    }

    explicit VectorViewMat(std::vector<matrix_t>& vm, scalar_t* scals) : scal_(&scals[0]) { 
      int accum = 0;
      for (size_t i = 0; i < vm.size(); ++i) {
        rows_.push_back(vm[i].rows());
        accumsizes_.push_back(accum);
        accum += vm[i].size();
      }
    }

    scalar_t& operator()(int i, int j, int k) {
      int vecind = 0;
      if (throw_if_accessed)
        throw std::out_of_range("VectorViewMat: this cannot be accessed");
      if (is_array) {
        vecind = matind2vecind(accumsizes_[i], j, k, rows_[i]);
      }
      else {
        vecind = matind2vecind(accumsizes_[0], j, k, rows_[0]);
      }
      return scal_[vecind];
    }
    scalar_t& operator()(int i, int j) {
      return operator()(i, j, 0);
    }
    scalar_t& operator()(int i) {
      return operator()(0, 0, i);
    }
    int matind2vecind(int i, int j, int k, int rows) {
      return i + k * rows + j;
    }
  private:
    scalar_t* scal_;
    std::vector<int> rows_;
    std::vector<int> accumsizes_;
  };

  /**
   *
   *  VectorViewMat that has const correctness.
   */
  template <typename T1, typename T2, bool is_array, bool throw_if_accessed>
  class VectorViewMat<const T1, const T2, is_array, throw_if_accessed> {
  public:
    typedef typename scalar_type<T1>::type matrix_t;
    typedef typename scalar_type<T2>::type scalar_t;

    explicit VectorViewMat(const matrix_t& m, const scalar_t* scals) : scal_(&scals[0]) {
      rows_.push_back(m.rows());
      accumsizes_.push_back(0);
    }

    explicit VectorViewMat(const std::vector<matrix_t>& vm, const scalar_t* scals) : scal_(&scals[0]) { 
      int accum = 0;
      for (size_t i = 0; i < vm.size(); ++i) {
        rows_.push_back(vm[i].rows());
        accumsizes_.push_back(accum);
        accum += vm[i].size();
      }
    }

    const scalar_t& operator()(int i, int j, int k) const {
      int vecind = 0;
      if (throw_if_accessed)
        throw std::out_of_range("VectorViewMat: this cannot be accessed");
      if (is_array) {
        vecind = matind2vecind(accumsizes_[i], j, k, rows_[i]);
      }
      else {
        vecind = matind2vecind(accumsizes_[0], j, k, rows_[0]);
      }
      return scal_[vecind];
    }
    int matind2vecind(int i, int j, int k, int rows) {
      return i + k * rows + j;
    }
  private:
    const scalar_t* scal_;
    const std::vector<int> rows_;
    const std::vector<int> accumsizes_;
  };

}
#endif


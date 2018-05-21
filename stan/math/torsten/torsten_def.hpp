#ifndef TORSTEN_DEF_HPP
#define TORSTEN_DEF_HPP

namespace refactor {
    template<typename T>
    using PKRecord = Eigen::Matrix<T, 1, Eigen::Dynamic>;

    template<typename T>
    using PKLinSystem = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

}

#endif

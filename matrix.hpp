#ifndef CUDATEST_MATRIX_HPP
#define CUDATEST_MATRIX_HPP

#include <vector>

namespace math {

template<class T, class Allocator = std::allocator<T>>
class matrix {
    size_t _rows;
    std::vector<T, Allocator> _data;

public:
    explicit matrix(const size_t rows, const size_t cols)
        : _rows{cols ? rows : 0}
        , _data(rows * cols) {}

    explicit matrix(const size_t rows, const size_t cols, const T& val)
        : _rows{cols ? rows : 0}
        , _data(rows * cols, val) {}

    size_t rows() const noexcept { return _rows; }
    size_t cols() const noexcept { return _data.size() / _rows; }
    size_t size() const noexcept { return _data.size(); }

    const T* data() const noexcept { return _data.data(); }
    T* data() noexcept { return _data.data(); }
};

}

#endif //CUDATEST_MATRIX_HPP
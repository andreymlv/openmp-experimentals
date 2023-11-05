#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include <omp.h>

class SubMatrix {
  const std::vector<std::vector<double>> *source;
  std::vector<double> replaceColumn;
  const SubMatrix *prev;
  size_t sz;
  int colIndex = -1;

public:
  SubMatrix(const std::vector<std::vector<double>> &src,
            const std::vector<double> &rc)
      : source(&src), replaceColumn(rc), prev(nullptr), colIndex(-1) {
    sz = replaceColumn.size();
  }

  SubMatrix(const SubMatrix &p) : source(nullptr), prev(&p), colIndex(-1) {
    sz = p.size() - 1;
  }

  SubMatrix(const SubMatrix &p, int deletedColumnIndex)
      : source(nullptr), prev(&p), colIndex(deletedColumnIndex) {
    sz = p.size() - 1;
  }

  int columnIndex() const { return colIndex; }
  void columnIndex(int index) { colIndex = index; }

  size_t size() const { return sz; }

  double index(int row, int col) const {
    if (source != nullptr) {
      if (col == colIndex) {
        return replaceColumn[row];
      } else {
        return (*source)[row][col];
      }
    } else {
      if (col < colIndex) {
        return prev->index(row + 1, col);
      } else {
        return prev->index(row + 1, col + 1);
      }
    }
  }

  double det() const {
    if (sz == 1) {
      return index(0, 0);
    }
    if (sz == 2) {
      return index(0, 0) * index(1, 1) - index(0, 1) * index(1, 0);
    }
    SubMatrix m(*this);
    double det = 0.0;
    int sign = 1;
#pragma omp parallel for reduction(+ : det)
    for (size_t c = 0; c < sz; ++c) {
      m.columnIndex(c);
      double d = m.det();
      det += index(0, c) * d * sign;
      sign = -sign;
    }
    return det;
  }
};

std::vector<double> solve(SubMatrix &matrix) {
  double det = matrix.det();
  if (det == 0.0) {
    throw std::runtime_error("The determinant is zero.");
  }

  std::vector<double> answer(matrix.size());
#pragma omp parallel for
  for (int i = 0; i < matrix.size(); ++i) {
    matrix.columnIndex(i);
    answer[i] = matrix.det() / det;
  }
  return answer;
}

std::vector<double>
solveCramer(const std::vector<std::vector<double>> &equations) {
  int size = equations.size();
  if (std::any_of(equations.cbegin(), equations.cend(),
                  [size](const std::vector<double> &a) {
                    return a.size() != size + 1;
                  })) {
    throw std::runtime_error("Each equation must have the expected size.");
  }

  std::vector<std::vector<double>> matrix(size);
  std::vector<double> column(size);
#pragma omp parallel for
  for (int r = 0; r < size; ++r) {
    column[r] = equations[r][size];
    matrix[r].resize(size);
    for (int c = 0; c < size; ++c) {
      matrix[r][c] = equations[r][c];
    }
  }

  SubMatrix sm(matrix, column);
  return solve(sm);
}

std::vector<std::vector<double>> generateRandomMatrix(double from, double to,
                                                      int n, int m) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distribution(from, to);

  std::vector<std::vector<double>> matrix(n, std::vector<double>(m));

  for (auto &row : matrix) {
    for (double &value : row) {
      value = distribution(gen);
    }
  }

  return matrix;
}

static void BM_CramerParallel(benchmark::State &state) {
  int n = state.range(0);

  // Initialize coefficients and constants (you can fill in your own values)

  for (auto _ : state) {
    auto solutions = solveCramer(generateRandomMatrix(-10, 10, n, n + 1));
  }
}

// static void BM_CramerSequential(benchmark::State& state) {
//     int n = state.range(0);
//     std::vector<std::vector<double>> coefficients(n, std::vector<double>(n,
//     0.0)); std::vector<double> constants(n, 0.0);

//     // Initialize coefficients and constants (you can fill in your own
//     values)

//     for (auto _ : state) {
//         std::vector<double> solutions = solveLinearSystem(coefficients,
//         constants);
//     }
// }

BENCHMARK(BM_CramerParallel)->Arg(100)->Arg(500)->Arg(1000);
// BENCHMARK(BM_CramerSequential)->Arg(100)->Arg(500)->Arg(1000);

BENCHMARK_MAIN();

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <benchmark/benchmark.h>

// Function to calculate the determinant of a matrix
double calculateDeterminant(std::vector<std::vector<double>> matrix) {
    int n = matrix.size();
    if (n == 1) {
        return matrix[0][0];
    } else if (n == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    } else {
        double det = 0.0;
        #pragma omp parallel for reduction(+:det)
        for (int i = 0; i < n; i++) {
            std::vector<std::vector<double>> submatrix(n - 1, std::vector<double>(n - 1, 0.0));
            for (int j = 1; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    if (k < i) {
                        submatrix[j - 1][k] = matrix[j][k];
                    } else if (k > i) {
                        submatrix[j - 1][k - 1] = matrix[j][k];
                    }
                }
            }
            double subdet = calculateDeterminant(submatrix);
            det += matrix[0][i] * subdet * (i % 2 == 0 ? 1 : -1);
        }
        return det;
    }
}

// Function to solve a system of linear equations using Cramer's rule
std::vector<double> solveLinearSystem(std::vector<std::vector<double>> coefficients, std::vector<double> constants) {
    int n = coefficients.size();
    double detA = calculateDeterminant(coefficients);
    if (std::abs(detA) < 1e-6) {
        // The system is singular or nearly singular
        throw std::runtime_error("The system of equations is singular or nearly singular.");
    }
    
    std::vector<double> solutions(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        std::vector<std::vector<double>> tempMatrix = coefficients;
        for (int j = 0; j < n; j++) {
            tempMatrix[j][i] = constants[j];
        }
        double detAi = calculateDeterminant(tempMatrix);
        solutions[i] = detAi / detA;
    }
    return solutions;
}

static void BM_CramerParallel(benchmark::State& state) {
    int n = state.range(0);
    std::vector<std::vector<double>> coefficients(n, std::vector<double>(n, 0.0));
    std::vector<double> constants(n, 0.0);

    // Initialize coefficients and constants (you can fill in your own values)

    for (auto _ : state) {
        std::vector<double> solutions = solveLinearSystem(coefficients, constants);
    }
}

// static void BM_CramerSequential(benchmark::State& state) {
//     int n = state.range(0);
//     std::vector<std::vector<double>> coefficients(n, std::vector<double>(n, 0.0));
//     std::vector<double> constants(n, 0.0);

//     // Initialize coefficients and constants (you can fill in your own values)

//     for (auto _ : state) {
//         std::vector<double> solutions = solveLinearSystem(coefficients, constants);
//     }
// }

BENCHMARK(BM_CramerParallel)->Arg(100)->Arg(500)->Arg(1000);
// BENCHMARK(BM_CramerSequential)->Arg(100)->Arg(500)->Arg(1000);

BENCHMARK_MAIN();

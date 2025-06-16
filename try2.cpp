#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <omp.h>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <chrono>

using namespace boost::math;
using namespace boost::math::quadrature;

// Clayton Copula - optimized with inline functions
class ClaytonCopula {
private:
    double theta;
    
public:
    ClaytonCopula(double theta_val) : theta(theta_val) {}
    
    inline double psi(double x) const {
        return std::pow(x, -1.0 / theta);
    }
    
    inline double psi1(double x) const {
        return (-1.0 / theta) * std::pow(x, -1.0 / theta - 1.0);
    }
    
    inline double psi2(double x) const {
        return (1.0 / theta) * (1.0 + 1.0 / theta) * std::pow(x, -1.0 / theta - 2.0);
    }
    
    inline double phi(double x) const {
        return std::pow(x, -theta);
    }
    
    inline double phi1(double x) const {
        return -theta * std::pow(x, -theta - 1.0);
    }
    
    inline double copula_density(double x1, double x2) const {
        return psi2(phi(x1) + phi(x2)) * phi1(x1) * phi1(x2);
    }
};

// Vectorized operations using Eigen
class VectorizedModel {
private:
    // Parameters
    static constexpr double shape1 = 2.6, scale1 = 1.0;
    static constexpr double shape2 = 2.0, scale2 = 1.4;
    static constexpr double lambda1 = 0.8, lambda2 = 0.5;
    static constexpr double N = 1000.0;
    static constexpr double R = 0.04;
    static constexpr double eta = 1;
    static constexpr double theta = 0.5;
    
    double FP1, FP2, b1, b2;
    
    // Boost distributions for maximum accuracy and speed
    gamma_distribution<double> gamma1, gamma2;
    ClaytonCopula copula;
    
public:
    VectorizedModel() : 
        gamma1(shape1, scale1), 
        gamma2(shape2, scale2),
        copula(theta) {
        
        FP1 = lambda1 * scale1 * shape1;
        FP2 = lambda2 * scale2 * shape2;
        b1 = 1.0 / FP1;
        b2 = 1.0 / FP2;
        
        std::cout << "FP1: " << FP1 << ", FP2: " << FP2 << std::endl;
        std::cout << "b1: " << b1 << ", b2: " << b2 << std::endl;
    }
    
    inline double F1_inv(double x) const {
        return quantile(gamma1, x);
    }
    
    inline double F2_inv(double x) const {
        return quantile(gamma2, x);
    }
    
    inline double n1(double p1, double p2) const {
        return N * std::exp(-b1 * p1) / (1.0 + std::exp(-b1 * p1 - b2 * p2));
    }
    
    inline double n2(double p1, double p2) const {
        return N * std::exp(-b2 * p2) / (1.0 + std::exp(-b1 * p1 - b2 * p2));
    }
    
    inline double alpha1(double p1, double p2) const {
        return n1(p1, p2) * lambda1;
    }
    
    inline double alpha2(double p1, double p2) const {
        return n2(p1, p2) * lambda2;
    }
    
    // Vectorized integrand computation
    inline double integrand(double x1, double x2, double p1, double p2) const {
        double inv1 = F1_inv(1.0 - x1);
        double inv2 = F2_inv(1.0 - x2);
        double A = std::exp(R * eta * (inv1 + inv2)) - 1.0;
        
        double a1 = alpha1(p1, p2);
        double a2 = alpha2(p1, p2);
        
        return A * copula.copula_density(a1 * x1, a2 * x2);
    }
    
    // Parallel 2D integration using OpenMP
    double parallel_integrate_2d(double p1, double p2, int n_points = 100) const {
        double x_min = 0.01, x_max = 0.99;
        double y_min = 0.01, y_max = 0.99;
        
        double dx = (x_max - x_min) / n_points;
        double dy = (y_max - y_min) / n_points;
        
        double sum = 0.0;
        
        #pragma omp parallel for reduction(+:sum) collapse(2) schedule(static)
        for (int i = 0; i < n_points; ++i) {
            for (int j = 0; j < n_points; ++j) {
                double x = x_min + (i + 0.5) * dx;
                double y = y_min + (j + 0.5) * dy;
                sum += integrand(x, y, p1, p2);
            }
        }
        
        return sum * dx * dy;
    }
    
    // Alternative: Use Boost's adaptive quadrature (slower but more accurate)
    double boost_integrate_2d(double p1, double p2) const {
        auto integrand_func = [this, p1, p2](double x1, double x2) {
            return this->integrand(x1, x2, p1, p2);
        };
        
        // Nest 1D integrations
        auto outer_integrand = [&](double x1) {
            auto inner_integrand = [&](double x2) {
                return integrand_func(x1, x2);
            };
            return gauss_kronrod<double, 15>::integrate(inner_integrand, 0.01, 0.99);
        };
        
        return gauss_kronrod<double, 15>::integrate(outer_integrand, 0.01, 0.99);
    }
    
    double Q_parallel(double p1, double p2) const {
        auto start = std::chrono::high_resolution_clock::now();
        
        double B = parallel_integrate_2d(p1, p2, 200); // Higher resolution
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Parallel integration time: " << duration.count() << " ms" << std::endl;
        
        return R * eta * (n1(p1, p2) * p1 + n2(p1, p2) * p2) - 
               B * alpha1(p1, p2) * alpha2(p1, p2);
    }
    
    double Q_boost(double p1, double p2) const {
        auto start = std::chrono::high_resolution_clock::now();
        
        double B = boost_integrate_2d(p1, p2);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Boost integration time: " << duration.count() << " ms" << std::endl;
        
        return R * eta * (n1(p1, p2) * p1 + n2(p1, p2) * p2) - 
               B * alpha1(p1, p2) * alpha2(p1, p2);
    }
    
    // Batch computation for multiple parameter sets
    std::vector<double> Q_batch(const std::vector<std::pair<double, double>>& params) const {
        std::vector<double> results(params.size());
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < params.size(); ++i) {
            results[i] = Q_parallel(params[i].first, params[i].second);
        }
        
        return results;
    }
    
    void benchmark() const {
        std::cout << "\n=== Performance Benchmark ===" << std::endl;
        
        // Test copula density
        auto start = std::chrono::high_resolution_clock::now();
        double copula_test = ClaytonCopula(1.0).copula_density(0.5, 0.3);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Copula density (2.5, 2.5, theta=0.5): " << copula_test << std::endl;
        
        // Test Q function with different methods
        std::cout << "\nTesting Q(4, 5):" << std::endl;
        
        double q_parallel = Q_parallel(2.5, 2.5);
        std::cout << "Q_parallel result: " << q_parallel << std::endl;
        
        double q_boost = Q_boost(2.5, 2.5);
        std::cout << "Q_boost result: " << q_boost << std::endl;
        
        // Batch test
        std::vector<std::pair<double, double>> test_params = {
            {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}, {5.0, 6.0}
        };
        
        start = std::chrono::high_resolution_clock::now();
        auto batch_results = Q_batch(test_params);
        end = std::chrono::high_resolution_clock::now();
        auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "\nBatch computation (" << test_params.size() << " parameter sets):" << std::endl;
        std::cout << "Time: " << batch_duration.count() << " ms" << std::endl;
        for (size_t i = 0; i < batch_results.size(); ++i) {
            std::cout << "Q(" << test_params[i].first << ", " << test_params[i].second 
                      << ") = " << batch_results[i] << std::endl;
        }
    }
};

int main() {
    // Set number of OpenMP threads
    omp_set_num_threads(std::min(8, omp_get_max_threads()));
    std::cout << "Using " << omp_get_max_threads() << " OpenMP threads" << std::endl;
    
    VectorizedModel model;
    model.benchmark();
    
    return 0;
}

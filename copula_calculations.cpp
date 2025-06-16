#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // For automatic vector conversions
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <omp.h> // For OpenMP parallelization
#include <algorithm> // For std::sort, std::max_element, std::min
#include <tuple> // For storing maxima temporarily
#include <cstring> // For std::memcpy

// Boost includes
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>

namespace py = pybind11;

// --- Global Model Parameters (Constants) ---
const double shape1 = 2.6;
const double scale1 = 1.0;
const double shape2 = 2.0;
const double scale2 = 1.4;
const double lambda1 = 0.8;
const double lambda2 = 0.5;
const double N = 1000.0;
const double R = 0.04;

const double FP1 = lambda1 * scale1 * shape1;
const double FP2 = lambda2 * scale2 * shape2;
const double b1 = (FP1 != 0) ? 1.0 / FP1 : std::numeric_limits<double>::infinity();
const double b2 = (FP2 != 0) ? 1.0 / FP2 : std::numeric_limits<double>::infinity();

// Boost distributions for maximum accuracy and speed
const boost::math::gamma_distribution<double> gamma1_dist(shape1, scale1);
const boost::math::gamma_distribution<double> gamma2_dist(shape2, scale2);

const double low_lim = 0.01; // Integration lower limit (matching Python ground truth)
const double high_lim = 0.99; // Integration upper limit (matching Python ground truth)
const double tol = 0.01; // Integration tolerance (matching Python ground truth epsabs/epsrel)

// --- Helper Functions ---
inline double clip(double val, double min_val, double max_val) {
    return std::max(min_val, std::min(val, max_val));
}

// --- Clayton Copula Class (from try2.cpp) ---
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

// --- Inverse CDFs (using Boost) ---
__attribute__((visibility("default"))) double F1_inv(double x) {
    x = clip(x, low_lim, high_lim);
    try {
        double result = boost::math::quantile(gamma1_dist, x);
        return std::isfinite(result) ? result : std::numeric_limits<double>::quiet_NaN();
    }
    catch (...) { return std::numeric_limits<double>::quiet_NaN(); }
}

__attribute__((visibility("default"))) double F2_inv(double x) {
    x = clip(x, low_lim, high_lim);
    try {
        double result = boost::math::quantile(gamma2_dist, x);
        return std::isfinite(result) ? result : std::numeric_limits<double>::quiet_NaN();
     }
    catch (...) { return std::numeric_limits<double>::quiet_NaN(); }
}

// --- Demand Functions ---
double n1(double p1, double p2) {
   double exp_arg = clip(-b1 * p1 - b2 * p2, -700.0, 700.0);
   double denominator = 1.0 + std::exp(exp_arg);
   if (denominator == 0 || !std::isfinite(denominator)) return 0.0;
   double num_exp_arg = clip(-b1 * p1, -700.0, 700.0);
   double result = N * std::exp(num_exp_arg) / denominator;
   return std::isfinite(result) ? result : 0.0;
}

double n2(double p1, double p2) {
   double exp_arg = clip(-b1 * p1 - b2 * p2, -700.0, 700.0);
   double denominator = 1.0 + std::exp(exp_arg);
   if (denominator == 0 || !std::isfinite(denominator)) return 0.0;
   double num_exp_arg = clip(-b2 * p2, -700.0, 700.0);
   double result = N * std::exp(num_exp_arg) / denominator;
    return std::isfinite(result) ? result : 0.0;
}

// --- Alpha Functions ---
double alpha1(double p1, double p2) {
    double n1_val = n1(p1, p2);
    return std::isfinite(n1_val) ? n1_val * lambda1 : 0.0;
}
double alpha2(double p1, double p2) {
    double n2_val = n2(p1, p2);
    return std::isfinite(n2_val) ? n2_val * lambda2 : 0.0;
}

// --- CopulaModelCalculator Class (adapted from VectorizedModel) ---
class CopulaModelCalculator {
private:
    ClaytonCopula copula;
    
public:
    CopulaModelCalculator(double theta_val) : copula(theta_val) {}
    
    inline double integrand(double x1, double x2, double p1, double p2, double current_eta) const {
        double inv1 = F1_inv(1.0 - x1);
        double inv2 = F2_inv(1.0 - x2);
        double A = std::exp(R * current_eta * (inv1 + inv2)) - 1.0;
        
        double a1 = alpha1(p1, p2);
        double a2 = alpha2(p1, p2);
        
        return A * copula.copula_density(a1 * x1, a2 * x2);
    }
    
    // Parallel 2D integration using OpenMP (kept for reference, but not used in Q_calc)
    double parallel_integrate_2d(double p1, double p2, double current_eta, int n_points = 100) const {
        double x_min = low_lim, x_max = high_lim;
        double y_min = low_lim, y_max = high_lim;
        
        double dx = (x_max - x_min) / n_points;
        double dy = (y_max - y_min) / n_points;
        
        double sum = 0.0;
        
        #pragma omp parallel for reduction(+:sum) collapse(2) schedule(static)
        for (int i = 0; i < n_points; ++i) {
            for (int j = 0; j < n_points; ++j) {
                double x = x_min + (i + 0.5) * dx;
                double y = y_min + (j + 0.5) * dy;
                sum += integrand(x, y, p1, p2, current_eta);
            }
        }
        
        return sum * dx * dy;
    }

    // 2D integration using Boost's adaptive Gauss-Kronrod quadrature
    double boost_integrate_2d(double p1, double p2, double current_eta) const {
        auto integrand_func = [this, p1, p2, current_eta](double x1, double x2) {
            return this->integrand(x1, x2, p1, p2, current_eta);
        };
        
        // Nest 1D integrations
        auto outer_integrand = [&](double x1) {
            auto inner_integrand = [&](double x2) {
                return integrand_func(x1, x2);
            };
            // Use the global tolerance 'tol' for integration
            return boost::math::quadrature::gauss_kronrod<double, 15>::integrate(inner_integrand, low_lim, high_lim, tol);
        };
        
        return boost::math::quadrature::gauss_kronrod<double, 15>::integrate(outer_integrand, low_lim, high_lim, tol);
    }
    
    double Q_calc(double p1, double p2, double current_eta) const {
        // Use the Boost integration method
        double B = boost_integrate_2d(p1, p2, current_eta);
        
        return R * current_eta * (n1(p1, p2) * p1 + n2(p1, p2) * p2) - 
               B * alpha1(p1, p2) * alpha2(p1, p2);
    }
};

// --- Q Function (C++) - updated to use CopulaModelCalculator ---
double Q_cpp(double p1, double p2, double current_eta, double current_theta) {
    CopulaModelCalculator calculator(current_theta);
    return calculator.Q_calc(p1, p2, current_eta);
}

// --- Wrapper Function for Python ---
// Calculates the Q matrix and finds the top 1 or 2 maxima (global + second highest)
__attribute__((visibility("default"))) py::list calculate_maxima_cpp(
    double current_theta,
    double current_eta,
    py::array_t<double> P1_plot_np,
    py::array_t<double> P2_plot_np,
    bool return_q_matrix = false)
{
    py::buffer_info p1_buf = P1_plot_np.request();
    py::buffer_info p2_buf = P2_plot_np.request();
    if (p1_buf.ndim != 1 || p2_buf.ndim != 1) throw std::runtime_error("Input price arrays must be 1-dimensional");

    double* p1_ptr = static_cast<double*>(p1_buf.ptr);
    double* p2_ptr = static_cast<double*>(p2_buf.ptr);
    size_t n_p1 = p1_buf.shape[0];
    size_t n_p2 = p2_buf.shape[0];

    std::vector<double> q_matrix(n_p1 * n_p2);

    // Calculate Q matrix in parallel
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n_p1; ++i) {
        for (size_t j = 0; j < n_p2; ++j) {
            q_matrix[i * n_p2 + j] = Q_cpp(p1_ptr[i], p2_ptr[j], current_eta, current_theta);
        }
    }

    py::list py_results;
    py::array_t<double> q_matrix_np; // Declare for potential return

    // Find all local maxima
    std::vector<py::dict> local_maxima;
    // Reduced minimum squared distance for distinct maxima to allow closer peaks
    double min_dist_sq = 0.1 * 0.1; // e.g., 0.1 units in p1/p2 space

    for (size_t i = 0; i < n_p1; ++i) {
        for (size_t j = 0; j < n_p2; ++j) {
            size_t current_idx = i * n_p2 + j;
            double current_q = q_matrix[current_idx];

            if (!std::isfinite(current_q)) continue;

            bool is_local_max = true;
            // Check 8 neighbors
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    if (di == 0 && dj == 0) continue; // Skip self

                    long long ni = (long long)i + di;
                    long long nj = (long long)j + dj;

                    // Check bounds
                    if (ni >= 0 && ni < n_p1 && nj >= 0 && nj < n_p2) {
                        double neighbor_q = q_matrix[ni * n_p2 + nj];
                        if (std::isfinite(neighbor_q) && neighbor_q > current_q) {
                            is_local_max = false;
                            break;
                        }
                    }
                }
                if (!is_local_max) break;
            }

            if (is_local_max) {
                py::dict max_info;
                max_info["max_Q"] = current_q;
                max_info["max_i"] = i;
                max_info["max_j"] = j;
                max_info["max_p1"] = p1_ptr[i];
                max_info["max_p2"] = p2_ptr[j];
                local_maxima.push_back(max_info);
            }
        }
    }

    // Sort local maxima by Q value in descending order
    std::sort(local_maxima.begin(), local_maxima.end(), [](const py::dict& a, const py::dict& b) {
        return a["max_Q"].cast<double>() > b["max_Q"].cast<double>();
    });

    // Filter for distinct maxima and add to results
    std::vector<py::dict> distinct_maxima;
    for (const auto& current_max : local_maxima) {
        bool is_distinct = true;
        double current_p1 = current_max["max_p1"].cast<double>();
        double current_p2 = current_max["max_p2"].cast<double>();

        for (const auto& existing_distinct_max : distinct_maxima) {
            double existing_p1 = existing_distinct_max["max_p1"].cast<double>();
            double existing_p2 = existing_distinct_max["max_p2"].cast<double>();
            
            double dist_sq = std::pow(current_p1 - existing_p1, 2) + std::pow(current_p2 - existing_p2, 2);
            if (dist_sq < min_dist_sq) {
                is_distinct = false;
                break;
            }
        }

        if (is_distinct) {
            distinct_maxima.push_back(current_max);
            if (distinct_maxima.size() >= 2) { // Limit to top 2 distinct maxima for plotting
                break;
            }
        }
    }

    // Add distinct maxima to py_results
    for (const auto& max_info : distinct_maxima) {
        py_results.append(max_info);
    }

    // Optionally copy Q matrix to NumPy array (do this once)
    if (return_q_matrix) {
         q_matrix_np = py::array_t<double>({n_p1, n_p2});
         py::buffer_info q_buf = q_matrix_np.request();
         double* q_ptr = static_cast<double*>(q_buf.ptr);
         std::memcpy(q_ptr, q_matrix.data(), q_matrix.size() * sizeof(double));
         // Add matrix to the first result dict if it exists, otherwise create a dummy dict
         if (!py_results.empty()) {
             py_results[0]["q_matrix"] = q_matrix_np;
         } else {
             py::dict dummy_info;
             dummy_info["q_matrix"] = q_matrix_np;
             py_results.append(dummy_info);
         }
    }

    return py_results;
}


// --- pybind11 Module Definition ---
PYBIND11_MODULE(copula_calculations, m) {
    m.doc() = "C++ extension for calculating copula-based Q function and finding maxima";

     m.def("calculate_maxima_cpp", &calculate_maxima_cpp, // Changed function name
           "Calculates Q matrix, finds top 1-2 maxima (global + second highest), returns list of dicts",
           py::arg("current_theta"), py::arg("current_eta"),
           py::arg("P1_plot_np"), py::arg("P2_plot_np"),
           py::arg("return_q_matrix") = false);

    // Temporarily expose F_inv functions for debugging
    m.def("F1_inv_cpp", &F1_inv, "Inverse CDF for F1");
    m.def("F2_inv_cpp", &F2_inv, "Inverse CDF for F2");
}

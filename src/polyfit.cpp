#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>
#include <limits>

typedef double (*kernel_func)(double, const void *);

Eigen::VectorXd fit(kernel_func f, double lb, double ub, int order, const void *data, double offset = 0.0) {
    Eigen::MatrixXd Vandermonde(order, order);
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(order, lb, ub);

    Vandermonde.col(0) = Eigen::VectorXd::Ones(order);
    for (int i_col = 1; i_col < order; i_col++)
        Vandermonde.col(i_col) = Vandermonde.col(i_col - 1).array() * x.array();

    Eigen::VectorXd y(order);
    for (int i = 0; i < order; i++)
        y(i) = f(x(i) + offset, data);

    return Vandermonde.lu().solve(y).reverse();
}

double eval(const Eigen::VectorXd &coeffs, double x) {
    double y = coeffs(0);
    for (auto j = 1; j < coeffs.size(); ++j)
        y = std::fma(y, x, coeffs[j]);

    return y;
}

void fit(kernel_func f, double lb, double ub, int order, const void *data, double *coeffs) {
    Eigen::Map<Eigen::VectorXd> coeffs_map(coeffs, order);
    coeffs_map = fit(f, lb, ub, order, data);
}

double eval_multi(const Eigen::VectorXd &coeffs, double x, double lb, double ub, int order, double h) {
    const double half_h = 0.5 * h;
    const int n         = coeffs.size() / order;
    const int i         = std::min(n - 1, int((x - lb) / h));
    const double x_i    = lb + half_h * (2 * i + 1);
    return eval(coeffs.segment(i * order, order), x - x_i);
}

void fit_multi(kernel_func f, double lb, double ub, int order, const void *data, double *coeffs, int n) {
    Eigen::Map<Eigen::VectorXd> coeffs_map(coeffs, order * n);
    const double half_h = (ub - lb) / (2 * n);

    for (int i = 0; i < n; i++) {
        const double offset                  = lb + half_h * (2 * i + 1);
        coeffs_map.segment(i * order, order) = fit(f, -half_h, half_h, order, data, offset);
    }
}

std::tuple<double, double> get_errors_for_auto(
    kernel_func f, const Eigen::VectorXd &sample_points, const Eigen::VectorXd &actual_vals, int order,
    const void *data, const Eigen::VectorXd &coeffs, double lb, double ub, double h, const int n_samples) {
    double max_rel_error = 0.0;
    double avg_rel_error = 0.0;

    for (int i = 0; i < n_samples; i++) {
        double x        = sample_points(i);
        double y        = eval_multi(coeffs, x, lb, ub, order, h);
        double actual_y = actual_vals(i);
        double error    = std::abs(1.0 - y / actual_y);
        avg_rel_error += error;
        if (error > max_rel_error) {
            max_rel_error = error;
        }
    }
    avg_rel_error /= n_samples;

    return {avg_rel_error, max_rel_error};
}

Eigen::VectorXd fit_multi_auto(kernel_func f, double lb, double ub, const void *data, int n_poly, double error_tol,
                               int min_order, int max_order, int n_samples) {
    double ub_safe     = ub - std::numeric_limits<double>::epsilon();
    uint64_t n_epsilon = 1;
    while (ub_safe == ub) {
        ub_safe -= n_epsilon * std::numeric_limits<double>::epsilon();
        n_epsilon *= 2;
    }

    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n_samples, lb, ub_safe);
    Eigen::VectorXd y = Eigen::VectorXd::Zero(n_samples);
    for (int i = 0; i < n_samples; i++)
        y(i) = f(x(i), data);

    for (int order = min_order; order < max_order; order++) {
        Eigen::VectorXd coeffs(order * n_poly);
        fit_multi(f, lb, ub, order, data, coeffs.data(), n_poly);

        auto [avg_rel_error, max_rel_error] =
            get_errors_for_auto(f, x, y, order, data, coeffs, lb, ub, (ub - lb) / n_poly, n_samples);
        if (avg_rel_error < error_tol) return coeffs;
    }

    return Eigen::VectorXd();
}

void eval_grid(const Eigen::VectorXd &coeffs, double x, double lb, double ub, int order, int n, double *output) {
    const double h      = (ub - lb) / n;
    const double half_h = 0.5 * h;
    const int i         = (x - lb) / h;
    const double dx     = x - (lb + half_h * (2 * i + 1));
    for (int j = 0; j < n; j++)
        output[j] = eval(coeffs.segment(j * order, order), dx);
}

void testfit() {
    using namespace Eigen;
    const double lb       = -0.7;
    const double ub       = 0.8;
    const int order       = 16;
    const int n_samples   = 100;
    const int max_order   = 16;
    const int min_order   = 3;
    const int n_poly      = 7;
    const double h        = (ub - lb) / n_poly;
    const int multi_order = 8;

    void *data    = NULL;
    kernel_func f = [](double x, const void *data) {
        return sin(pow(x, 2) * cos(x));
    };

    VectorXd coeffs = fit(f, lb, ub, order, data);

    VectorXd x = VectorXd::LinSpaced(n_samples, lb, ub - std::numeric_limits<double>::epsilon());
    VectorXd y = VectorXd::Zero(n_samples);
    for (int i = 0; i < n_samples; i++)
        y(i) = eval(coeffs, x(i));

    VectorXd yraw = VectorXd::Zero(n_samples);
    for (int i = 0; i < n_samples; i++)
        yraw(i) = f(x(i), data);

    auto stderr = (y - yraw).array().abs().sum() / n_samples;
    std::cout << stderr << std::endl;

    VectorXd coeffs_multi = VectorXd::Zero(multi_order * n_poly);
    fit_multi(f, lb, ub, multi_order, data, coeffs_multi.data(), n_poly);

    {
        VectorXd y_multi = VectorXd::Zero(n_samples);
        for (int i = 0; i < n_samples; i++)
            y_multi(i) = eval_multi(coeffs_multi, x(i), lb, ub, multi_order, h);
        auto stderr_multi = (y_multi - yraw).array().abs().sum() / n_samples;
        std::cout << stderr_multi << std::endl;
    }

    VectorXd y_grid       = VectorXd::Zero(n_poly);
    VectorXd x_grid_multi = VectorXd::LinSpaced(n_poly, lb + 0.75 * h, ub - 0.25 * h);
    VectorXd y_grid_multi = VectorXd::Zero(n_poly);
    eval_grid(coeffs_multi, lb + 1.75 * h, lb, ub, multi_order, n_poly, y_grid.data());
    for (int i = 0; i < n_poly; i++)
        y_grid_multi(i) = f(x_grid_multi(i), data);

    auto stderr_grid = (y_grid - y_grid_multi).array().abs().sum() / n_poly;
    std::cout << stderr_grid << std::endl;

    auto coeffs_auto = fit_multi_auto(f, lb, ub, data, n_poly, 1E-6, min_order, max_order, n_samples);
    auto auto_order  = coeffs_auto.size() / n_poly;

    VectorXd y_auto = VectorXd::Zero(n_samples);
    for (int i = 0; i < n_samples; i++)
        y_auto(i) = eval_multi(coeffs_auto, x(i), lb, ub, auto_order, h);
    auto stderr_auto = (y_auto - yraw).array().abs().sum() / n_samples;
    std::cout << stderr_auto << std::endl;
}

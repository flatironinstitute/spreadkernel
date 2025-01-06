#include <doctest/doctest.h>
#include <polyfit.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

namespace spreadkernel::polyfit {

std::vector<double> vandermonde_inverse(int order, double *x) {
    // https://github.com/yveschris/possibly-the-fastest-analytical-inverse-of-vandermonde-matrices
    // No license was specified, though the algorithm is generally known
    std::vector<double> p(order + 1);
    std::vector<double> C(order);

    C[0]         = 1;
    p[order - 1] = -x[0];
    p[order]     = 1;
    for (int i = 1; i < order; i++) {
        auto pt = p;
        for (int j = 0; j < i + 1; j++) {
            p[order - i + j - 1] = pt[order - i + j - 1] - x[i] * pt[order - i + j];
        }

        auto Cp = C;
        for (int j = 0; j < i; j++) {
            C[j] = Cp[j] / (x[j] - x[i]);
            C[i] -= C[j];
        }
    }

    std::vector<double> Vinv(order * order);
    std::vector<double> c(order);
    for (int i = 0; i < order; i++) {
        c[order - 1] = 1;
        for (int j = order - 2; j >= 0; j--)
            c[j] = p[j + 1] + x[i] * c[j + 1];

        for (int j = 0; j < order; j++)
            Vinv[j * order + i] = c[j] * C[i];
    }

    return Vinv;
}

std::vector<double> linspaced(int n, double lb, double ub) {
    std::vector<double> x(n);
    for (int i = 0; i < n; i++)
        x[i] = lb + (ub - lb) * i / (n - 1);

    return x;
}

void fit(kernel_func f, double lb, double ub, int order, const void *data, double *coeffs, double offset) {
    std::vector<double> x = linspaced(order, lb, ub);

    std::vector<double> y(order);
    for (int i = 0; i < order; i++)
        y[i] = f(x[i] + offset, data);

    std::vector<double> Vinv = vandermonde_inverse(order, x.data());
    for (int i = 0; i < order; i++) {
        coeffs[order - i - 1] = 0;
        for (int j = 0; j < order; j++)
            coeffs[order - i - 1] += Vinv[i * order + j] * y[j];
    }
}

double eval(const double *coeffs, int order, double x) {
    double y = coeffs[0];
    for (auto j = 1; j < order; ++j)
        y = std::fma(y, x, coeffs[j]);

    return y;
}

std::vector<double> fit(kernel_func f, double lb, double ub, int order, const void *data) {
    std::vector<double> coeffs(order);
    fit(f, lb, ub, order, data, coeffs.data(), 0.0);
    return coeffs;
}

double eval_multi(const std::vector<double> &coeffs, double x, double lb, double ub, int order, double h) {
    const double half_h = 0.5 * h;
    const int n         = coeffs.size() / order;
    const int i         = std::min(n - 1, int((x - lb) / h));
    const double x_i    = lb + half_h * (2 * i + 1);
    return eval(coeffs.data() + i * order, order, x - x_i);
}

void fit_multi(kernel_func f, double lb, double ub, int order, const void *data, double *coeffs, int n) {
    const double half_h = (ub - lb) / (2 * n);

    for (int i = 0; i < n; i++) {
        const double offset = lb + half_h * (2 * i + 1);
        fit(f, -half_h, half_h, order, data, coeffs + i * order, offset);
    }
}

std::tuple<double, double> get_errors_for_auto(kernel_func f, const std::vector<double> &sample_points,
                                               const std::vector<double> &actual_vals, int order, const void *data,
                                               const std::vector<double> &coeffs, double lb, double ub, double h,
                                               const int n_samples, bool use_rel_error = false) {
    double max_error = 0.0;
    double avg_error = 0.0;

    for (int i = 0; i < n_samples; i++) {
        double x        = sample_points[i];
        double y        = eval_multi(coeffs, x, lb, ub, order, h);
        double actual_y = actual_vals[i];
        double error    = use_rel_error ? std::abs(1.0 - y / actual_y) : std::abs(y - actual_y);
        avg_error += error;
        if (error > max_error) {
            max_error = error;
        }
    }
    avg_error /= n_samples;

    return {avg_error, max_error};
}

std::vector<double> fit_multi_auto(kernel_func f, double lb, double ub, const void *data, int n_poly, double error_tol,
                                   int min_order, int max_order, int n_samples) {
    double ub_safe     = ub - std::numeric_limits<double>::epsilon();
    uint64_t n_epsilon = 1;
    while (ub_safe == ub) {
        ub_safe -= n_epsilon * std::numeric_limits<double>::epsilon();
        n_epsilon *= 2;
    }

    std::vector<double> x = linspaced(n_samples, lb, ub_safe);
    std::vector<double> y(n_samples);
    for (int i = 0; i < n_samples; i++)
        y[i] = f(x[i], data);

    for (int order = min_order; order < max_order; order++) {
        std::vector<double> coeffs(order * n_poly);
        fit_multi(f, lb, ub, order, data, coeffs.data(), n_poly);

        auto [avg_rel_error, max_rel_error] =
            get_errors_for_auto(f, x, y, order, data, coeffs, lb, ub, (ub - lb) / n_poly, n_samples);
        if (avg_rel_error < error_tol) return coeffs;
    }

    return std::vector<double>();
}

void eval_grid(const std::vector<double> &coeffs, double x, double lb, double ub, int order, int n, double *output) {
    const double h      = (ub - lb) / n;
    const double half_h = 0.5 * h;
    const int i         = (x - lb) / h;
    const double dx     = x - (lb + half_h * (2 * i + 1));
    for (int j = 0; j < n; j++)
        output[j] = eval(coeffs.data() + j * order, order, dx);
}

double abs_error(const std::vector<double> &y1, const std::vector<double> &y2) {
    double error = 0.0;
    for (int i = 0; i < y1.size(); i++)
        error += std::abs(y1[i] - y2[i]);

    return error / y1.size();
}
} // namespace spreadkernel::polyfit

TEST_CASE("SPREADKERNEL fits/evals") {
    using namespace spreadkernel::polyfit;
    const double lb       = -0.7;
    const double ub       = 0.8;
    const int order       = 16;
    const int n_samples   = 100;
    const int max_order   = 16;
    const int min_order   = 2;
    const int n_poly      = 7;
    const double h        = (ub - lb) / n_poly;
    const int multi_order = 8;
    const double tol      = 1E-8;

    void *data    = NULL;
    kernel_func f = [](double x, const void *data) {
        return sin(pow(x, 2) * cos(x));
    };

    std::vector<double> coeffs = fit(f, lb, ub, order, data);

    std::vector<double> x = linspaced(n_samples, lb, ub - std::numeric_limits<double>::epsilon());
    std::vector<double> y(n_samples);

    for (int i = 0; i < n_samples; i++)
        y[i] = eval(coeffs.data(), order, x[i]);

    std::vector<double> yraw(n_samples);
    for (int i = 0; i < n_samples; i++)
        yraw[i] = f(x[i], data);

    CHECK(abs_error(y, yraw) < tol);

    std::vector<double> coeffs_multi(multi_order * n_poly);
    fit_multi(f, lb, ub, multi_order, data, coeffs_multi.data(), n_poly);

    {
        std::vector<double> y_multi(n_samples);
        for (int i = 0; i < n_samples; i++)
            y_multi[i] = eval_multi(coeffs_multi, x[i], lb, ub, multi_order, h);
        CHECK(abs_error(y_multi, yraw) < tol);
    }

    std::vector<double> y_grid(n_poly);
    std::vector<double> x_grid_multi = linspaced(n_poly, lb + 0.75 * h, ub - 0.25 * h);
    std::vector<double> y_grid_multi(n_poly);
    eval_grid(coeffs_multi, lb + 1.75 * h, lb, ub, multi_order, n_poly, y_grid.data());
    for (int i = 0; i < n_poly; i++)
        y_grid_multi[i] = f(x_grid_multi[i], data);
    CHECK(abs_error(y_grid, y_grid_multi) < tol);

    auto coeffs_auto = fit_multi_auto(f, lb, ub, data, n_poly, tol, min_order, max_order, n_samples);
    auto auto_order  = coeffs_auto.size() / n_poly;

    std::vector<double> y_auto(n_samples);
    for (int i = 0; i < n_samples; i++)
        y_auto[i] = eval_multi(coeffs_auto, x[i], lb, ub, auto_order, h);
    CHECK(abs_error(y_auto, yraw) < tol);
}

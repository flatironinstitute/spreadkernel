#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

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
    const int i         = (x - lb) / h;
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

void testfit() {
    const double lb = -0.7;
    const double ub = 0.8;
    const int order = 16;
    void *data      = NULL;
    kernel_func f   = [](double x, const void *data) {
        return sin(pow(x, 2) * cos(x));
    };

    Eigen::VectorXd coeffs = fit(f, lb, ub, order, data);

    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(100, lb, ub - std::numeric_limits<double>::epsilon());
    Eigen::VectorXd y = Eigen::VectorXd::Zero(100);
    for (int i = 0; i < 100; i++)
        y(i) = eval(coeffs, x(i));

    Eigen::VectorXd yraw = Eigen::VectorXd::Zero(100);
    for (int i = 0; i < 100; i++)
        yraw(i) = f(x(i), data);

    auto stderr = (y - yraw).array().abs().sum() / 100;
    std::cout << stderr << std::endl;

    const int n_poly             = 7;
    const double h               = (ub - lb) / n_poly;
    const int multi_order        = 8;
    Eigen::VectorXd coeffs_multi = Eigen::VectorXd::Zero(multi_order * n_poly);
    fit_multi(f, lb, ub, multi_order, data, coeffs_multi.data(), n_poly);
    Eigen::VectorXd y_multi = Eigen::VectorXd::Zero(100);
    for (int i = 0; i < 100; i++)
        y_multi(i) = eval_multi(coeffs_multi, x(i), lb, ub, multi_order, h);
    auto stderr_multi = (y_multi - yraw).array().abs().sum() / 100;
    std::cout << stderr_multi << std::endl;
}

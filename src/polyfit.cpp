#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

typedef double (*kernel_func)(double, void *);

Eigen::VectorXd fit(kernel_func f, double lb, double ub, int order, void *data) {
    Eigen::MatrixXd Vandermonde(order, order);
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(order, lb, ub);

    Vandermonde.col(0) = Eigen::VectorXd::Ones(order);
    for (int i_col = 1; i_col < order; i_col++)
        for (int i_row = 0; i_row < order; i_row++)
            Vandermonde(i_row, i_col) = Vandermonde(i_row, i_col - 1) * x(i_row);

    Eigen::VectorXd y(order);
    for (int i = 0; i < order; i++)
        y(i) = f(x(i), data);

    return Vandermonde.lu().solve(y).reverse();
}

double eval(const Eigen::VectorXd &coeffs, double x) {
    double y = coeffs(0);
    for (auto j = 1; j < coeffs.size(); ++j)
        y = std::fma(y, x, coeffs[j]);

    return y;
}

void testfit() {
    double lb     = 0;
    double ub     = 1;
    int order     = 16;
    void *data    = NULL;
    kernel_func f = [](double x, void *data) {
        return sin(pow(x, 2) * cos(x));
    };

    Eigen::VectorXd coeffs = fit(f, lb, ub, order, data);

    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(100, lb, ub);
    Eigen::VectorXd y = Eigen::VectorXd::Zero(100);
    for (int i = 0; i < 100; i++)
        y(i) = eval(coeffs, x(i));

    Eigen::VectorXd yraw = Eigen::VectorXd::Zero(100);
    for (int i = 0; i < 100; i++)
        yraw(i) = f(x(i), data);

    auto stderr = (y - yraw).array().abs().sum() / 100;
    std::cout << stderr << std::endl;
}

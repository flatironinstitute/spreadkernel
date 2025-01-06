#ifndef POLYFIT_H
#define POLYFIT_H

#include <vector>

namespace spreadkernel::polyfit {
using kernel_func = double (*)(double, const void *);

std::vector<double> fit(kernel_func f, double lb, double ub, int order, const void *data);
std::vector<double> fit_multi_auto(kernel_func f, double lb, double ub, const void *data, int n_poly, double error_tol,
                                   int min_order, int max_order, int n_samples);
}

#endif

#ifndef POLYFIT_H
#define POLYFIT_H

#include <vector>

namespace spreadkernel::polyfit {
using kernel_func = double (*)(double, const void *);

template <typename Real>
class Polyfit {
  public:
    Polyfit(kernel_func f, const void *data, Real lb, Real ub, int n_poly, double tol, int min_order, int max_order,
            int n_samples);
    Polyfit() = default;

    Real eval(Real x) const;
    Real operator()(Real x) const { return eval(x); }

  private:
    Real lb{0.0}, ub{0.0};
    int n_poly{0};
    int order{0};
    Real h{0.0};
    Real half_h{0.0};
    Real inv_h{0.0};
    std::vector<Real> coeffs{};
};

} // namespace spreadkernel::polyfit

#endif

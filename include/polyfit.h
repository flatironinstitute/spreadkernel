#ifndef POLYFIT_H
#define POLYFIT_H

#include <array>
#include <vector>

#include <spreadkernel_defs.h>

namespace spreadkernel::polyfit {
using kernel_func = double (*)(double, const void *);

template <typename Real>
using evaluator = void (*)(Real *SPREADKERNEL_RESTRICT,
                           const std::array<std::array<Real, SPREADKERNEL_MAX_WIDTH>, SPREADKERNEL_MAX_ORDER> &,
                           Real);

template <typename Real>
class Polyfit {
  public:
    Polyfit(kernel_func f, const void *data, Real lb, Real ub, int n_poly, double tol, int min_order, int max_order,
            int n_samples);
    Polyfit() = default;

    Real eval(Real x) const;
    void eval(Real x, Real *SPREADKERNEL_RESTRICT y) const;
    void operator()(Real x, Real *SPREADKERNEL_RESTRICT y) const { eval(x, y); }
    Real operator()(Real x) const { return eval(x); }

  private:
    Real lb{0.0}, ub{0.0};
    int width{0};
    int order{0};
    Real h{0.0};
    Real half_h{0.0};
    Real inv_h{0.0};
    std::vector<Real> coeffs_vec{};
    evaluator<Real> vec_eval{nullptr};
    alignas(64) std::array<std::array<Real, SPREADKERNEL_MAX_WIDTH>, SPREADKERNEL_MAX_ORDER> coeffs_arr{{0.0}};
};

} // namespace spreadkernel::polyfit

#endif

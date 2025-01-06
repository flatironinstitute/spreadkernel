#include <iostream>
#include <nanobench.h>
#include <random>
#include <spreadkernel.h>
#include <tuple>
#include <vector>

template<int Ns, typename T>
void spread2d_finufft(const std::vector<T> &ker_x, const std::vector<T> &ker_y, const std::vector<int> &sx,
                      const std::vector<int> &sy, int Nx, int Ny, int M, T *SPREADKERNEL_RESTRICT out) {
    memset(out, 0, Nx * Ny * sizeof(T));
    for (int i = 0; i < M; ++i) {
        for (int yl = 0; yl < Ns; ++yl) {
            const int y = sy[i] + yl;
            for (int xl = 0; xl < Ns; ++xl) {
                const int x = sx[i] + xl;
                out[y * Nx + x] += ker_x[xl] * ker_y[yl];
            }
        }
    }
}

template<int Ns, typename T>
void spread2d_naive_output(const std::vector<T> &ker_x, const std::vector<T> &ker_y, const std::vector<int> &sx,
                           const std::vector<int> &sy, int Nx, int Ny, int M, T *SPREADKERNEL_RESTRICT out) {
    memset(out, 0, Nx * Ny * sizeof(T));

    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            for (int i = 0; i < M; ++i) {
                const int xl = x - sx[i];
                const int yl = y - sy[i];
                if (xl >= 0 && xl < Ns && yl >= 0 && yl < Ns) {
                    out[y * Nx + x] += ker_x[xl] * ker_y[yl];
                }
            }
        }
    }
}

template<int Ns, typename T>
void spread2d_naive_output_wide_kernel(const std::vector<T> &ker_x, const std::vector<T> &ker_y,
                                       const std::vector<int> &sx, const std::vector<int> &sy, int Nx, int Ny, int M,
                                       T *SPREADKERNEL_RESTRICT out) {
    memset(out, 0, Nx * Ny * sizeof(T));

    for (int i = 0; i < M; ++i)
        for (int y = 0; y < Ny; ++y)
            for (int x = 0; x < Nx; ++x)
                out[y * Nx + x] += ker_x[x + i * Nx] * ker_y[y + i * Ny];
}

template<int Ns, typename T>
std::tuple<std::vector<T>, std::vector<T>, std::vector<int>, std::vector<int>> generate_data(
    int Nx, int Ny, int M, bool wide_kernel = false) {
    std::vector<T> ker_x(Ns * M);
    std::vector<T> ker_y(Ns * M);
    if (wide_kernel) {
        ker_x.resize(M * Nx);
        ker_y.resize(M * Ny);
    }
    std::vector<int> sx(M);
    std::vector<int> sy(M);
    std::mt19937 gen(42);
    std::uniform_int_distribution<> startx_dis(0, Nx - 1 - Ns);
    std::uniform_int_distribution<> starty_dis(0, Ny - 1 - Ns);
    std::uniform_real_distribution<T> ker_dis(0, 1);

    const int ker_stride_x = wide_kernel ? Nx : Ns;
    const int ker_stride_y = wide_kernel ? Ny : Ns;
    for (int i = 0; i < M; ++i) {
        sx[i] = startx_dis(gen);
        sy[i] = starty_dis(gen);

        const T x_str     = 2 * ker_dis(gen) / Ns;
        const T y_str     = 2 * ker_dis(gen) / Ns;
        const int start_x = wide_kernel ? sx[i] : 0;
        const int start_y = wide_kernel ? sy[i] : 0;
        for (int j = 0; j < Ns / 2; ++j) {
            ker_x[i * ker_stride_x + j + start_x] = j * x_str;
            ker_y[i * ker_stride_y + j + start_y] = j * y_str;
        }
        for (int j = Ns / 2; j < Ns; ++j) {
            ker_x[i * ker_stride_x + j + start_x] = (Ns - j) * x_str;
            ker_y[i * ker_stride_y + j + start_y] = (Ns - j) * y_str;
        }
    }

    return {ker_x, ker_y, sx, sy};
}

int main() {
    const int Nx                                    = 100;
    const int Ny                                    = 100;
    const int M                                     = 10000;
    constexpr int Ns                                = 8;
    using Real                                      = float;
    auto [ker_x, ker_y, sx, sy]                     = generate_data<Ns, Real>(Nx, Ny, M);
    auto [ker_x_wide, ker_y_wide, sx_wide, sy_wide] = generate_data<Ns, Real>(Nx, Ny, M, true);

    std::vector<Real> out_finufft(Nx * Ny);
    ankerl::nanobench::Bench().run("spread2d_finufft",
                                   [&] { spread2d_finufft<Ns>(ker_x, ker_y, sx, sy, Nx, Ny, M, out_finufft.data()); });

    std::vector<Real> out_naive(Nx * Ny);
    ankerl::nanobench::Bench().run("spread2d_naive_output", [&] {
        std::vector<Real> out(Nx * Ny);
        spread2d_naive_output<Ns>(ker_x, ker_y, sx, sy, Nx, Ny, M, out_naive.data());
    });

    // std::vector<Real> out_naive_wide(Nx * Ny);
    // ankerl::nanobench::Bench().run("spread2d_wide", [&] {
    //     std::vector<Real> out(Nx * Ny);
    //     spread2d_naive_output_wide_kernel<Ns>(ker_x_wide, ker_y_wide, sx_wide, sy_wide, Nx, Ny, M,
    //                                           out_naive_wide.data());
    // });

    for (int i = 0; i < Nx * Ny; ++i) {
        if (out_finufft[i] != out_naive[i]) {
            std::cerr << "Mismatch at naive " << i << ": " << out_finufft[i] << " != " << out_naive[i] << std::endl;
        }

        // if (out_finufft[i] != out_naive_wide[i]) {
        //     std::cerr << "Mismatch at wide " << i << ": " << out_finufft[i] << " != " << out_naive_wide[i]
        //               << std::endl;
        // }
    }

    return 0;
}

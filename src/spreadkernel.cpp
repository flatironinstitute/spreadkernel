#include <array>
#include <chrono>
#include <cstdint>

#include <limits>
#include <polyfit.h>
#include <spreadkernel.h>

#include <doctest/doctest.h>
#include <spdlog/spdlog.h>
#include <xsimd/xsimd.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using BIGINT  = int64_t;
using UBIGINT = uint64_t;
using FLT     = double;

namespace spreadkernel {

class CNTime {
  public:
    void start() {
        initial =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count() *
            1e-6;
    }
    double restart() {
        double delta = elapsedsec();
        start();
        return delta;
    }
    double elapsedsec() {
        std::uint64_t now =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count();
        const double nowsec = now * 1e-6;
        return nowsec - initial;
    }

  private:
    double initial;
};

namespace { // anonymous namespace for internal structs equivalent to declaring everything
            // static

template <class T, uint8_t N>
constexpr auto GetPaddedSIMDWidth() {
    // helper function to get the SIMD width with padding for the given number of elements
    // that minimizes the number of iterations
    return xsimd::make_sized_batch<T, find_optimal_simd_width<T, N>()>::type::size;
}

template <class T, uint8_t ns>
constexpr auto get_padding() {
    // helper function to get the padding for the given number of elements
    // ns is known at compile time, rounds ns to the next multiple of the SIMD width
    // then subtracts ns to get the padding using a bitwise and trick
    // WARING: this trick works only for power of 2s
    // SOURCE: Agner Fog's VCL manual
    constexpr uint8_t width = GetPaddedSIMDWidth<T, ns>();
    return ((ns + width - 1) & (-width)) - ns;
}

template <class T, uint8_t ns>
constexpr auto get_padding_helper(uint8_t runtime_ns) {
    // helper function to get the padding for the given number of elements where ns is
    // known at runtime, it uses recursion to find the padding
    // this allows to avoid having a function with a large number of switch cases
    // as GetPaddedSIMDWidth requires a compile time value
    // it cannot be a lambda function because of the template recursion
    if constexpr (ns < 2) {
        return 0;
    } else {
        if (runtime_ns == ns) {
            return get_padding<T, ns>();
        } else {
            return get_padding_helper<T, ns - 1>(runtime_ns);
        }
    }
}

template <class T>
uint8_t get_padding(uint8_t ns) {
    // return the padding as a function of the number of elements
    // MAX_WIDTH is the maximum number of elements that we can have
    // that's why is hardcoded here
    return get_padding_helper<T, SPREADKERNEL_MAX_WIDTH>(ns);
}

SPREADKERNEL_NEVER_INLINE
void print_subgrid_info(int ndims, BIGINT offset1, BIGINT offset2, BIGINT offset3, UBIGINT padded_size1, UBIGINT size1,
                        UBIGINT size2, UBIGINT size3, UBIGINT M0);
} // namespace

static SPREADKERNEL_ALWAYS_INLINE void set_kernel_args(FLT *args, FLT x, const spreadkernel_opts &opts) noexcept;
static SPREADKERNEL_ALWAYS_INLINE void evaluate_kernel_vector(FLT *ker, FLT *args,
                                                              const spreadkernel_opts &opts) noexcept;

static void spread_subproblem_1d(BIGINT off1, UBIGINT size1, FLT *du0, UBIGINT M0, FLT *kx0, FLT *dd0,
                                 const spreadkernel_opts &opts) noexcept;
template <bool thread_safe>
static void add_wrapped_subgrid(BIGINT offset1, BIGINT offset2, BIGINT offset3, UBIGINT padded_size1, UBIGINT size1,
                                UBIGINT size2, UBIGINT size3, UBIGINT N1, UBIGINT N2, UBIGINT N3,
                                FLT *SPREADKERNEL_RESTRICT data_uniform, const FLT *du0);
static void bin_sort_singlethread(BIGINT *ret, UBIGINT M, const FLT *kx, const FLT *ky, const FLT *kz, UBIGINT N1,
                                  UBIGINT N2, UBIGINT N3, double bin_size_x, double bin_size_y, double bin_size_z,
                                  int debug);
void bin_sort_multithread(BIGINT *ret, UBIGINT M, FLT *kx, FLT *ky, FLT *kz, UBIGINT N1, UBIGINT N2, UBIGINT N3,
                          double bin_size_x, double bin_size_y, double bin_size_z, int debug, int nthr);
static void get_subgrid(BIGINT &offset1, BIGINT &offset2, BIGINT &offset3, BIGINT &padded_size1, BIGINT &size1,
                        BIGINT &size2, BIGINT &size3, UBIGINT M0, FLT *kx0, FLT *ky0, FLT *kz0, int ndims,
                        const spreadkernel_opts &opts);
bool index_sort(BIGINT *sort_indices, UBIGINT N1, UBIGINT N2, UBIGINT N3, UBIGINT M, FLT *kx, FLT *ky, FLT *kz,
                const spreadkernel_opts &opts);
int spread_sorted(const BIGINT *sort_indices, UBIGINT N1, UBIGINT N2, UBIGINT N3,
                  FLT *SPREADKERNEL_RESTRICT data_uniform, UBIGINT M, FLT *SPREADKERNEL_RESTRICT kx,
                  FLT *SPREADKERNEL_RESTRICT ky, FLT *SPREADKERNEL_RESTRICT kz, const FLT *data_nonuniform,
                  const spreadkernel_opts &opts, bool did_sort);

auto fold_rescale(const auto x, const UBIGINT N) noexcept { return x; }

// With a a length-n array, writes out min(a) to lo and max(a) to hi,
// so that all a values lie in [lo,hi].
// If n==0, lo and hi are not finite.
void arrayrange(BIGINT n, FLT *a, FLT *lo, FLT *hi) {
    *lo = INFINITY;
    *hi = -INFINITY;
    for (BIGINT m = 0; m < n; ++m) {
        if (a[m] < *lo) *lo = a[m];
        if (a[m] > *hi) *hi = a[m];
    }
}

// rule for getting number of spreading dimensions from the list of Ns per dim.
// Split out, Barnett 7/26/18
static constexpr uint8_t ndims_from_Ns(const UBIGINT N1, const UBIGINT N2, const UBIGINT N3) {
    return 1 + (N2 > 1) + (N3 > 1);
}

/* This makes a decision whether or not to sort the NU pts (influenced by
   opts.sort), and if yes, calls either single- or multi-threaded bin sort,
   writing reordered index list to sort_indices. If decided not to sort, the
   identity permutation is written to sort_indices.
   The permutation is designed to make RAM access close to contiguous, to
   speed up spreading/interpolation, in the case of disordered NU points.

   Inputs:
    M        - number of input NU points.
    kx,ky,kz - length-M arrays of real coords of NU pts. Domain is [-pi, pi),
                points outside are folded in.
               (only kz used in 1D, only kx and ky used in 2D.)
    N1,N2,N3 - integer sizes of overall box (set N2=N3=1 for 1D, N3=1 for 2D).
               1 = x (fastest), 2 = y (medium), 3 = z (slowest).
    opts     - spreading options struct, see ../include/finufft_spread_opts.h
   Outputs:
    sort_indices - a good permutation of NU points. (User must preallocate
                   to length M.) Ie, kx[sort_indices[j]], j=0,..,M-1, is a good
                   ordering for the x-coords of NU pts, etc.
    returned value - whether a sort was done (1) or not (0).

   Barnett 2017; split out by Melody Shih, Jun 2018. Barnett nthr logic 2024.
*/
bool index_sort(BIGINT *sort_indices, UBIGINT N1, UBIGINT N2, UBIGINT N3, UBIGINT M, FLT *kx, FLT *ky, FLT *kz,
                const spreadkernel_opts &opts) {
    CNTime timer{};
    uint8_t ndims = ndims_from_Ns(N1, N2, N3);
    auto N        = N1 * N2 * N3; // U grid (periodic box) sizes

    // heuristic binning box size for U grid... affects performance:
    double bin_size_x = 16, bin_size_y = 4, bin_size_z = 4;
    // put in heuristics based on cache sizes (only useful for single-thread) ?

    int better_to_sort = !(ndims == 1 && (M > 1000 * N1));  // 1D small-N

    timer.start();                                          // if needed, sort all the NU pts...
    bool did_sort = false;
    auto maxnthr  = MY_OMP_GET_MAX_THREADS();               // used if both below opts default
    if (opts.nthreads > 0) maxnthr = opts.nthreads;         // user nthreads overrides, without limit
    if (opts.sort_threads > 0) maxnthr = opts.sort_threads; // high-priority override, also no limit
    // At this point: maxnthr = the max threads sorting could use
    // (we don't print warning here, since: no showwarn in spread_opts, and finufft
    // already warned about it. spreadinterp-only advanced users will miss a warning)
    if (opts.sort == 1 || (opts.sort == 2 && better_to_sort)) {
        // store a good permutation ordering of all NU pts (dim=1,2 or 3)
        int sort_debug = (opts.debug >= 2);         // show timing output?
        int sort_nthr  = opts.sort_threads;         // 0, or user max # threads for sort
#ifndef _OPENMP
        sort_nthr = 1;                              // if single-threaded lib, override user
#endif
        if (sort_nthr == 0)                         // multithreaded auto choice: when N>>M, one thread is better!
            sort_nthr = (10 * M > N) ? maxnthr : 1; // heuristic
        if (sort_nthr == 1)
            bin_sort_singlethread(sort_indices, M, kx, ky, kz, N1, N2, N3, bin_size_x, bin_size_y, bin_size_z,
                                  sort_debug);
        else // sort_nthr>1, user fixes # threads (>=2)
            bin_sort_multithread(sort_indices, M, kx, ky, kz, N1, N2, N3, bin_size_x, bin_size_y, bin_size_z,
                                 sort_debug, sort_nthr);
        if (opts.debug) printf("\tsorted (%d threads):\t%.3g s\n", sort_nthr, timer.elapsedsec());
        did_sort = true;
    } else {
#pragma omp parallel for num_threads(maxnthr) schedule(static, 1000000)
        for (BIGINT i = 0; i < M; i++) // here omp helps xeon, hinders i7
            sort_indices[i] = i;       // the identity permutation
        if (opts.debug) printf("\tnot sorted (sort=%d): \t%.3g s\n", (int)opts.sort, timer.elapsedsec());
    }
    return did_sort;
}

// --------------------------------------------------------------------------
// Spread NU pts in sorted order to a uniform grid. See spreadinterp() for doc.
int spread_sorted(const BIGINT *sort_indices, UBIGINT N1, UBIGINT N2, UBIGINT N3,
                  FLT *SPREADKERNEL_RESTRICT data_uniform, UBIGINT M, FLT *SPREADKERNEL_RESTRICT kx,
                  FLT *SPREADKERNEL_RESTRICT ky, FLT *SPREADKERNEL_RESTRICT kz, const FLT *data_nonuniform,
                  const spreadkernel_opts &opts, bool did_sort) {
    CNTime timer{};
    const auto ndims = ndims_from_Ns(N1, N2, N3);
    const auto N     = N1 * N2 * N3;             // output array size
    const auto ns    = opts.nspread;             // abbrev. for w, kernel width
    auto nthr        = MY_OMP_GET_MAX_THREADS(); // guess # threads to use to spread
    if (opts.nthreads > 0) nthr = opts.nthreads; // user override, now without limit
#ifndef _OPENMP
    nthr = 1;                                    // single-threaded lib must override user
#endif
    if (opts.debug)
        printf("\tspread %dD (M=%lld; N1=%lld,N2=%lld,N3=%lld), nthr=%d\n", ndims, (long long)M, (long long)N1,
               (long long)N2, (long long)N3, nthr);
    timer.start();
    std::fill(data_uniform, data_uniform + N, 0.0); // zero the output array
    if (opts.debug) printf("\tzero output array\t%.3g s\n", timer.elapsedsec());
    if (M == 0)                                     // no NU pts, we're done
        return 0;

    auto spread_single = (nthr == 1) || (M * 100 < N); // low-density heuristic?
    spread_single      = false;                        // for now
    timer.start();
    if (spread_single) {                               // ------- Basic single-core t1 spreading ------
        for (UBIGINT j = 0; j < M; j++) {
            // *** todo, not urgent
            // ... (question is: will the index wrapping per NU pt slow it down?)
        }
        if (opts.debug) printf("\tt1 simple spreading:\t%.3g s\n", timer.elapsedsec());
    } else { // ------- Fancy multi-core blocked t1 spreading ----
             // Splits sorted inds (jfm's advanced2), could double RAM.
        // choose nb (# subprobs) via used nthreads:
        auto nb = std::min((UBIGINT)nthr, M);            // simply split one subprob per thr...
        if (nb * (BIGINT)opts.max_subproblem_size < M) { // ...or more subprobs to cap size
            nb = 1 + (M - 1) / opts.max_subproblem_size; // int div does
                                                         // ceil(M/opts.max_subproblem_size)
            if (opts.debug) printf("\tcapping subproblem sizes to max of %d\n", opts.max_subproblem_size);
        }
        if (M * 1000 < N) { // low-density heuristic: one thread per NU pt!
            nb = M;
            if (opts.debug) printf("\tusing low-density speed rescue nb=M...\n");
        }
        if (!did_sort && nthr == 1) {
            nb = 1;
            if (opts.debug) printf("\tunsorted nthr=1: forcing single subproblem...\n");
        }
        if (opts.debug && nthr > opts.atomic_threshold)
            printf("\tnthr big: switching add_wrapped OMP from critical to atomic (!)\n");

        std::vector<UBIGINT> brk(nb + 1); // NU index breakpoints defining nb subproblems
        for (int p = 0; p <= nb; ++p)
            brk[p] = (M * p + nb - 1) / nb;

#pragma omp parallel num_threads(nthr)
        {
            // local copies of NU pts and data for each subproblem
            std::vector<FLT> kx0{}, ky0{}, kz0{}, dd0{}, du0{};
#pragma omp for schedule(dynamic, 1)                       // each is big
            for (int isub = 0; isub < nb; isub++) {        // Main loop through the subproblems
                const auto M0 = brk[isub + 1] - brk[isub]; // # NU pts in this subproblem
                // copy the location and data vectors for the nonuniform points
                kx0.resize(M0);
                ky0.resize(M0 * (N2 > 1));
                kz0.resize(M0 * (N3 > 1));
                dd0.resize(M0);                                  // complex strength data
                for (auto j = 0; j < M0; j++) {                  // todo: can avoid this copying?
                    const auto kk = sort_indices[j + brk[isub]]; // NU pt from subprob index list

                    // FIXME: implement wrapping
                    kx0[j] = kx[kk];             // fold_rescale(kx[kk], N1);
                    if (N2 > 1) ky0[j] = ky[kk]; // fold_rescale(ky[kk], N2);
                    if (N3 > 1) kz0[j] = kz[kk]; // fold_rescale(kz[kk], N3);
                    dd0[j] = data_nonuniform[kk];
                }
                // get the subgrid which will include padding by roughly nspread/2
                // get_subgrid sets
                BIGINT offset1, offset2, offset3, padded_size1, size1, size2, size3;
                // sets offsets and sizes
                get_subgrid(offset1, offset2, offset3, padded_size1, size1, size2, size3, M0, kx0.data(), ky0.data(),
                            kz0.data(), ndims, opts);
                if (opts.debug > 1) {
                    print_subgrid_info(ndims, offset1, offset2, offset3, padded_size1, size1, size2, size3, M0);
                }
                // allocate output data for this subgrid
                du0.resize(padded_size1 * size2 * size3); // complex
                // Spread to subgrid without need for bounds checking or wrapping
                if (ndims == 1)
                    spread_subproblem_1d(offset1, padded_size1, du0.data(), M0, kx0.data(), dd0.data(), opts);
                else if (ndims == 2)
                    throw std::runtime_error("2D not implemented yet");
                // spread_subproblem_2d(offset1, offset2, padded_size1, size2, du0.data(), M0, kx0.data(), ky0.data(),
                //                      dd0.data(), opts);
                else
                    throw std::runtime_error("3D not implemented yet");
                // spread_subproblem_3d(offset1, offset2, offset3, padded_size1, size2, size3, du0.data(), M0,
                //                      kx0.data(), ky0.data(), kz0.data(), dd0.data(), opts);

                // do the adding of subgrid to output
                if (nthr > opts.atomic_threshold) { // see above for debug reporting
                    add_wrapped_subgrid<true>(offset1, offset2, offset3, padded_size1, size1, size2, size3, N1, N2, N3,
                                              data_uniform,
                                              du0.data()); // R Blackwell's atomic version
                } else {
#pragma omp critical
                    add_wrapped_subgrid<false>(offset1, offset2, offset3, padded_size1, size1, size2, size3, N1, N2,
                                               N3, data_uniform, du0.data());
                }
            } // end main loop over subprobs
        }
        if (opts.debug) printf("\tt1 fancy spread: \t%.3g s (%ld subprobs)\n", timer.elapsedsec(), nb);
    } // end of choice of which t1 spread type to use
    return 0;
};

///////////////////////////////////////////////////////////////////////////

void setup_spreader(spreadkernel_opts &opts, int dim) {
    if (opts.max_subproblem_size == 0) opts.max_subproblem_size = (dim == 1) ? 10000 : 100000;
    assert(opts.nspread >= SPREADKERNEL_MIN_WIDTH);
    assert(opts.nspread <= SPREADKERNEL_MAX_WIDTH);
    assert(opts.ker);
    assert(opts.grid_delta[0] > 0);
    assert(opts.nspread >= SPREADKERNEL_MIN_WIDTH);
    assert(opts.nspread <= SPREADKERNEL_MAX_WIDTH);

    if (opts.kerevalmeth)
        opts.kerpoly = polyfit::Polyfit(opts.ker, opts.ker_data, opts.grid_delta[0], opts.nspread, opts.eps,
                                        SPREADKERNEL_MIN_WIDTH, SPREADKERNEL_MAX_WIDTH, 100);
}

SPREADKERNEL_ALWAYS_INLINE FLT evaluate_kernel(FLT x, const spreadkernel_opts &opts) {
    if (abs(x) >= opts.kerpoly.ub)
        return 0.0;
    else
        return opts.ker(x, opts.ker_data);
}

template <uint8_t ns>
SPREADKERNEL_ALWAYS_INLINE void set_kernel_args(FLT *args, FLT x) noexcept {
    // Fills vector args[] with kernel arguments x, x+1, ..., x+ns-1.
    // needed for the vectorized kernel eval of Ludvig af K.
    for (int i = 0; i < ns; i++)
        args[i] = x + (FLT)i;
}

/* 1D spreader from nonuniform to uniform subproblem grid, without wrapping.
   FIXME: Ignores kerevalmeth: always uses the polynomial evaluation method.
   Inputs:
   off1 - integer offset of left end of du subgrid from that of overall fine
          periodized output grid {0,1,...N-1}.
   size1 - integer length of output subgrid du
   M - number of NU pts in subproblem
   kx (length M) - are rescaled NU source locations, should lie in
                   [off1+ns/2,off1+size1-1-ns/2] so as kernels stay in bounds
   dd (length M) - source strengths
   Outputs:
   du (length size1 real) - preallocated uniform subgrid array
*/
template <uint8_t ns, bool kerevalmeth>
SPREADKERNEL_NEVER_INLINE void spread_subproblem_1d_kernel(
    const BIGINT off1, const UBIGINT size1, FLT *SPREADKERNEL_RESTRICT du, const UBIGINT M, const FLT *const kx,
    const FLT *const dd, const spreadkernel_opts &opts) noexcept {
    using simd_type                 = PaddedSIMD<FLT, ns>;
    using arch_t                    = typename simd_type::arch_type;
    static constexpr auto alignment = arch_t::alignment();
    static constexpr auto simd_size = simd_type::size;
    static constexpr auto n_parts   = ns / simd_size + (ns % simd_size > 0);
    static constexpr auto tot_size  = n_parts * simd_size;
    static_assert(n_parts > 0, "n_parts must be greater than 0");

    alignas(alignment) std::array<FLT, n_parts * simd_size> ker{0};
    std::fill(du, du + size1, 0);

    const FLT h              = opts.grid_delta[0];        // grid spacing
    const FLT half_h         = 0.5 * h;                   // half grid spacing
    const FLT inv_h          = 1.0 / h;                   // inverse grid spacing
    const FLT ker_half_width = 0.5 * opts.nspread * h;    // half width of the kernel
    const FLT lb1            = off1 * h;                  // left bound of the subgrid
    for (uint64_t i = 0; i < M; i++) {
        const auto dx     = kx[i] - lb1 - ker_half_width; // x location for first kernel eval
        const BIGINT j    = inv_h * (dx + half_h);        // bin index for first kernel eval
        const auto dx_min = dx - j * h;                   // distance to the leftmost grid point in eval
        assert(std::abs(dx_min) <= half_h);

        opts.kerpoly(dx_min, ker.data());         // evaluate the kernel
        auto *SPREADKERNEL_RESTRICT trg = du + j; // restrict helps compiler to vectorize

        const auto dd_pt = simd_type(dd[i]);
        for (uint8_t offset = 0; offset < tot_size; offset += simd_size) {
            const auto ker0  = simd_type::load_aligned(ker.data() + offset);
            const auto du_pt = simd_type::load_unaligned(trg + offset);
            const auto res   = xsimd::fma(ker0, dd_pt, du_pt);
            res.store_unaligned(trg + offset);
        }
    }
}

/* this is a dispatch function that will call the correct kernel based on the ns
  it recursively iterates from SPREADKERNEL_MAX_WIDTH to SPREADKERNEL_MIN_WIDTH
  it generates the following code:
  if (ns == SPREADKERNEL_MAX_WIDTH) {
    if (opts.kerevalmeth) {
      return spread_subproblem_1d_kernel<SPREADKERNEL_MAX_WIDTH, true>(off1, size1, du, M, kx, dd,
      opts);
   } else {
      return spread_subproblem_1d_kernel<SPREADKERNEL_MAX_WIDTH, false>(off1, size1, du, M, kx, dd,
      opts);
  }
  if (ns == SPREADKERNEL_MAX_WIDTH-1) {
    if (opts.kerevalmeth) {
      return spread_subproblem_1d_kernel<SPREADKERNEL_MAX_WIDTH-1, true>(off1, size1, du, M, kx, dd,
      opts);
    } else {
      return spread_subproblem_1d_kernel<SPREADKERNEL_MAX_WIDTH-1, false>(off1, size1, du, M, kx,
      dd, opts);
    }
  }
  ...
  NOTE: using a big SPREADKERNEL_MAX_WIDTH will generate a lot of code
        if SPREADKERNEL_MAX_WIDTH gets too large it will crash the compiler with a compile time
        stack overflow. Older compiler will just throw an internal error without
        providing any useful information on the error.
        This is a known issue with template metaprogramming.
        If you increased SPREADKERNEL_MAX_WIDTH and the code does not compile, try reducing it.
*/
template <uint8_t NS>
static void spread_subproblem_1d_dispatch(const BIGINT off1, const UBIGINT size1, FLT *SPREADKERNEL_RESTRICT du,
                                          const UBIGINT M, const FLT *kx, const FLT *dd,
                                          const spreadkernel_opts &opts) noexcept {
    static_assert(SPREADKERNEL_MIN_WIDTH <= NS && NS <= SPREADKERNEL_MAX_WIDTH,
                  "NS must be in the range (SPREADKERNEL_MIN_WIDTH, SPREADKERNEL_MAX_WIDTH)");
    if constexpr (NS == SPREADKERNEL_MIN_WIDTH) { // Base case
        if (opts.kerevalmeth)
            return spread_subproblem_1d_kernel<SPREADKERNEL_MIN_WIDTH, true>(off1, size1, du, M, kx, dd, opts);
        else {
            return spread_subproblem_1d_kernel<SPREADKERNEL_MIN_WIDTH, false>(off1, size1, du, M, kx, dd, opts);
        }
    } else {
        if (opts.nspread == NS) {
            if (opts.kerevalmeth) {
                return spread_subproblem_1d_kernel<NS, true>(off1, size1, du, M, kx, dd, opts);
            } else {
                return spread_subproblem_1d_kernel<NS, false>(off1, size1, du, M, kx, dd, opts);
            }
        } else {
            return spread_subproblem_1d_dispatch<NS - 1>(off1, size1, du, M, kx, dd, opts);
        }
    }
}

/* spreader from dd (NU) to du (uniform) in 2D without wrapping.
   See above docs/notes for spread_subproblem_2d.
   kx,ky (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in both dims.
   dd (size M complex) are complex source strengths
   du (size size1*size2) is complex uniform output array
   For algoritmic details see spread_subproblem_1d_kernel.
*/
void spread_subproblem_1d(BIGINT off1, UBIGINT size1, FLT *du, UBIGINT M, FLT *kx, FLT *dd,
                          const spreadkernel_opts &opts) noexcept {
    spread_subproblem_1d_dispatch<SPREADKERNEL_MAX_WIDTH>(off1, size1, du, M, kx, dd, opts);
}

template <bool thread_safe>
void add_wrapped_subgrid(BIGINT offset1, BIGINT offset2, BIGINT offset3, UBIGINT padded_size1, UBIGINT size1,
                         UBIGINT size2, UBIGINT size3, UBIGINT N1, UBIGINT N2, UBIGINT N3,
                         FLT *SPREADKERNEL_RESTRICT data_uniform, const FLT *const du0)
/* Add a large subgrid (du0) to output grid (data_uniform),
   with periodic wrapping to N1,N2,N3 box.
   offset1,2,3 give the offset of the subgrid from the lowest corner of output.
   padded_size1,2,3 give the size of subgrid.
   Works in all dims. Thread-safe variant of the above routine,
   using atomic writes (R Blackwell, Nov 2020).
   Merged the thread_safe and the not thread_safe version of the function into one
   (M. Barbone 06/24).
*/
{
    std::vector<BIGINT> o2(size2), o3(size3);
    static auto accumulate = [](FLT &a, FLT b) {
        if constexpr (thread_safe) { // NOLINT(*-branch-clone)
#pragma omp atomic
            a += b;
        } else {
            a += b;
        }
    };

    BIGINT y = offset2, z = offset3; // fill wrapped ptr lists in slower dims y,z...
    for (int i = 0; i < size2; ++i) {
        if (y < 0) y += BIGINT(N2);
        if (y >= N2) y -= BIGINT(N2);
        o2[i] = y++;
    }
    for (int i = 0; i < size3; ++i) {
        if (z < 0) z += BIGINT(N3);
        if (z >= N3) z -= BIGINT(N3);
        o3[i] = z++;
    }
    UBIGINT nlo = (offset1 < 0) ? -offset1 : 0;                      // # wrapping below in x
    UBIGINT nhi = (offset1 + size1 > N1) ? offset1 + size1 - N1 : 0; // " above in x
    // this triple loop works in all dims
    for (int dz = 0; dz < size3; dz++) {                                                  // use ptr lists in each axis
        const auto oz = N1 * N2 * o3[dz];                                                 // offset due to z (0 in <3D)
        for (int dy = 0; dy < size2; dy++) {
            const auto oy                   = N1 * o2[dy] + oz;                           // off due to y & z (0 in 1D)
            auto *SPREADKERNEL_RESTRICT out = data_uniform + 2 * oy;
            const auto in                   = du0 + 2 * padded_size1 * (dy + size2 * dz); // ptr to subgrid array
            auto o                          = 2 * (offset1 + N1);                         // 1d offset for output
            for (auto j = 0; j < 2 * nlo; j++) { // j is really dx/2 (since re,im parts)
                accumulate(out[j + o], in[j]);
            }
            o = 2 * offset1;
            for (auto j = 2 * nlo; j < 2 * (size1 - nhi); j++) {
                accumulate(out[j + o], in[j]);
            }
            o = 2 * (offset1 - N1);
            for (auto j = 2 * (size1 - nhi); j < 2 * size1; j++) {
                accumulate(out[j + o], in[j]);
            }
        }
    }
}

void bin_sort_singlethread(BIGINT *ret, const UBIGINT M, const FLT *kx, const FLT *ky, const FLT *kz, const UBIGINT N1,
                           const UBIGINT N2, const UBIGINT N3, const double bin_size_x, const double bin_size_y,
                           const double bin_size_z, const int debug)
/* Returns permutation of all nonuniform points with good RAM access,
 * ie less cache misses for spreading, in 1D, 2D, or 3D. Single-threaded version
 *
 * This is achieved by binning into cuboids (of given bin_size within the
 * overall box domain), then reading out the indices within
 * these bins in a Cartesian cuboid ordering (x fastest, y med, z slowest).
 * Finally the permutation is inverted, so that the good ordering is: the
 * NU pt of index ret[0], the NU pt of index ret[1],..., NU pt of index ret[M-1]
 *
 * Inputs: M - number of input NU points.
 *         kx,ky,kz - length-M arrays of real coords of NU pts in [-pi, pi).
 *                    Points outside this range are folded into it.
 *         N1,N2,N3 - integer sizes of overall box (N2=N3=1 for 1D, N3=1 for 2D)
 *         bin_size_x,y,z - what binning box size to use in each dimension
 *                    (in rescaled coords where ranges are [0,Ni] ).
 *                    For 1D, only bin_size_x is used; for 2D, it & bin_size_y.
 * Output:
 *         writes to ret a vector list of indices, each in the range 0,..,M-1.
 *         Thus, ret must have been preallocated for M BIGINTs.
 *
 * Notes: I compared RAM usage against declaring an internal vector and passing
 * back; the latter used more RAM and was slower.
 * Avoided the bins array, as in JFM's spreader of 2016,
 * tidied up, early 2017, Barnett.
 * Timings (2017): 3s for M=1e8 NU pts on 1 core of i7; 5s on 1 core of xeon.
 * Simplified by Martin Reinecke, 6/19/23 (no apparent effect on speed).
 */
{
    const auto isky = (N2 > 1), iskz = (N3 > 1); // ky,kz avail? (cannot access if not)
    // here the +1 is needed to allow round-off error causing i1=N1/bin_size_x,
    // for kx near +pi, ie foldrescale gives N1 (exact arith would be 0 to N1-1).
    // Note that round-off near kx=-pi stably rounds negative to i1=0.
    const auto nbins1         = BIGINT(FLT(N1) / bin_size_x + 1);
    const auto nbins2         = isky ? BIGINT(FLT(N2) / bin_size_y + 1) : 1;
    const auto nbins3         = iskz ? BIGINT(FLT(N3) / bin_size_z + 1) : 1;
    const auto nbins          = nbins1 * nbins2 * nbins3;
    const auto inv_bin_size_x = FLT(1.0 / bin_size_x);
    const auto inv_bin_size_y = FLT(1.0 / bin_size_y);
    const auto inv_bin_size_z = FLT(1.0 / bin_size_z);
    // count how many pts in each bin
    std::vector<BIGINT> counts(nbins, 0);

    for (auto i = 0; i < M; i++) {
        // find the bin index in however many dims are needed
        const auto i1  = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
        const auto i2  = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
        const auto i3  = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
        const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
        ++counts[bin];
    }

    // compute the offsets directly in the counts array (no offset array)
    BIGINT current_offset = 0;
    for (BIGINT i = 0; i < nbins; i++) {
        BIGINT tmp = counts[i];
        counts[i]  = current_offset; // Reinecke's cute replacement of counts[i]
        current_offset += tmp;
    } // (counts now contains the index offsets for each bin)

    for (auto i = 0; i < M; i++) {
        // find the bin index (again! but better than using RAM)
        const auto i1    = BIGINT(fold_rescale(kx[i], N1) * inv_bin_size_x);
        const auto i2    = isky ? BIGINT(fold_rescale(ky[i], N2) * inv_bin_size_y) : 0;
        const auto i3    = iskz ? BIGINT(fold_rescale(kz[i], N3) * inv_bin_size_z) : 0;
        const auto bin   = i1 + nbins1 * (i2 + nbins2 * i3);
        ret[counts[bin]] = BIGINT(i); // fill the inverse map on the fly
        ++counts[bin];                // update the offsets
    }
}

void bin_sort_multithread(BIGINT *ret, UBIGINT M, FLT *kx, FLT *ky, FLT *kz, UBIGINT N1, UBIGINT N2, UBIGINT N3,
                          double bin_size_x, double bin_size_y, double bin_size_z, int debug, int nthr)
/* Mostly-OpenMP'ed version of bin_sort.
   For documentation see: bin_sort_singlethread.
   Caution: when M (# NU pts) << N (# U pts), is SLOWER than single-thread.
   Originally by Barnett 2/8/18
   Explicit #threads control argument 7/20/20.
   Improved by Martin Reinecke, 6/19/23 (up to 50% faster at 1 thr/core).
   Todo: if debug, print timing breakdowns.
 */
{
    bool isky = (N2 > 1), iskz = (N3 > 1);                // ky,kz avail? (cannot access if not)
    UBIGINT nbins1 = N1 / bin_size_x + 1, nbins2, nbins3; // see above note on why +1
    nbins2         = isky ? N2 / bin_size_y + 1 : 1;
    nbins3         = iskz ? N3 / bin_size_z + 1 : 1;
    UBIGINT nbins  = nbins1 * nbins2 * nbins3;
    if (nthr == 0)                       // should never happen in spreadinterp use
        fprintf(stderr, "[%s] nthr (%d) must be positive!\n", __func__, nthr);
    int nt = std::min(M, UBIGINT(nthr)); // handle case of less points than threads
    std::vector<UBIGINT> brk(nt + 1);    // list of start NU pt indices per thread

    // distribute the NU pts to threads once & for all...
    for (int t = 0; t <= nt; ++t)
        brk[t] = (UBIGINT)(0.5 + M * t / (double)nt); // start index for t'th chunk

    // set up 2d array (nthreads * nbins), just its pointers for now
    // (sub-vectors will be initialized later)
    std::vector<std::vector<UBIGINT>> counts(nt);

#pragma omp parallel num_threads(nt)
    {                                    // parallel binning to each thread's count. Block done once per thread
        int t = MY_OMP_GET_THREAD_NUM(); // (we assume all nt threads created)
        auto &my_counts(counts[t]);      // name for counts[t]
        my_counts.resize(nbins, 0);      // allocate counts[t], now in parallel region
        for (auto i = brk[t]; i < brk[t + 1]; i++) {
            // find the bin index in however many dims are needed
            BIGINT i1 = fold_rescale(kx[i], N1) / bin_size_x, i2 = 0, i3 = 0;
            if (isky) i2 = fold_rescale(ky[i], N2) / bin_size_y;
            if (iskz) i3 = fold_rescale(kz[i], N3) / bin_size_z;
            const auto bin = i1 + nbins1 * (i2 + nbins2 * i3);
            ++my_counts[bin]; // no clash btw threads
        }
    }

    // inner sum along both bin and thread (inner) axes to get global offsets
    UBIGINT current_offset = 0;
    for (UBIGINT b = 0; b < nbins; ++b) // (not worth omp)
        for (int t = 0; t < nt; ++t) {
            UBIGINT tmp  = counts[t][b];
            counts[t][b] = current_offset;
            current_offset += tmp;
        } // counts[t][b] is now the index offset as if t ordered fast, b slow

#pragma omp parallel num_threads(nt)
    {
        int t = MY_OMP_GET_THREAD_NUM();
        auto &my_counts(counts[t]);
        for (UBIGINT i = brk[t]; i < brk[t + 1]; i++) {
            // find the bin index (again! but better than using RAM)
            UBIGINT i1 = fold_rescale(kx[i], N1) / bin_size_x, i2 = 0, i3 = 0;
            if (isky) i2 = fold_rescale(ky[i], N2) / bin_size_y;
            if (iskz) i3 = fold_rescale(kz[i], N3) / bin_size_z;
            UBIGINT bin         = i1 + nbins1 * (i2 + nbins2 * i3);
            ret[my_counts[bin]] = i; // inverse is offset for this NU pt and thread
            ++my_counts[bin];        // update the offsets; no thread clash
        }
    }
}

void get_subgrid(BIGINT &offset1, BIGINT &offset2, BIGINT &offset3, BIGINT &padded_size1, BIGINT &size1, BIGINT &size2,
                 BIGINT &size3, UBIGINT M, FLT *kx, FLT *ky, FLT *kz, int ndims, const spreadkernel_opts &opts)
/* Writes out the integer offsets and sizes of a "subgrid" (cuboid subset of
   Z^ndims) large enough to enclose all of the nonuniform points with
   (non-periodic) padding of half the kernel width ns to each side in
   each relevant dimension.

 Inputs:
   M - number of nonuniform points, ie, length of kx array (and ky if ndims>1,
       and kz if ndims>2)
   kx,ky,kz - coords of nonuniform points (ky only read if ndims>1,
              kz only read if ndims>2). To be useful for spreading, they are
              assumed to be in [0,Nj] for dimension j=1,..,ndims.
   ns - (positive integer) spreading kernel width.
   ndims - space dimension (1,2, or 3).

 Outputs:
   offset1,2,3 - left-most coord of cuboid in each dimension (up to ndims)
   padded_size1,2,3   - size of cuboid in each dimension.
                 Thus the right-most coord of cuboid is offset+size-1.
   Returns offset 0 and size 1 for each unused dimension (ie when ndims<3);
   this is required by the calling code.

 Example:
      inputs:
          ndims=1, M=2, kx[0]=0.2, ks[1]=4.9, ns=3
      outputs:
          offset1=-1 (since kx[0] spreads to {-1,0,1}, and -1 is the min)
          padded_size1=8 (since kx[1] spreads to {4,5,6}, so subgrid is {-1,..,6}
                   hence 8 grid points).
 Notes:
   1) Works in all dims 1,2,3.
   2) Rounding of the kx (and ky, kz) to the grid is tricky and must match the
   rounding step used in spread_subproblem_{1,2,3}d. Namely, the ceil of
   (the NU pt coord minus ns/2) gives the left-most index, in each dimension.
   This being done consistently is crucial to prevent segfaults in subproblem
   spreading. This assumes that max() and ceil() commute in the floating pt
   implementation.
   Originally by J Magland, 2017. AHB realised the rounding issue in
   6/16/17, but only fixed a rounding bug causing segfault in (highly
   inaccurate) single-precision with N1>>1e7 on 11/30/20.
   3) Requires O(M) RAM reads to find the k array bnds. Almost negligible in
   tests.
*/
{
    const FLT ns2    = (FLT)opts.nspread / 2;
    const FLT inv_dx = 1.0 / opts.grid_delta[0];             // inverse grid spacing
    FLT min_kx, max_kx;                                      // 1st (x) dimension: get min/max of nonuniform points
    arrayrange(M, kx, &min_kx, &max_kx);
    offset1      = (BIGINT)std::ceil(inv_dx * min_kx - ns2); // min index touched by kernel
    size1        = (BIGINT)std::ceil(inv_dx * max_kx - ns2) - offset1 + opts.nspread; // int(ceil) first!
    padded_size1 = size1 + get_padding<FLT>(opts.nspread);
    if (ndims > 1) {
        FLT min_ky, max_ky;                          // 2nd (y) dimension: get min/max of nonuniform points
        const FLT inv_dy = 1.0 / opts.grid_delta[1]; // inverse grid spacing
        arrayrange(M, ky, &min_ky, &max_ky);
        offset2 = (BIGINT)std::ceil(min_ky * inv_dy - ns2);
        size2   = (BIGINT)std::ceil(max_ky * inv_dy - ns2) - offset2 + opts.nspread;
    } else {
        offset2 = 0;
        size2   = 1;
    }
    if (ndims > 2) {
        FLT min_kz, max_kz;                          // 3rd (z) dimension: get min/max of nonuniform points
        const FLT inv_dz = 1.0 / opts.grid_delta[2]; // inverse grid spacing
        arrayrange(M, kz, &min_kz, &max_kz);
        offset3 = (BIGINT)std::ceil(inv_dz * min_kz - ns2);
        size3   = (BIGINT)std::ceil(inv_dz * max_kz - ns2) - offset3 + opts.nspread;
    } else {
        offset3 = 0;
        size3   = 1;
    }
}

template <uint8_t ns, uint8_t kerevalmeth, class T, class simd_type, typename... V>
auto ker_eval(FLT *SPREADKERNEL_RESTRICT ker, const spreadkernel_opts &opts, const V... elems) noexcept {
    /* Utility function that allows to move the kernel evaluation outside the spreader for
       clarity
       Inputs are:
       ns = kernel width
       kerevalmeth = kernel evaluation method
       T = (single or double precision) type of the kernel
       simd_type = xsimd::batch for Horner
       vectorization (default is the optimal simd size)
       finufft_spread_opts as Horner needs
       the oversampling factor
       elems = kernel arguments
       Examples usage is
       ker_eval<ns,kerevalmeth>(opts, x, y, z) // for 3D or
       ker_eval<ns, kerevalmeth>(opts, x, y) // for 2D or
       ker_eval<ns, kerevalmeth>(opts, x) // for 1D
     */
    const std::array inputs{elems...};
    // compile time loop, no performance overhead
    for (auto i = 0; i < sizeof...(elems); ++i) {
        // compile time branch no performance overhead
        if constexpr (kerevalmeth == 0) {
            alignas(simd_type::arch_type::alignment()) std::array<T, SPREADKERNEL_MAX_WIDTH> kernel_args{};
            set_kernel_args<ns>(kernel_args.data(), inputs[i]);
            for (auto j = 0; j < ns; ++j)
                evaluate_kernel(kernel_args[j], opts);
        }
        if constexpr (kerevalmeth == 1) {
            // FIXME: fill in horner eval here
        }
    }
    return ker;
}

namespace {

void print_subgrid_info(int ndims, BIGINT offset1, BIGINT offset2, BIGINT offset3, UBIGINT padded_size1, UBIGINT size1,
                        UBIGINT size2, UBIGINT size3, UBIGINT M0) {
    printf("size1 %ld, padded_size1 %ld\n", size1, padded_size1);
    switch (ndims) {
    case 1:
        printf("\tsubgrid: off %lld\t siz %lld\t #NU %lld\n", (long long)offset1, (long long)padded_size1,
               (long long)M0);
        break;
    case 2:
        printf("\tsubgrid: off %lld,%lld\t siz %lld,%lld\t #NU %lld\n", (long long)offset1, (long long)offset2,
               (long long)padded_size1, (long long)size2, (long long)M0);
        break;
    case 3:
        printf("\tsubgrid: off %lld,%lld,%lld\t siz %lld,%lld,%lld\t #NU %lld\n", (long long)offset1,
               (long long)offset2, (long long)offset3, (long long)padded_size1, (long long)size2, (long long)size3,
               (long long)M0);
        break;
    default:
        printf("Invalid number of dimensions: %d\n", ndims);
        break;
    }
}
} // namespace
} // namespace spreadkernel

extern "C" {
int spread_kernel_init(UBIGINT N1, UBIGINT N2, UBIGINT N3, spreadkernel_opts *opts) {
    spdlog::info("Initializing spread kernel with N1: {}, N2: {}, N3: {}", N1, N2, N3);
    spreadkernel::setup_spreader(*opts, spreadkernel::ndims_from_Ns(N1, N2, N3));
    return SPREADKERNEL_SUCCESS;
}

int spread_kernel(UBIGINT N1, UBIGINT N2, UBIGINT N3, FLT *data_uniform, UBIGINT M, FLT *kx, FLT *ky, FLT *kz,
                  FLT *data_nonuniform, spreadkernel_opts *opts) {
    std::unique_ptr<BIGINT[]> sort_indices(new BIGINT[M]);
    if (!sort_indices) {
        fprintf(stderr, "%s failed to allocate sort_indices!\n", __func__);
        return SPREADKERNEL_ERR_SPREAD_ALLOC;
    }
    auto did_sort = spreadkernel::index_sort(sort_indices.get(), N1, N2, N3, M, kx, ky, kz, *opts);
    spreadkernel::spread_sorted(sort_indices.get(), N1, N2, N3, data_uniform, M, kx, ky, kz, data_nonuniform, *opts,
                                did_sort);

    return SPREADKERNEL_SUCCESS;
}
}

TEST_CASE("SPREADKERNEL setup spreader") {
    spreadkernel_opts opts;
    UBIGINT N1 = 100, N2 = 100, N3 = 100;
    std::fill(opts.grid_delta, opts.grid_delta + 3, 1.0);
    opts.nspread = 6;
    opts.eps     = 1e-7;
    opts.ker     = [](double x, const void *) {
        return exp(-(x * x));
    };

    spread_kernel_init(N1, N2, N3, &opts);

    CHECK(std::abs(opts.kerpoly.eval(1.2) - opts.ker(1.2, nullptr)) < opts.eps);
}

TEST_CASE("SPREADKERNEL 1d subproblem") {
    spreadkernel_opts opts;
    const UBIGINT N1 = 100, N2 = 1, N3 = 1;
    std::fill(opts.grid_delta, opts.grid_delta + 3, 1.3);
    opts.kerevalmeth = SPREADKERNEL_EVAL_HORNER_DIRECT;
    opts.nspread     = 5;
    opts.eps         = 1e-7;
    opts.ker         = [](double x, const void *) {
        return exp(-(x * x));
    };

    // Initialization required for all subcases
    spread_kernel_init(N1, N2, N3, &opts);
    REQUIRE(opts.kerpoly.order);

    constexpr auto simd_size = xsimd::simd_type<FLT>::size;
    std::vector<FLT> outgrid_split_eval(N1), outgrid_single_eval(N1);

    // Exact center of the kernel. On center gridpoint when nspread is odd
    FLT test_x[]   = {0.5 * opts.nspread * opts.grid_delta[0], 1.5 * opts.nspread * opts.grid_delta[0]};
    FLT test_str[] = {1.32, 1.32};
    // spread with no offset about center gridpoint
    spreadkernel::spread_subproblem_1d(0, N1, outgrid_split_eval.data(), 1, test_x, test_str, opts);
    // spread with nspread offset about offset center gridpoint
    spreadkernel::spread_subproblem_1d(opts.nspread, N1 - opts.nspread, outgrid_split_eval.data() + opts.nspread, 1,
                                       &test_x[1], &test_str[1], opts);

    CHECK(std::abs(outgrid_split_eval[2] - test_str[0]) < 1E-7);
    CHECK(std::abs(outgrid_split_eval[2 + opts.nspread] - test_str[1]) < 1E-7);

    // Single eval for both points. Should have identical results to 'split' eval
    spreadkernel::spread_subproblem_1d(0, N1, outgrid_single_eval.data(), 2, test_x, test_str, opts);
    for (int i = 0; i < 2 * opts.nspread; i++)
        CHECK(
            1.0 - std::abs(outgrid_split_eval[i] / outgrid_single_eval[i]) <= 5 * std::numeric_limits<FLT>::epsilon());

    std::vector<FLT> outgrid_full(N1);
    spread_kernel(N1, N2, N3, outgrid_full.data(), 2, test_x, nullptr, nullptr, test_str, &opts);
    for (int i = 0; i < 2 * opts.nspread; i++)
        CHECK(1.0 - std::abs(outgrid_single_eval[i] / outgrid_full[i]) <= 5 * std::numeric_limits<FLT>::epsilon());
}

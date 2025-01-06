#ifndef SPREADKERNEL_H
#define SPREADKERNEL_H

#include <vector>

#ifdef _OPENMP
#include <omp.h>
// point to actual omp utils
static inline int MY_OMP_GET_NUM_THREADS() { return omp_get_num_threads(); }
static inline int MY_OMP_GET_MAX_THREADS() { return omp_get_max_threads(); }
static inline int MY_OMP_GET_THREAD_NUM() { return omp_get_thread_num(); }
static inline void MY_OMP_SET_NUM_THREADS(int x) { omp_set_num_threads(x); }
#else
// non-omp safe dummy versions of omp utils...
static inline int MY_OMP_GET_NUM_THREADS() { return 1; }
static inline int MY_OMP_GET_MAX_THREADS() { return 1; }
static inline int MY_OMP_GET_THREAD_NUM() { return 0; }
static inline void MY_OMP_SET_NUM_THREADS(int) {}
#endif

#if defined(_MSC_VER)
#define SPREADKERNEL_ALWAYS_INLINE __forceinline inline
#define SPREADKERNEL_NEVER_INLINE  __declspec(noinline)
#define SPREADKERNEL_RESTRICT      __restrict
#define SPREADKERNEL_UNREACHABLE   __assume(0)
#define SPREADKERNEL_UNLIKELY(x)   (x)
#define SPREADKERNEL_LIKELY(x)     (x)
#elif defined(__GNUC__) || defined(__clang__)
#define SPREADKERNEL_ALWAYS_INLINE __attribute__((always_inline)) inline
#define SPREADKERNEL_NEVER_INLINE  __attribute__((noinline))
#define SPREADKERNEL_RESTRICT      __restrict__
#define SPREADKERNEL_UNREACHABLE   __builtin_unreachable()
#define SPREADKERNEL_UNLIKELY(x)   __builtin_expect(!!(x), 0)
#define SPREADKERNEL_LIKELY(x)     __builtin_expect(!!(x), 1)
#else
#define SPREADKERNEL_ALWAYS_INLINE inline
#define SPREADKERNEL_NEVER_INLINE
#define SPREADKERNEL_RESTRICT
#define SPREADKERNEL_UNREACHABLE
#define SPREADKERNEL_UNLIKELY(x) (x)
#define SPREADKERNEL_LIKELY(x)   (x)
#endif

enum {
    SPREADKERNEL_SUCCESS                    = 0,
    SPREADKERNEL_WARN_EPS_TOO_SMALL         = 1,
    SPREADKERNEL_ERR_MAXNALLOC              = 2,
    SPREADKERNEL_ERR_SPREAD_BOX_SMALL       = 3,
    SPREADKERNEL_ERR_SPREAD_PTS_OUT_RANGE   = 4, // DEPRECATED
    SPREADKERNEL_ERR_SPREAD_ALLOC           = 5,
    SPREADKERNEL_ERR_SPREAD_DIR             = 6,
    SPREADKERNEL_ERR_UPSAMPFAC_TOO_SMALL    = 7,
    SPREADKERNEL_ERR_HORNER_WRONG_BETA      = 8,
    SPREADKERNEL_ERR_NTRANS_NOTVALID        = 9,
    SPREADKERNEL_ERR_TYPE_NOTVALID          = 10,
    SPREADKERNEL_ERR_ALLOC                  = 11,
    SPREADKERNEL_ERR_DIM_NOTVALID           = 12,
    SPREADKERNEL_ERR_SPREAD_THREAD_NOTVALID = 13,
    SPREADKERNEL_ERR_NDATA_NOTVALID         = 14,
    SPREADKERNEL_ERR_CUDA_FAILURE           = 15,
    SPREADKERNEL_ERR_PLAN_NOTVALID          = 16,
    SPREADKERNEL_ERR_METHOD_NOTVALID        = 17,
    SPREADKERNEL_ERR_BINSIZE_NOTVALID       = 18,
    SPREADKERNEL_ERR_INSUFFICIENT_SHMEM     = 19,
    SPREADKERNEL_ERR_NUM_NU_PTS_INVALID     = 20,
    SPREADKERNEL_ERR_INVALID_ARGUMENT       = 21,
    SPREADKERNEL_ERR_LOCK_FUNS_INVALID      = 22
};

typedef double (*spreadkernel_kernel_func)(double, const void *);

enum spreadkernel_eval_method : int {
    SPREADKERNEL_EVAL_DIRECT     = 0, // direct, for debugging
    SPREADKERNEL_EVAL_HORNER_VEC = 1, // Horner polynomial evaluation, vectorized over grid points. Fastest. Default.
    SPREADKERNEL_EVAL_HORNER_DIRECT = 2, // Horner polynomial evaluation, one evaluation per grid point, for debugging
};

enum spreadkernel_sort_strategy : int {
    SPREADKERNEL_SORT_NONE   = 0, // no sorting
    SPREADKERNEL_SORT_SORTED = 1, // sorted non-uniform pts
    SPREADKERNEL_SORT_AUTO   = 2, // heuristic choice
};

typedef struct spreadkernel_opts {
    int nspread             = 0;                            // w, the kernel width in grid pts
    int sort                = SPREADKERNEL_SORT_AUTO;       // 0: don't sort NU pts, 1: do, 2: heuristic choice
    int kerevalmeth         = SPREADKERNEL_EVAL_HORNER_VEC; // kernel evaluation method (see spreadkernel_eval_method)
    int nthreads            = 0;                            // # threads for spreading (0: use max avail)
    int sort_threads        = 0;                            // # threads for sort (0: auto-choice up to nthreads)
    int max_subproblem_size = 0;                            // # pts per t1 subprob; sets extra RAM per thread. 0: auto
    int debug               = 0;                            // 0: silent, 1: small text output, 2: verbose
    int atomic_threshold    = 10;              // num threads before switching spread_sorted to using atomic ops
    double lower_bounds[3]  = {0.0, 0.0, 0.0}; // lower bounds of the uniform grid
    double grid_delta[3]    = {0.0, 0.0, 0.0}; // grid spacing
    int periodicity[3]      = {0, 0, 0};       // 0: no, 1: yes
    double ker_half_width   = 0.0;             // half the kernel width in real units
    spreadkernel_kernel_func ker = nullptr;    // ptr to the kernel function
    void *ker_data               = nullptr;    // ptr to the kernel data (e.g. constants to modify your kernel func)
    double eps                   = 1e-7;       // error tolerance for kernel approximation
    std::vector<double> coeffs;                // coefficients for Horner polynomial
} spreadkernel_opts;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif

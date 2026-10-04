#ifndef __LBFGS_H__
#define __LBFGS_H__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
    enum {
        LBFGS_SUCCESS = 0,
        LBFGS_CONVERGENCE = 0,
        LBFGS_STOP,
        LBFGS_ALREADY_MINIMIZED,
        LBFGSERR_UNKNOWNERROR = -1024,
        LBFGSERR_LOGICERROR,
        LBFGSERR_OUTOFMEMORY,
        LBFGSERR_CANCELED,
        LBFGSERR_INVALID_N,
        LBFGSERR_INVALID_N_SSE,
        LBFGSERR_INVALID_X_SSE,
        LBFGSERR_INVALID_EPSILON,
        LBFGSERR_INVALID_TESTPERIOD,
        LBFGSERR_INVALID_DELTA,
        LBFGSERR_INVALID_LINESEARCH,
        LBFGSERR_INVALID_MINSTEP,
        LBFGSERR_INVALID_MAXSTEP,
        LBFGSERR_INVALID_FTOL,
        LBFGSERR_INVALID_WOLFE,
        LBFGSERR_INVALID_GTOL,
        LBFGSERR_INVALID_XTOL,
        LBFGSERR_INVALID_MAXLINESEARCH,
        LBFGSERR_INVALID_ORTHANTWISE,
        LBFGSERR_INVALID_ORTHANTWISE_START,
        LBFGSERR_INVALID_ORTHANTWISE_END,
        LBFGSERR_OUTOFINTERVAL,
        LBFGSERR_INCORRECT_TMINMAX,
        LBFGSERR_ROUNDING_ERROR,
        LBFGSERR_MINIMUMSTEP,
        LBFGSERR_MAXIMUMSTEP,
        LBFGSERR_MAXIMUMLINESEARCH,
        LBFGSERR_MAXIMUMITERATION,
        LBFGSERR_WIDTHTOOSMALL,
        LBFGSERR_INVALIDPARAMETERS,
        LBFGSERR_INCREASEGRADIENT,
    };

    enum {
        LBFGS_LINESEARCH_DEFAULT = 0,
        LBFGS_LINESEARCH_MORETHUENTE = 0,
        LBFGS_LINESEARCH_BACKTRACKING_ARMIJO = 1,
        LBFGS_LINESEARCH_BACKTRACKING = 2,
        LBFGS_LINESEARCH_BACKTRACKING_WOLFE = 2,
        LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 3,
    };

    struct eval_data {
        float* b;
        float* Axb2;
        float* x_copy;
        const int* ri_x;
        const int* ri_y;
        int m;
        int rows;
        int cols;
        int full_rows;   // > 0 only for the subsampled-chroma solver: full-res plane dims
        int full_cols;
        float tv_lambda = 0.0f;  // > 0 enables total-variation fusion in evaluate()

    };

    typedef struct {
        int             m;
        float epsilon;
        int             past;
        float delta;
        int             max_iterations;
        int             linesearch;
        int             max_linesearch;
        float min_step;
        float max_step;
        float ftol;
        float wolfe;
        float gtol;
        float xtol;
        float orthantwise_c;
        int             orthantwise_start;
        int             orthantwise_end;
    } lbfgs_parameter_t;

    typedef float(*lbfgs_evaluate_t)(
        void* instance,
        const float* x,
        eval_data data,
        float* g,
        const int n,
        const float step
        );

    typedef int (*lbfgs_progress_t)(
        void* instance,
        const float* x,
        const float* g,
        const float fx,
        const float xnorm,
        const float gnorm,
        const float step,
        int n,
        int k,
        int ls
        );

    int lbfgs(
        int n,
        float* x,
        eval_data data,
        float* ptr_fx,
        lbfgs_evaluate_t proc_evaluate,
        lbfgs_progress_t proc_progress,
        void* instance,
        lbfgs_parameter_t* param
    );

    void lbfgs_parameter_init(lbfgs_parameter_t* param);

    float* lbfgs_malloc(int n);

    void lbfgs_free(float* x);

    const char* lbfgs_strerror(int err);


#endif/*__cplusplus*/

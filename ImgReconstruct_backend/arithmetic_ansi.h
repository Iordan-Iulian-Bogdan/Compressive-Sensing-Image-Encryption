/*
 *      ANSI C implementation of vector operations.
 *
 * Copyright (c) 2007-2010 Naoaki Okazaki
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

 /* $Id$ */

#include <stdlib.h>
#include <memory.h>
#include <intrin.h>

#define fsigndiff(x, y) (*(x) * (*(y) / fabs(*(y))) < 0.)


inline static void* vecalloc(size_t size)
{
    void* memblock = malloc(size);
    if (memblock) {
        memset(memblock, 0, size);
    }
    return memblock;
}

inline static void vecfree(void* memblock)
{
    free(memblock);
}

inline static void vecset(float* x, const float c, const int n) {
    __m256 vec = _mm256_set1_ps(c);
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        _mm256_store_ps(&x[i], vec);
    }
    for (; i < n; ++i) {
        x[i] = c;
    }
}

inline static void veccpy(float* y, const float* x, const int n) {
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        _mm256_store_ps(&y[i], _mm256_load_ps(&x[i]));
    }
    for (; i < n; ++i) {
        y[i] = x[i];
    }
}

inline static void vecncpy(float* y, const float* x, const int n) {
    int i;
    __m256 neg_one = _mm256_set1_ps(-1.0f);
    for (i = 0; i <= n - 8; i += 8) {
        _mm256_store_ps(&y[i], _mm256_mul_ps(_mm256_load_ps(&x[i]), neg_one));
    }
    for (; i < n; ++i) {
        y[i] = -x[i];
    }
}

inline static void vecadd(float* y, const float* x, const float c, const int n) {
    int i;
    __m256 factor = _mm256_set1_ps(c);
    for (i = 0; i <= n - 8; i += 8) {
        _mm256_store_ps(&y[i], _mm256_fmadd_ps(_mm256_load_ps(&x[i]), factor, _mm256_load_ps(&y[i])));
    }
    for (; i < n; ++i) {
        y[i] += c * x[i];
    }
}

inline static void vecdiff(float* z, const float* x, const float* y, const int n) {
    int i = 0;
    //for (i = 0; i <= n - 8; i += 8) {
        //_mm256_store_ps(&z[i], _mm256_sub_ps(_mm256_load_ps(&x[i]), _mm256_load_ps(&y[i])));
    //}
    for (; i < n; ++i) {
        z[i] = x[i] - y[i];
    }
}

inline static void vecscale(float* y, const float c, const int n) {
    int i;
    __m256 factor = _mm256_set1_ps(c);
    for (i = 0; i <= n - 8; i += 8) {
        _mm256_store_ps(&y[i], _mm256_mul_ps(_mm256_load_ps(&y[i]), factor));
    }
    for (; i < n; ++i) {
        y[i] *= c;
    }
}

inline static void vecmul(float* y, const float* x, const int n) {
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        _mm256_store_ps(&y[i], _mm256_mul_ps(_mm256_load_ps(&x[i]), _mm256_load_ps(&y[i])));
    }
    for (; i < n; ++i) {
        y[i] *= x[i];
    }
}

inline static void vecdot(float* s, const float* x, const float* y, const int n) {
    __m256 sum = _mm256_setzero_ps();
    int i;

    for (i = 0; i <= n - 8; i += 8) {
        sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_load_ps(&x[i]), _mm256_load_ps(&y[i])));
    }

    // Horizontal sum of all elements in the vector sum
    float temp[8];
    _mm256_store_ps(temp, sum);
    *s = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

    // Sum the remaining elements
    for (; i < n; ++i) {
        *s += x[i] * y[i];
    }
}

inline static void vec2norm(float* s, const float* x, const int n) {
    vecdot(s, x, x, n);
    *s = (float)sqrt(*s);
}

inline static void vec2norminv(float* s, const float* x, const int n) {
    vec2norm(s, x, n);
    *s = (float)(1.0f / *s);
}
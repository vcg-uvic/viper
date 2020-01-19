// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     https://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "Common.h"
#include "CudaUtils.cuh"
#include <thrust/device_vector.h>

namespace viper {

struct CudaState {
    thrust::device_vector<Vec3> x;
    thrust::device_vector<float> r;
    thrust::device_vector<Quaternion> q;
};

struct CudaSimData {
    CudaState X;  // current
    CudaState Xp; // previous
    CudaState Xi; // initial

    thrust::device_vector<float> w;
    thrust::device_vector<float> wq;
    thrust::device_vector<float> wr;

    thrust::device_vector<Matrix4> b;
    thrust::device_vector<Matrix4> bp;
    thrust::device_vector<Matrix4> bi;

    thrust::device_vector<uint8_t> xa;
    thrust::device_vector<uint8_t> qa;
};

struct CudaProjections {
    thrust::device_vector<int> id;
    thrust::device_vector<Vec6> dx;

    void resize(int n) {
        id.resize(n);
        dx.resize(n);
    }

    void setZero() {
        thrust::fill(id.begin(), id.end(), 0);
        thrust::fill(dx.begin(), dx.end(), Vec6::Zero());
    }
};

struct CudaStatePtr {
    CudaStatePtr(CudaSimData &S) {
        x = ptr(S.X.x);
        q = ptr(S.X.q);
        r = ptr(S.X.r);

        xp = ptr(S.Xp.x);
        qp = ptr(S.Xp.q);
        rp = ptr(S.Xp.r);

        xi = ptr(S.Xi.x);
        qi = ptr(S.Xi.q);
        ri = ptr(S.Xi.r);

        b = ptr(S.b);
        bp = ptr(S.bp);
        bi = ptr(S.bi);

        w = ptr(S.w);
        wq = ptr(S.wq);
        wr = ptr(S.wr);

        xa = ptr(S.xa);
        qa = ptr(S.qa);
    }

    Vec3 *x;
    Quaternion *q;
    float *r;

    Vec3 *xp;
    Quaternion *qp;
    float *rp;

    Vec3 *xi;
    Quaternion *qi;
    float *ri;

    Matrix4 *b;
    Matrix4 *bp;
    Matrix4 *bi;

    float *w;
    float *wr;
    float *wq;

    uint8_t *xa;
    uint8_t *qa;
};

struct CudaProjectionsPtr {
    CudaProjectionsPtr(CudaProjections &P, int offset = 0) {
        id = ptr(P.id) + offset;
        dx = ptr(P.dx) + offset;
    }

    int *id;
    Vec6 *dx;
};

} // namespace viper
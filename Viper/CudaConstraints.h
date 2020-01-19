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
#include "State.h"

namespace viper {

struct ConstraintBase {
    bool enabled = true;
};

struct C_distance : public ConstraintBase {
    C_distance() = default;
    C_distance(int a, int b, float restDistance, float compliance = 0.f)
        : a(a), b(b), rDist(restDistance), compliance(compliance) {}
    int a;
    int b;
    float rDist;
    float compliance; // inverse stiffness
};

struct C_distancemax : public ConstraintBase {
    C_distancemax() = default;
    C_distancemax(int a, int b, float max_distance)
        : a(a), b(b), max_distance(max_distance) {}
    int a;
    int b;
    float max_distance;
};

struct C_skinning : public ConstraintBase {
    C_skinning() = default;
    C_skinning(int i, int t0, int t1, float w0, float w1)
        : i(i), t0(t0), t1(t1), w0(w0), w1(w1) {}
    int i;
    int t0;
    int t1;
    float w0;
    float w1;
};

struct C_volume : public ConstraintBase {
    C_volume() = default;
    C_volume(int a, int b, float Vr, float compliance = 0.0)
        : a(a), b(b), Vr(Vr), compliance(compliance) {}
    C_volume(int a, int b, const SimulationState &state, float compliance = 0.0)
        : a(a), b(b), compliance(compliance) {
        float ra = state.r[a];
        float rb = state.r[b];
        float d = (state.x[a] - state.x[b]).norm();
        if (d > 1e-7f) {
            float e = (rb - ra) / d;
            float L = d + (ra - rb) * e;
            Vr = M_PIf / 3.0f *
                 ((ra * ra * ra - rb * rb * rb) * (e * e * e - 3.f * e) +
                  L * (1.f - e * e) * (ra * ra + ra * rb + rb * rb));
        } else
            Vr = 0.f;
    }

    int a;
    int b;
    float Vr;
    float compliance;
};

struct C_volume2 : public ConstraintBase {
    C_volume2() = default;
    C_volume2(int a, int b, int c, const SimulationState &state,
              float compliance = 0.0)
        : a(a), b(b), c(c), compliance(compliance) {
        l0 = (state.x[a] - state.x[b]).norm();
        r0 = 0.5f * (state.r[a] + state.r[b]);
    }

    int a;
    int b;
    int c;
    float compliance;
    float l0;
    float r0;
};

struct C_bend : public ConstraintBase {
    C_bend() = default;
    C_bend(int a, int b, const SimulationState &state, float compliance = 0.f)
        : a(a), b(b), compliance(compliance) {
        darbouxRest = state.q[a].conjugate() * state.q[b];
    }

    int a;
    int b;
    Quaternion darbouxRest;
    float compliance;
};

struct C_stretch : public ConstraintBase {
    C_stretch() = default;
    C_stretch(int a, int b, int c, float L, float compliance = 0.f)
        : a(a), b(b), c(c), L(L), compliance(compliance) {}

    int a;   // particle id
    int b;   // partile id
    int c;   // pill id
    float L; // rest length
    float compliance;
};

struct C_collpp : public ConstraintBase {
    int a;
    int b;
};

struct C_collision : public ConstraintBase {
    Vec2i a;
    Vec2i b;
};

struct C_radius : public ConstraintBase {
    C_radius() = default;
    C_radius(int a, float r, float compliance = 0.0)
        : a(a), r(r), compliance(compliance) {}
    int a;
    float r;
    float compliance;
};

struct C_bilap : public ConstraintBase {
    C_bilap() = default;
    C_bilap(const std::vector<int> &A, int i, float compliance = 0.0)
        : compliance(compliance) {
        int n = A.size();
        if (i == 0) {
            ids[0] = A[i];
            ids[1] = A[i];
            ids[2] = A[i];
            ids[3] = A[i + 1];
            ids[4] = A[i + 2];
            w[0] = 0.f;
            w[1] = 0.f;
            w[2] = 2.f;
            w[3] = -3.f;
            w[4] = 1.f;
        } else if (i == 1) {
            ids[0] = A[i];
            ids[1] = A[i - 1];
            ids[2] = A[i];
            ids[3] = A[i + 1];
            ids[4] = A[i + 2];
            w[0] = 0.f;
            w[1] = -3.f;
            w[2] = 6.f;
            w[3] = -4.f;
            w[4] = 1.f;
        } else if (i > 1 && i < n - 2) {
            ids[0] = A[i - 2];
            ids[1] = A[i - 1];
            ids[2] = A[i];
            ids[3] = A[i + 1];
            ids[4] = A[i + 2];
            w[0] = 1.f;
            w[1] = -4.f;
            w[2] = 6.f;
            w[3] = -4.f;
            w[4] = 1.f;
        } else if (i == n - 2) {
            ids[0] = A[i - 2];
            ids[1] = A[i - 1];
            ids[2] = A[i];
            ids[3] = A[i + 1];
            ids[4] = A[i];
            w[0] = 1.f;
            w[1] = -4.f;
            w[2] = 6.f;
            w[3] = -3.f;
            w[4] = 0.f;
        } else if (i == n - 1) {
            ids[0] = A[i - 2];
            ids[1] = A[i - 1];
            ids[2] = A[i];
            ids[3] = A[i];
            ids[4] = A[i];
            w[0] = 1.f;
            w[1] = -3.f;
            w[2] = 2.f;
            w[3] = 0.f;
            w[4] = 0.f;
        }
    }

    int ids[5];
    float w[5];
    float compliance;
};

#define SHAPE_MATCHING_MAX 11

struct C_shape : public ConstraintBase {
    C_shape() = default;
    C_shape(const std::vector<int> &ids, const SimulationState &state,
            float compliance = 0.0)
        : q(Quaternion::Identity()), K(1.f), compliance(compliance) {
        n = ids.size();
        assert(n <= SHAPE_MATCHING_MAX);
        assert(n > 1);

        Vec3 center = Vec3::Zero();
        for (int i = 0; i < n; i++)
            center += state.x[ids[i]];
        center /= (float)n;

        for (int i = 0; i < SHAPE_MATCHING_MAX; i++) {
            if (i < n) {
                id[i] = ids[i];
                xp[i] = state.x[ids[i]] - center;
            } else {
                id[i] = 0;
                xp[i] = Vec3::Zero();
            }
        }

        r = 0.f;
        for (int i = 0; i < n; i++)
            r += state.r[id[i]];
        r /= (float)n;
    }

    int n;
    int id[SHAPE_MATCHING_MAX];
    Vec3 xp[SHAPE_MATCHING_MAX];
    Quaternion q;
    float K;
    float r;
    float compliance;
};

struct C_shape2 : public ConstraintBase {
    C_shape2() = default;
    C_shape2(const std::vector<int> &ids, const std::vector<int> &qas,
             const std::vector<int> &qbs, const SimulationState &state) {
        n = ids.size();
        assert(n <= SHAPE_MATCHING_MAX);
        assert(n > 1);
        assert(n == qas.size() && n == qbs.size());

        for (int i = 0; i < n; i++) {
            id[i] = ids[i];
            qa[i] = qas[i];
            qb[i] = qbs[i];
        }

        q = Quaternion::Identity();
    }

    int n;
    int id[SHAPE_MATCHING_MAX];
    int qa[SHAPE_MATCHING_MAX];
    int qb[SHAPE_MATCHING_MAX];
    Quaternion q;
};

struct C_touch : public ConstraintBase {
    C_touch() = default;
    C_touch(int a, int b) : a(a), b(b) {}
    int a;
    int b;
};

struct ConstraintsCPU {
    std::vector<C_distance> distance;
    std::vector<C_distancemax> distancemax;
    std::vector<C_skinning> skinning;
    std::vector<C_volume> volume;
    std::vector<C_bend> bend;
    std::vector<C_stretch> stretch;
    std::vector<C_radius> radius;
    std::vector<C_bilap> bilap;
    std::vector<C_shape> shape;
    std::vector<C_touch> touch;

    std::vector<C_volume2> volume2;
    std::vector<C_shape2> shape2;
};

} // namespace viper
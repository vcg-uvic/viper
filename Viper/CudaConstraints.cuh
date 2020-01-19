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
#include "CudaConstraints.h"
#include "CudaData.cuh"
#include <Eigen/SVD>

namespace viper {

struct C_distance_solve {
    C_distance_solve(C_distance *C, CudaStatePtr S, CudaProjectionsPtr P,
                     float *L, float dt)
        : C(C), S(S), P(P), L(L), dt(dt) {}
    C_distance *C;
    CudaStatePtr S;
    CudaProjectionsPtr P;
    float *L;
    float dt;

    __host__ __device__ void operator()(int i) const {
        int a = C[i].a;
        int b = C[i].b;
        Vec3 ab = S.x[b] - S.x[a];
        float dist = ab.norm();
        Vec3 n = -ab / dist;

        float alpha = C[i].compliance / (dt * dt);
        float c = dist - C[i].rDist;
        float dL = (-c - alpha * L[i]) / (S.w[a] + S.w[b] + alpha);
        Vec3 corr = dL * n;
        L[i] += dL;

        P.id[2 * i + 0] = a;
        P.id[2 * i + 1] = b;
        P.dx[2 * i + 0].head<3>() = S.w[a] * corr;
        P.dx[2 * i + 1].head<3>() = -S.w[b] * corr;
        P.dx[2 * i + 0][4] = 1.f;
        P.dx[2 * i + 1][4] = 1.f;
    }
};

struct C_distancemax_solve {
    C_distancemax_solve(C_distancemax *C, CudaStatePtr S, CudaProjectionsPtr P)
        : C(C), S(S), P(P) {}
    C_distancemax *C;
    CudaStatePtr S;
    CudaProjectionsPtr P;

    __host__ __device__ void operator()(int i) const {
        int a = C[i].a;
        int b = C[i].b;
        Vec3 ab = S.x[b] - S.x[a];
        float dist = ab.norm();
        Vec3 n = ab / dist;
        Vec3 corr = (dist - C[i].max_distance) / (S.w[a] + S.w[b]) * n;
        float s = dist > C[i].max_distance;
        P.id[2 * i + 0] = a;
        P.id[2 * i + 1] = b;
        P.dx[2 * i + 0].head<3>() = S.w[a] * corr * s;
        P.dx[2 * i + 1].head<3>() = -S.w[b] * corr * s;
        P.dx[2 * i + 0][4] = s;
        P.dx[2 * i + 1][4] = s;
    }
};

struct C_skinning_solve {
    C_skinning_solve(C_skinning *C, CudaStatePtr S) : C(C), S(S) {}
    C_skinning *C;
    CudaStatePtr S;

    __host__ __device__ Vec4 toH(Vec3 v) const {
        return Vec4(v[0], v[1], v[2], 1.f);
    }

    __host__ __device__ void operator()(int i) const {
        Matrix4 T0 = Matrix4::Identity();
        Matrix4 T1 = Matrix4::Identity();

        Matrix4 T0i = Matrix4::Identity();
        Matrix4 T1i = Matrix4::Identity();

        if (C[i].t0 != -1) {
            T0 = S.b[C[i].t0];
            T0i = S.bp[C[i].t0];
        }
        if (C[i].t1 != -1) {
            T1 = S.b[C[i].t1];
            T1i = S.bp[C[i].t1];
        }

        Vec3 p = S.x[C[i].i];
        Vec3 proj0 = (T0 * T0i * toH(p)).head<3>();
        Vec3 proj1 = (T1 * T1i * toH(p)).head<3>();
        Vec3 proj = C[i].w0 * proj0 + C[i].w1 * proj1;

        if (S.w[C[i].i] > 0) {
            S.x[C[i].i] = proj;
            S.xp[C[i].i] += proj - p;
        }
    }
};

struct C_collpp_solve {
    C_collpp_solve(C_collpp *C, CudaStatePtr S, CudaProjectionsPtr P)
        : C(C), S(S), P(P) {}
    C_collpp *C;
    CudaStatePtr S;
    CudaProjectionsPtr P;

    __host__ __device__ void operator()(int i) const {
        int a = C[i].a;
        int b = C[i].b;
        Vec3 ab = S.x[b] - S.x[a];
        float d = ab.norm();
        float R = S.r[a] + S.r[b];
        Vec3 n = ab / d;
        float s = d < R;
        Vec3 corr = (d - R) / (S.w[a] + S.w[b]) * n * s;
        P.id[2 * i + 0] = a;
        P.id[2 * i + 1] = b;
        P.dx[2 * i + 0].head<3>() = S.w[a] * corr;
        P.dx[2 * i + 1].head<3>() = -S.w[b] * corr;
        P.dx[2 * i + 0][4] = s;
        P.dx[2 * i + 1][4] = s;
    }
};

struct C_volume_solve {
    C_volume_solve(C_volume *C, CudaStatePtr S, CudaProjectionsPtr P, float *L,
                   float dt)
        : C(C), S(S), P(P), Lambda(L), dt(dt) {}
    C_volume *C;
    CudaStatePtr S;
    CudaProjectionsPtr P;
    float *Lambda;
    float dt;

    __host__ __device__ void operator()(int i) const {
        int a = C[i].a;
        int b = C[i].b;
        float ra = S.r[a];
        float rb = S.r[b];
        float Vr = C[i].Vr;
        Vec3 ba = S.x[a] - S.x[b];
        float d = ba.norm();
        float e = (rb - ra) / d;
        float L = d + (ra - rb) * e;
        float e2 = e * e;
        float e3 = e2 * e;
        float e4 = e3 * e;
        float ra2 = ra * ra;
        float rb2 = rb * rb;
        float ra3 = ra2 * ra;
        float rb3 = rb2 * rb;
        float rarb = (ra2 + ra * rb + rb2);
        float gpa_n = (M_PIf * (e - e3) * (ra3 - rb3) / d +
                       M_PIf / 3.f * rarb * (1.f - 3.f * e4 + 2.f * e2)) /
                      Vr;
        float gra =
            (M_PIf * ra2 * (ra / d * (1.f - e2) + e3 - 3.f * e) +
             M_PIf * rb3 / d * (e2 - 1.f) +
             M_PIf / 3.f *
                 (+4.f * rarb * (e - e3) + L * (1.f - e2) * (2.f * ra + rb))) /
            Vr;
        float grb =
            (M_PIf * rb2 * (rb / d * (1.f - e2) + 3.f * e - e3) +
             M_PIf * ra3 / d * (e2 - 1.f) +
             M_PIf / 3.f *
                 (-4.f * rarb * (e - e3) + L * (1.f - e2) * (ra + 2.f * rb))) /
            Vr;
        Vec3 ban = ba / d;

        float wSum = (S.w[a] + S.w[b]) * gpa_n * gpa_n + S.wr[a] * gra * gra +
                     S.wr[b] * grb * grb;
        // float wSum = S.wr[a] * gra * gra + S.wr[b] * grb * grb;
        float V = M_PIf / 3.0f *
                  ((ra3 - rb3) * (e3 - 3.f * e) + L * (1.f - e2) * rarb);
        float c = V / Vr - 1.f;

        float alpha = C[i].compliance / (dt * dt);
        float dL = (-c - alpha * Lambda[i]) / (wSum + alpha);

        Vec3 dpa = gpa_n * ban * dL;
        float dra = gra * dL;
        float drb = grb * dL;

        P.id[2 * i + 0] = a;
        P.id[2 * i + 1] = b;
        P.dx[2 * i + 0].head<3>() = S.w[a] * dpa;
        P.dx[2 * i + 1].head<3>() = -S.w[b] * dpa;
        P.dx[2 * i + 0][3] = S.wr[a] * dra;
        P.dx[2 * i + 1][3] = S.wr[b] * drb;
        P.dx[2 * i + 0].tail<2>() = Vec2::Ones();
        P.dx[2 * i + 1].tail<2>() = Vec2::Ones();
        Lambda[i] += dL;

        // printf("V/Vr: %f\n", V / Vr);
    }
};

struct C_volume2_solve {
    C_volume2_solve(C_volume2 *C, CudaStatePtr S, CudaProjectionsPtr P,
                    float *L, float dt)
        : C(C), S(S), P(P), Lambda(L), dt(dt) {}
    C_volume2 *C;
    CudaStatePtr S;
    CudaProjectionsPtr P;
    float *Lambda;
    float dt;

    __host__ __device__ void operator()(int i) const {
        int a = C[i].a;
        int b = C[i].b;
        int c = C[i].c;
        float s = 0.5f * (S.r[a] + S.r[b]);
        float s0 = C[i].r0;
        float l0 = C[i].l0;
        Vec3 ab = S.x[b] - S.x[a];
        Quaternion qc = S.q[c];
        Quaternion q = S.q[c];
        Vec3 R3 = ab.normalized();

        float r3ab = R3.dot(ab);
        float X = s * s * r3ab - s0 * s0 * l0;
        float W = X;

        Vec3 gp = -s * s * R3;
        float gr = s * r3ab;
        Vec4 gq = Vec4::Zero();

        float wSum = (S.wr[a] + S.wr[b]) * gr * gr +
                     (S.w[a] + S.w[b]) * gp.squaredNorm() +
                     S.wq[c] * gq.squaredNorm() + 1e-6f;
        float lambda = W / wSum;
        Vec3 dp = -lambda * gp;
        float dr = -lambda * gr;
        Vec4 dq = -lambda * gq;

        float w = 1.0f;
        P.id[3 * i + 0] = a;
        P.id[3 * i + 1] = b;
        P.id[3 * i + 2] = c;
        P.dx[3 * i + 0].head<3>() = S.w[a] * dp * w;
        P.dx[3 * i + 1].head<3>() = -S.w[b] * dp * w;
        P.dx[3 * i + 0][3] = S.wr[a] * dr * w;
        P.dx[3 * i + 1][3] = S.wr[b] * dr * w;
        P.dx[3 * i + 2].head<4>() = S.wq[c] * dq * w;
        P.dx[3 * i + 0].tail<2>() = Vec2::Ones() * w;
        P.dx[3 * i + 1].tail<2>() = Vec2::Ones() * w;
        P.dx[3 * i + 2][4] = 1.f * w;
    }
};

struct C_bend_solve {
    C_bend_solve(C_bend *C, CudaStatePtr S, CudaProjectionsPtr P, int N,
                 float *L, float dt)
        : C(C), S(S), P(P), N(N), L(L), dt(dt) {}
    C_bend *C;
    CudaStatePtr S;
    CudaProjectionsPtr P;
    int N;
    float *L;
    float dt;

    __device__ void operator()(int i) const {
        Quaternion qa = S.q[C[i].a];
        Quaternion qb = S.q[C[i].b];
        float wa = S.wq[C[i].a];
        float wb = S.wq[C[i].b];

        Quaternion omega = qa.conjugate() * qb; // darboux vector
        Quaternion omega_plus;
        omega_plus.coeffs() =
            omega.coeffs() +
            C[i].darbouxRest.coeffs(); // delta Omega with -Omega_0
        omega.coeffs() =
            omega.coeffs() -
            C[i].darbouxRest.coeffs(); // delta Omega with + omega_0
        if (omega.squaredNorm() > omega_plus.squaredNorm())
            omega = omega_plus;

        Vec3 lambda = Vec3(L[3 * i + 0], L[3 * i + 1], L[3 * i + 2]);
        float alpha = C[i].compliance / (dt * dt);
        Vec3 Cx = Vec3(omega.x(), omega.y(), omega.z());
        Vec3 dL = (-Cx - alpha * lambda) / (wa + wb + alpha);
        Quaternion dLq = Quaternion(0.f, dL[0], dL[1], dL[2]);

        Quaternion da = qb * dLq;
        Quaternion db = qa * dLq;
        da.coeffs() *= -wa;
        db.coeffs() *= wb;

        float w = 1.f;
        P.id[2 * i + 0] = C[i].a + N;
        P.id[2 * i + 1] = C[i].b + N;
        P.dx[2 * i + 0].head<4>() = da.coeffs() * w;
        P.dx[2 * i + 1].head<4>() = db.coeffs() * w;
        P.dx[2 * i + 0][4] = w;
        P.dx[2 * i + 1][4] = w;
        L[3 * i + 0] += dL[0];
        L[3 * i + 1] += dL[1];
        L[3 * i + 2] += dL[2];
    }
};

struct C_stretch_solve {
    C_stretch_solve(C_stretch *C, CudaStatePtr S, CudaProjectionsPtr P, int N,
                    float *L, float dt)
        : C(C), S(S), P(P), N(N), L(L), dt(dt) {}
    C_stretch *C;
    CudaStatePtr S;
    CudaProjectionsPtr P;
    int N;
    float *L;
    float dt;

    __device__ void operator()(int i) const {
        {
            int a = C[i].a;
            int b = C[i].b;
            int c = C[i].c;
            const Quaternion &qc = S.q[c];
            float wa = S.w[a];
            float wb = S.w[b];
            float wq = S.wq[c];

            Vec3 d3;
            d3[0] = 2.0 * (qc.x() * qc.z() + qc.w() * qc.y());
            d3[1] = 2.0 * (qc.y() * qc.z() - qc.w() * qc.x());
            d3[2] = qc.w() * qc.w() - qc.x() * qc.x() - qc.y() * qc.y() +
                    qc.z() * qc.z();

            Vec3 Cx = (S.x[b] - S.x[a]) / C[i].L - d3;
            Vec3 lambda = Vec3(L[i * 3 + 0], L[i * 3 + 1], L[i * 3 + 2]);
            float alpha = C[i].compliance / (dt * dt);
            float l2 = C[i].L * C[i].L;
            Vec3 dL = (-Cx - alpha * lambda) * l2 /
                      (wa + wb + 4 * l2 * wq + alpha * l2);
            Vec3 dp = dL / C[i].L;

            Quaternion q_e_3_bar(qc.z(), -qc.y(), qc.x(),
                                 -qc.w()); // compute q*e_3.conjugate (cheaper
                                           // than quaternion product)
            Vec3 dq_v = -2.f * wq * dL;
            Quaternion dq =
                Quaternion(0.0, dq_v.x(), dq_v.y(), dq_v.z()) * q_e_3_bar;

            if (C[i].L < 1e-6f)
                printf("i: %d L: %f abc: %d %d %d \n", i, C[i].L, a, b, c);

            P.id[3 * i + 0] = a;
            P.id[3 * i + 1] = b;
            P.dx[3 * i + 0].head<3>() = -wa * dp;
            P.dx[3 * i + 1].head<3>() = wb * dp;
            P.dx[3 * i + 0][4] = 1.f;
            P.dx[3 * i + 1][4] = 1.f;
            P.id[3 * i + 2] = c + N;
            P.dx[3 * i + 2].head<4>() = dq.coeffs();
            P.dx[3 * i + 2][4] = 1.f;
            L[3 * i + 0] += dL[0];
            L[3 * i + 1] += dL[1];
            L[3 * i + 2] += dL[2];
        }
    }
};

struct C_radius_solve {
    C_radius_solve(C_radius *C, CudaStatePtr S, CudaProjectionsPtr P, float *L,
                   float dt)
        : C(C), S(S), P(P), L(L), dt(dt) {}
    C_radius *C;
    CudaStatePtr S;
    CudaProjectionsPtr P;
    float *L;
    float dt;

    __device__ void operator()(int i) const {
        int id = C[i].a;
        float w = S.wr[id];
        float r0 = C[i].r;
        float alpha = C[i].compliance / (dt * dt);
        float c = S.r[id] / r0 - 1.0f;
        float dL = (-c - alpha * L[i]) / (w / (r0 * r0) + alpha);
        float dr = w / r0 * dL;

        if (w < 1e-6f)
            return;

        P.id[i] = id;
        P.dx[i][3] = dr;
        P.dx[i][4] = 1.f;
        L[i] += dL;
    }
};

struct C_touch_solve {
    C_touch_solve(C_touch *C, CudaStatePtr S, CudaProjectionsPtr P, float *L,
                  float dt)
        : C(C), S(S), P(P), L(L), dt(dt) {}
    C_touch *C;
    CudaStatePtr S;
    CudaProjectionsPtr P;
    float *L;
    float dt;

    __device__ void operator()(int i) const {
        int a = C[i].a;
        int b = C[i].b;
        Vec3 ba = S.x[a] - S.x[b];
        float dist = ba.norm();
        float c = dist - S.r[a] - S.r[b];
        float wSum = S.w[a] + S.w[b] + S.wr[a] + S.wr[b];
        Vec3 dx = -c / wSum * ba.normalized();
        float dr = c / wSum;

        float w = 1.f;
        P.id[2 * i + 0] = C[i].a;
        P.id[2 * i + 1] = C[i].b;
        P.dx[2 * i + 0].head<3>() = S.w[a] * dx * w;
        P.dx[2 * i + 1].head<3>() = -S.w[b] * dx * w;
        P.dx[2 * i + 0][3] = S.wr[a] * dr * w;
        P.dx[2 * i + 1][3] = S.wr[b] * dr * w;
        P.dx[2 * i + 0][4] = w;
        P.dx[2 * i + 1][4] = w;
        P.dx[2 * i + 0][5] = w;
        P.dx[2 * i + 1][5] = w;
    }
};

struct C_bilap_solve {
    C_bilap_solve(C_bilap *C, CudaStatePtr S, CudaProjectionsPtr P, float *L,
                  float dt)
        : C(C), S(S), P(P), L(L), dt(dt) {}
    C_bilap *C;
    CudaStatePtr S;
    CudaProjectionsPtr P;
    float *L;
    float dt;

    __device__ void operator()(int i) const {
        int id = C[i].ids[2];
        float r = 0.f;
        for (int k = 0; k < 5; k++) {
            if (k == 2)
                continue;
            r -= C[i].w[k] * S.r[C[i].ids[k]];
        }
        r /= C[i].w[2];

        float alpha = C[i].compliance / (dt * dt);
        float c = S.r[id] - r;
        float dL = (-c - alpha * L[i]) / (S.w[id] + alpha);
        float dr = S.w[id] * dL;

        if (S.wr[id] < 1e-6f)
            return;

        float w = 1.f;
        P.id[i] = id;
        P.dx[i][3] = dr * w;
        P.dx[i][5] = w;
        L[i] += dL;
    }
};

struct C_shape2_solve {
    C_shape2_solve(C_shape2 *C, CudaStatePtr S, CudaProjectionsPtr P, int N,
                   float *L, float dt)
        : C(C), S(S), P(P), N(N), L(L), dt(dt) {}
    C_shape2 *C;
    CudaStatePtr S;
    CudaProjectionsPtr P;
    int *o;
    int N;
    float *L;
    float dt;

    __device__ void operator()(int i) const {
        int n = C[i].n;
        Matrix4 T_source[SHAPE_MATCHING_MAX];
        Matrix4 T_target[SHAPE_MATCHING_MAX];

        for (int k = 0; k < n; k++) {
            T_target[k] = Matrix4::Identity();
            T_target[k].block<3, 3>(0, 0) =
                S.r[C[i].id[k]] *
                S.q[C[i].qa[k]].slerp(0.5, S.q[C[i].qb[k]]).toRotationMatrix();
            T_target[k].col(3).head<3>() = S.x[C[i].id[k]];

            T_source[k] = Matrix4::Identity();
            T_source[k].block<3, 3>(0, 0) =
                S.ri[C[i].id[k]] * S.qi[C[i].qa[k]]
                                       .slerp(0.5, S.qi[C[i].qb[k]])
                                       .toRotationMatrix();
            T_source[k].col(3).head<3>() = S.xi[C[i].id[k]];
        }

        Vec3 mean_source = Vec3::Zero();
        Vec3 mean_target = Vec3::Zero();
        for (int k = 0; k < n; k++) {
            mean_source += T_source[k].col(3).head<3>();
            mean_target += T_target[k].col(3).head<3>();
        }
        mean_source /= n;
        mean_target /= n;

        for (int k = 0; k < n; k++) {
            T_source[k].col(3).head<3>() -= mean_source;
            T_target[k].col(3).head<3>() -= mean_target;
        }

        Matrix3 A = Matrix3::Zero();
        for (int k = 0; k < n; k++) {
            Eigen::Matrix<float, 3, 4> a = T_target[k].block<3, 4>(0, 0);
            Eigen::Matrix<float, 3, 4> b = T_source[k].block<3, 4>(0, 0);
            A += a * b.transpose();
        }
        A = A / (4.0 * n);

        for (int iter = 0; iter < 100; iter++) {
            Matrix3 R = C[i].q.matrix();
            Vec3 omega =
                (R.col(0).cross(A.col(0)) + R.col(1).cross(A.col(1)) +
                 R.col(2).cross(A.col(2))) *
                (1.0 / fabs(R.col(0).dot(A.col(0)) + R.col(1).dot(A.col(1)) +
                            R.col(2).dot(A.col(2))) +
                 1.0e-9);
            float w = (float)omega.norm();
            if (w < 1.0e-6f)
                break;
            C[i].q = Quaternion(Rotation(w, (float)(1.0 / w) * omega)) * C[i].q;
            C[i].q.normalize();
        }
        Matrix3 R = C[i].q.matrix();

        float nom = 0.0f;
        float den = 0.0f;
        for (int k = 0; k < n; k++) {
            Eigen::Matrix<float, 3, 4> a = T_source[k].block<3, 4>(0, 0);
            Eigen::Matrix<float, 3, 4> b = T_target[k].block<3, 4>(0, 0);
            nom += (R * a).cwiseProduct(b).sum();
            den += a.cwiseProduct(a).sum();
        }
        float s = nom / den;
        Vec3 Rm = R * mean_source;
        Vec3 t = mean_target - s * Rm;

        Quaternion qR = Quaternion(R);
        int M = 3 * SHAPE_MATCHING_MAX;

        t = mean_target - mean_source;
        for (int k = 0; k < n; k++) {
            int id = C[i].id[k];
            int a = C[i].qa[k];
            int b = C[i].qb[k];

            Vec3 dx = s * R * (S.xi[id] - mean_source) + mean_target - S.x[id];
            Vec4 dqa = (C[i].q * S.qi[a]).coeffs() - S.q[a].coeffs();
            Vec4 dqb = (C[i].q * S.qi[b]).coeffs() - S.q[b].coeffs();
            float dr = s * S.ri[id] - S.r[id];

            if (S.w[id] > 1e-6f) {
                P.id[i * M + 0 * SHAPE_MATCHING_MAX + k] = id;
                P.dx[i * M + 0 * SHAPE_MATCHING_MAX + k].head<3>() = dx;
                P.dx[i * M + 0 * SHAPE_MATCHING_MAX + k][3] = dr;
                P.dx[i * M + 0 * SHAPE_MATCHING_MAX + k][4] = 1.0f;
                P.dx[i * M + 0 * SHAPE_MATCHING_MAX + k][5] = 1.0f;
            }

            P.id[i * M + 1 * SHAPE_MATCHING_MAX + k] = a + N;
            P.dx[i * M + 1 * SHAPE_MATCHING_MAX + k].head<4>() = dqa;
            P.dx[i * M + 1 * SHAPE_MATCHING_MAX + k][4] = 1.0f;

            P.id[i * M + 2 * SHAPE_MATCHING_MAX + k] = b + N;
            P.dx[i * M + 2 * SHAPE_MATCHING_MAX + k].head<4>() = dqb;
            P.dx[i * M + 2 * SHAPE_MATCHING_MAX + k][4] = 1.0f;
        }
    }
};

struct C_shape_solve {
    C_shape_solve(C_shape *C, CudaStatePtr S, CudaProjectionsPtr P, float *L,
                  float dt)
        : C(C), S(S), P(P), L(L), dt(dt) {}
    C_shape *C;
    CudaStatePtr S;
    CudaProjectionsPtr P;
    int *o;
    float *L;
    float dt;

    __device__ void operator()(int i) const {
        int n = C[i].n;
        Vec3 center = Vec3::Zero();
        for (int k = 0; k < SHAPE_MATCHING_MAX; k++) {
            float s = k < n;
            center += S.x[C[i].id[k]] * s;
        }
        center /= (float)n;

        Vec3 x[SHAPE_MATCHING_MAX];
        for (int k = 0; k < SHAPE_MATCHING_MAX; k++)
            x[k] = S.x[C[i].id[k]] - center;

        Matrix3 A = Matrix3::Zero();
        for (int k = 0; k < SHAPE_MATCHING_MAX; k++)
            A += x[k] * C[i].xp[k].transpose();

        for (int iter = 0; iter < 10; iter++) {
            Matrix3 R = C[i].q.matrix();
            Vec3 omega =
                (R.col(0).cross(A.col(0)) + R.col(1).cross(A.col(1)) +
                 R.col(2).cross(A.col(2))) *
                (1.0 / fabs(R.col(0).dot(A.col(0)) + R.col(1).dot(A.col(1)) +
                            R.col(2).dot(A.col(2))) +
                 1.0e-9);
            float w = (float)omega.norm();
            if (w < 1.0e-6f)
                break;
            C[i].q = Quaternion(Rotation(w, (float)(1.0 / w) * omega)) * C[i].q;
            C[i].q.normalize();
        }

        float scale = 0.0f;
        for (int k = 0; k < SHAPE_MATCHING_MAX; k++) {
            float s = k < n;
            scale += S.r[C[i].id[k]] * s;
        }
        scale /= (float)n * C[i].r;

        float wSum = 0.f;
        Vec3 grad[SHAPE_MATCHING_MAX];
        for (int k = 0; k < SHAPE_MATCHING_MAX; k++) {
            grad[k] = x[k] - C[i].q * C[i].xp[k] * scale;
            wSum += grad[k].squaredNorm();
        }

        float alpha = 0.f;
        float dL = (-wSum - alpha * L[i]) / (wSum + alpha);

        float weight = 1.f;
        for (int k = 0; k < SHAPE_MATCHING_MAX; k++) {
            if (k < n && S.w[C[i].id[k]] > 0.f) {
                Vec3 corr = dL * grad[k];
                P.id[i * SHAPE_MATCHING_MAX + k] = C[i].id[k];
                P.dx[i * SHAPE_MATCHING_MAX + k].head<3>() = corr * weight;
                P.dx[i * SHAPE_MATCHING_MAX + k][4] = weight;
            }
        }

        L[i] += dL;
    }
};

__device__ float closestPtPointPill(const Vec3 &p, const Vec3 &a, const Vec3 &b,
                                    float ra, float rb, float L, float sigma,
                                    float &t, Vec &c, float &r) {
    Vec3 ab = b - a;
    Vec3 abn = ab / L;
    Vec3 ap = p - a;
    Vec3 p_proj = ap - abn * ap.dot(abn);
    Vec3 offset = abn * p_proj.norm() * sigma;
    t = min(1.f, max(0.f, abn.dot(ap + offset) / L));
    c = (1.f - t) * a + t * b;
    r = (1.f - t) * ra + t * rb;
    if (std::isnan(sigma))
        return min(ap.norm() - ra, (p - b).norm() - rb);
    return (c - p).norm() - r;
}

__device__ float pillDistanceU(const Vec3 *x, const float *r, const Vec2i &a,
                               const Vec2i &b, float LB, float sigmaB, float u,
                               float &v, Vec3 &pa, Vec3 &pb) {
    pa = (1.f - u) * x[a(0)] + u * x[a(1)];
    float ra = (1.f - u) * r[a(0)] + u * r[a(1)];
    float rc;
    return closestPtPointPill(pa, x[b(0)], x[b(1)], r[b(0)], r[b(1)], LB,
                              sigmaB, v, pb, rc) -
           ra;
}

__device__ float closestPtPills(const Vec3 *x, const float *r, const Vec2i &a,
                                const Vec2i &b, Vec2 &uv, Vec &pa, Vec &pb) {
    Vec2 range = Vec2(0.f, 1.f);
    float eps = 1e-6f;
    float LB = (x[b(0)] - x[b(1)]).norm();
    float sigmaB = tan(asin((r[b[1]] - r[b[0]]) / LB));
    for (int j = 0; j < 10; j++) {
        float mid = 0.5f * range.sum();
        float ua = mid - eps;
        float ub = mid + eps;
        float v;
        Vec3 pa, pb;
        float fa = pillDistanceU(x, r, a, b, LB, sigmaB, ua, v, pa, pb);
        float fb = pillDistanceU(x, r, a, b, LB, sigmaB, ub, v, pa, pb);

        if (fa == fb)
            range = Vec2(ua, ub);
        else if (fa > fb)
            range(0) = ua;
        else
            range(1) = ub;
    }
    uv(0) = 0.5f * range.sum();
    return pillDistanceU(x, r, a, b, LB, sigmaB, uv(0), uv(1), pa, pb);
}

struct C_collision_solve {
    C_collision_solve(C_collision *C, CudaStatePtr S, CudaProjectionsPtr P)
        : C(C), S(S), P(P) {}
    C_collision *C;
    CudaStatePtr S;
    CudaProjectionsPtr P;

    __device__ void operator()(int i) const {
        int a0 = C[i].a[0];
        int a1 = C[i].a[1];
        int b0 = C[i].b[0];
        int b1 = C[i].b[1];
        Vec2 uv;
        Vec3 pa, pb;
        float dist = closestPtPills(S.x, S.r, C[i].a, C[i].b, uv, pa, pb);

        float alpha1 = 1.f - uv(0);
        float alpha2 = uv(0);
        float beta1 = 1.f - uv(1);
        float beta2 = uv(1);

        Vec3 ba = (pa - pb).normalized();
        float wSum = S.w[a0] * alpha1 * alpha1 + S.w[a1] * alpha2 * alpha2 +
                     S.w[b0] * beta1 * beta1 + S.w[b1] * beta2 * beta2;
        Vec3 corr = dist / (wSum + 1e-7f) * ba;
        float s = dist < 0.0f && wSum > 1e-5f;

        P.id[4 * i + 0] = a0;
        P.id[4 * i + 1] = a1;
        P.id[4 * i + 2] = b0;
        P.id[4 * i + 3] = b1;
        P.dx[4 * i + 0].head<3>() = -alpha1 * S.w[a0] * corr * s;
        P.dx[4 * i + 1].head<3>() = -alpha2 * S.w[a1] * corr * s;
        P.dx[4 * i + 2].head<3>() = beta1 * S.w[b0] * corr * s;
        P.dx[4 * i + 3].head<3>() = beta2 * S.w[b1] * corr * s;
        P.dx[4 * i + 0][4] = s;
        P.dx[4 * i + 1][4] = s;
        P.dx[4 * i + 2][4] = s;
        P.dx[4 * i + 3][4] = s;
    }
};

} // namespace viper
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

#include "CollisionGrid.cuh"
#include "ConstraintsInfo.h"
#include "CudaConstraints.cuh"
#include "CudaData.cuh"
#include "CudaSolver.h"
#include "CudaUtils.cuh"
#include <fstream>
#include <thrust/binary_search.h>
#include <thrust/gather.h>

#define MIN_RADIUS 0.001f
#define COLLISION_MARGIN 0.002f

namespace viper {

struct CudaConstraints {
    thrust::device_vector<C_skinning> skinning;
    thrust::device_vector<C_distance> dist;
    thrust::device_vector<C_distancemax> distmax;
    thrust::device_vector<C_volume> vol;
    thrust::device_vector<C_volume2> vol2;
    thrust::device_vector<C_bend> bend;
    thrust::device_vector<C_stretch> stretch;
    thrust::device_vector<C_radius> radius;
    thrust::device_vector<C_shape> shape;
    thrust::device_vector<C_shape2> shape2;
    thrust::device_vector<C_touch> touch;
    thrust::device_vector<C_bilap> bilap;
    thrust::device_vector<C_collpp> collpp;
    thrust::device_vector<C_collision> collision;
};

struct CudaSolverData {
    CudaSimData S;      // state
    CudaProjections Pc; // projections per constraint
    CudaProjections Pp; // projections per particle
    CudaProjections Pt; // projections per particle temp
    CudaConstraints C;  // constraints
    CollisionGrid cgrid;

    thrust::device_vector<int> c_perm;
};

struct floor_friction {
    CudaStatePtr state;

    floor_friction(CudaStatePtr S) : state(S) {}

    __device__ void operator()(int i) const {
        if (state.xa[i] == 0)
            return;

        bool isTouching = state.x[i][1] - state.r[i] < 1e-6f;
        if (state.w[i] > 1e-6f && isTouching) {
            float pen = min(0.f, state.x[i][1] - state.r[i]);
            Vec3 dx = state.x[i] - state.xp[i];
            Vec3 tandx = Vec3(dx[0], 0.f, dx[2]);
            float tan_norm = tandx.norm();
            float mu_s = 0.01;
            float mu_k = 3.0;
            float factor = 0.99f;

            float d = abs(pen);
            if (tan_norm > mu_s * d)
                factor = min(0.99, mu_k * d / tan_norm);

            state.x[i][1] -= pen; // normal y
            state.x[i][0] -= tandx[0] * factor; // tangential x
            state.x[i][2] -= tandx[2] * factor; // tangential z
        }
        float wall = 20.0f;
        float newx = max(-wall, min(wall, state.x[i][0]));
        float newz = max(-wall, min(wall, state.x[i][2]));

    }
};

struct V_integration {
    CudaStatePtr state;
    float dt;
    Vec3 gravity;
    float damping;

    V_integration(CudaStatePtr S, float dt, Vec3 g, float damping)
        : state(S), dt(dt), gravity(g), damping(damping) {}

    __device__ void operator()(int i) const {
        if (state.xa[i] == 0)
            return;

        Vec3 v = Vec3::Zero();
        if (state.w[i] > 1e-6f)
            v = ((state.x[i] - state.xp[i]) / dt + dt * gravity) * damping;
        state.xp[i] = state.x[i];
        state.x[i] += dt * v;

        float dr = 0.f;
        if (state.wr[i] > 1e-6f)
            dr = (state.r[i] - state.rp[i]) * damping;
        state.rp[i] = state.r[i];
        state.r[i] += dr;
    }
};

struct Vq_integration {
    CudaStatePtr state;
    float dt;
    float damping;

    Vq_integration(CudaStatePtr S, float dt, float damping)
        : state(S), dt(dt), damping(damping) {}

    __device__ void operator()(int i) const {
        if (state.qa[i] == 0)
            return;

        Vec3 vq = 2.f / dt * (state.qp[i].conjugate() * state.q[i]).vec();
        vq *= damping;
        Quaternion vqq;
        vqq.w() = 0.f;
        vqq.vec() = vq;
        state.qp[i] = state.q[i];
        state.q[i] =
            state.qp[i].coeffs() + 0.5f * dt * (state.qp[i] * vqq).coeffs();
        state.q[i].normalize();
    }
};

struct bend_damping {
    CudaStatePtr state;
    C_bend *C;
    float dt;
    float damping;

    __device__ void operator()(int i) const {
        Quaternion &qa = state.q[C[i].a];
        Quaternion &qb = state.q[C[i].b];
        Quaternion &qap = state.qp[C[i].a];
        Quaternion &qbp = state.qp[C[i].b];

        Vec3 vqa = 2.f / dt * (qap.conjugate() * qa).vec();
        Vec3 vqb = 2.f / dt * (qbp.conjugate() * qb).vec();

        Vec3 dv = (vqb - vqa) * (1.0f - damping);

        vqa += dv;
        vqb -= dv;

        Quaternion vqaq, vqbq;
        vqaq.w() = 0.f;
        vqbq.w() = 0.f;

        vqaq.vec() = vqa;
        vqbq.vec() = vqb;

        qa = qap.coeffs() + 0.5f * dt * (qap * vqaq).coeffs();
        qb = qbp.coeffs() + 0.5f * dt * (qbp * vqbq).coeffs();

        qa.normalize();
        qb.normalize();
    }
};

struct apply_projection_particles {
    Vec3 *x;
    float *r;
    Vec6 *dx;
    int *id;
    uint8_t *a;

    apply_projection_particles(Vec3 *x, float *r, Vec6 *dx, int *id, uint8_t *a)
        : x(x), r(r), dx(dx), id(id), a(a) {}

    __device__ void operator()(int i) const {
        int k = id[i];
        if (a[k] == 0)
            return;

        if (dx[i][4] > 1e-6f)
            x[k] += dx[i].head<3>() / dx[i][4];

        if (dx[i][5] > 1e-6f)
            r[k] = fmaxf(MIN_RADIUS, r[k] + dx[i][3] / dx[i][5]);
    }
};

struct apply_projection_frames {
    Quaternion *x;
    Vec6 *dx;
    int *id;
    int N;
    uint8_t *a;

    apply_projection_frames(Quaternion *x, Vec6 *dx, int *id, uint8_t *a, int N)
        : x(x), dx(dx), id(id), N(N), a(a) {}

    __device__ void operator()(int i) const {
        int k = id[i] - N;
        if (a[k] == 0)
            return;

        if (dx[i][4] > 1e-6f)
            x[k].coeffs() += dx[i].head<4>() / dx[i][4];
        x[k].normalize();
    }
};

struct generate_pills_proxys {
    Vec3 *x;
    float *r;
    Vec2i *pills;
    Vec3 *sx;
    float *sr;

    generate_pills_proxys(Vec3 *x, float *r, Vec2i *pills, Vec3 *sx, float *sr)
        : x(x), r(r), pills(pills), sx(sx), sr(sr) {}

    __device__ void operator()(int i) const {
        int a = pills[i][0];
        int b = pills[i][1];
        Vec3 s0 = x[a];
        Vec3 s1 = x[b];
        float r0 = r[a];
        float r1 = r[b];
        Vec3 d = s1 - s0;
        float l = d.norm();
        Vec3 dl = d / (l + FLT_EPSILON);
        sx[i] = (s1 + s0 + dl * (r1 - r0)) / 2;
        sr[i] = (l + r0 + r1) / 2;
    }
};

struct generate_collisions {
    Vec2i *pills;
    const Vec2i *coll_pairs;
    C_collision *C;

    generate_collisions(Vec2i *pills, const Vec2i *coll_pairs, C_collision *C)
        : pills(pills), coll_pairs(coll_pairs), C(C) {}

    __device__ void operator()(int i) const {
        int a = coll_pairs[i][0];
        int b = coll_pairs[i][1];
        C[i].a = pills[a];
        C[i].b = pills[b];
        C[i].enabled = true;
    }
};

struct collision_filter {
    Vec2i *pills;
    int *group;
    CudaStatePtr S;

    collision_filter(Vec2i *pills, int *group, CudaStatePtr S)
        : pills(pills), group(group), S(S) {}

    __device__ bool operator()(const Vec2i &c) {
        int a0 = pills[c[0]][0];
        int a1 = pills[c[0]][1];
        int b0 = pills[c[1]][0];
        int b1 = pills[c[1]][1];
        int zeroa = S.w[a0] < 1e-6f || S.w[a1] < 1e-6f;
        int zerob = S.w[b0] < 1e-6f || S.w[b1] < 1e-6f;

        return group[c[0]] != group[c[1]] && (zeroa + zerob < 2);
    }
};

CudaSolver::CudaSolver() { gpu = new CudaSolverData(); }

CudaSolver::~CudaSolver() {
    // delete gpu;
}

template <typename T> struct DisabledPredicate {
    bool operator()(const T &constraint) { return !constraint.enabled; }
};

template <typename CPUVec, typename GPUVec>
void upload_and_filter(GPUVec &gpu_vec, const CPUVec &cpu_vec) {
    using T = typename GPUVec::value_type;
    gpu_vec = cpu_vec;
    if (gpu_vec.size() > 0) {
        gpu_vec.erase(thrust::remove_if(thrust::device, gpu_vec.begin(),
                                        gpu_vec.end(), DisabledPredicate<T>()),
                      gpu_vec.end());
    }
}

double CudaSolver::solve(ConstraintsCPU &constraints, SimulationState &state,
                         const std::vector<Vec2i> &pills,
                         const std::vector<int> &group, float dt, const Vec3 &g,
                         int iterations, bool floor, float damping) {
    CudaConstraints &C = gpu->C;
    CudaSimData &S = gpu->S;
    CudaProjections &Pc = gpu->Pc;
    CudaProjections &Pt = gpu->Pt;
    CudaProjections &Pp = gpu->Pp;

    // CPU -> GPU
    S.X.x = state.x;
    S.X.q = state.q;
    S.X.r = state.r;

    S.Xp.x = state.xp;
    S.Xp.q = state.qp;
    S.Xp.r = state.rp;

    S.Xi.x = state.xi;
    S.Xi.q = state.qi;
    S.Xi.r = state.ri;

    S.b = state.b;
    S.bp = state.bp;
    S.bi = state.bi;

    S.w = state.w;
    S.wq = state.wq;
    S.wr = state.wr;

    S.xa = state.xa;
    S.qa = state.qa;

    thrust::device_vector<Vec2i> gpu_pills = pills;
    thrust::device_vector<int> pill_groups = group;

    int N = state.x.size(); // particles count
    int M = state.q.size(); // pills count

    upload_and_filter(C.dist, constraints.distance);
    upload_and_filter(C.distmax, constraints.distancemax);
    upload_and_filter(C.skinning, constraints.skinning);
    upload_and_filter(C.vol, constraints.volume);
    upload_and_filter(C.vol2, constraints.volume2);
    upload_and_filter(C.bend, constraints.bend);
    upload_and_filter(C.stretch, constraints.stretch);
    upload_and_filter(C.bilap, constraints.bilap);
    upload_and_filter(C.shape, constraints.shape);
    upload_and_filter(C.shape2, constraints.shape2);
    upload_and_filter(C.radius, constraints.radius);
    upload_and_filter(C.touch, constraints.touch);

    tic();

    // time integration
    thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(N),
                     V_integration(CudaStatePtr(S), dt, g, damping));
    thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(M),
                     Vq_integration(CudaStatePtr(S), dt, damping));
    thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator((int)C.skinning.size()),
                     C_skinning_solve(ptr(C.skinning), CudaStatePtr(S)));

    float t_velocity = toc();
    thrust::device_vector<Vec3> sp(M);
    thrust::device_vector<float> sr(M);
    thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(M),
                     generate_pills_proxys(ptr(S.X.x), ptr(S.X.r),
                                           ptr(gpu_pills), ptr(sp), ptr(sr)));
    if (M < 2) {
        C.collision.resize(0);
    } else {
        const thrust::device_vector<Vec2i> &coll_pairs =
            gpu->cgrid.test_particles(sp, sr, COLLISION_MARGIN);
        thrust::device_vector<Vec2i> coll_pairs_filtered(coll_pairs.size());
        auto valid_coll_end =
            thrust::copy_if(thrust::device, coll_pairs.begin(),
                            coll_pairs.end(), coll_pairs_filtered.begin(),
                            collision_filter(ptr(gpu_pills), ptr(pill_groups),
                                             CudaStatePtr(S)));
        coll_pairs_filtered.erase(valid_coll_end, coll_pairs_filtered.end());
        int K = coll_pairs_filtered.size();
        C.collision.resize(K);
        thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(K),
                         generate_collisions(ptr(gpu_pills),
                                             ptr(coll_pairs_filtered),
                                             ptr(C.collision)));
    }

    float t_collision = toc();
    ConstraintsInfo cInfo;
    cInfo.add("distance", C.dist.size(), 2, 1);
    cInfo.add("volume", C.vol.size(), 2, 1);
    cInfo.add("volume2", C.vol2.size(), 3, 1);
    cInfo.add("bend", C.bend.size(), 2, 3);
    cInfo.add("stretch", C.stretch.size(), 3, 3);
    cInfo.add("bilap", C.bilap.size(), 1, 1);
    cInfo.add("shape", C.shape.size(), SHAPE_MATCHING_MAX, 1);
    cInfo.add("shape2", C.shape2.size(), 3 * SHAPE_MATCHING_MAX, 1);
    cInfo.add("radius", C.radius.size(), 1, 1);
    cInfo.add("collision", C.collision.size(), 4, 0);
    cInfo.add("touch", C.touch.size(), 2, 0);

    int np = cInfo.get_np();
    int nl = cInfo.get_nl();
    std::map<std::string, int> o = cInfo.get_o();
    std::map<std::string, int> ol = cInfo.get_ol();

    int n_cst = C.dist.size() + C.vol.size() + C.bend.size() +
                C.stretch.size() + C.shape.size() + C.radius.size() +
                C.collision.size() + C.vol2.size() + C.shape2.size();
    bool permutation_built = false;
    Pc.resize(np);
    Pt.resize(np);
    Pp.resize(N + M);
    thrust::device_vector<float> L(nl); // XPBD
    thrust::fill(L.begin(), L.end(), 0.f);
    thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator((int)C.bend.size()),
                     bend_damping{CudaStatePtr(S), ptr(C.bend), dt, 0.98f});

    for (int i = 0; i < iterations; i++) {
        bool collisions_only = (i % 2 == 1) || i == -1;
        Pc.setZero();
        Pp.setZero();
        if (!collisions_only) {
            thrust::for_each(
                thrust::device, thrust::make_counting_iterator(0),
                thrust::make_counting_iterator((int)C.dist.size()),
                C_distance_solve(ptr(C.dist), CudaStatePtr(S),
                                 CudaProjectionsPtr(Pc, o["distance"]),
                                 ptr(L, ol["distance"]), dt));
            thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator((int)C.vol.size()),
                             C_volume_solve(ptr(C.vol), CudaStatePtr(S),
                                            CudaProjectionsPtr(Pc, o["volume"]),
                                            ptr(L, ol["volume"]), dt));
            thrust::for_each(
                thrust::device, thrust::make_counting_iterator(0),
                thrust::make_counting_iterator((int)C.vol2.size()),
                C_volume2_solve(ptr(C.vol2), CudaStatePtr(S),
                                CudaProjectionsPtr(Pc, o["volume2"]),
                                ptr(L, ol["volume2"]), dt));
            thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator((int)C.bend.size()),
                             C_bend_solve(ptr(C.bend), CudaStatePtr(S),
                                          CudaProjectionsPtr(Pc, o["bend"]), N,
                                          ptr(L, ol["bend"]), dt));
            thrust::for_each(
                thrust::device, thrust::make_counting_iterator(0),
                thrust::make_counting_iterator((int)C.stretch.size()),
                C_stretch_solve(ptr(C.stretch), CudaStatePtr(S),
                                CudaProjectionsPtr(Pc, o["stretch"]), N,
                                ptr(L, ol["stretch"]), dt));
            thrust::for_each(
                thrust::device, thrust::make_counting_iterator(0),
                thrust::make_counting_iterator((int)C.bilap.size()),
                C_bilap_solve(ptr(C.bilap), CudaStatePtr(S),
                              CudaProjectionsPtr(Pc, o["bilap"]),
                              ptr(L, ol["bilap"]), dt));
            thrust::for_each(
                thrust::device, thrust::make_counting_iterator(0),
                thrust::make_counting_iterator((int)C.shape.size()),
                C_shape_solve(ptr(C.shape), CudaStatePtr(S),
                              CudaProjectionsPtr(Pc, o["shape"]),
                              ptr(L, ol["shape"]), dt));
            thrust::for_each(
                thrust::device, thrust::make_counting_iterator(0),
                thrust::make_counting_iterator((int)C.shape2.size()),
                C_shape2_solve(ptr(C.shape2), CudaStatePtr(S),
                               CudaProjectionsPtr(Pc, o["shape2"]), N,
                               ptr(L, ol["shape2"]), dt));
            thrust::for_each(
                thrust::device, thrust::make_counting_iterator(0),
                thrust::make_counting_iterator((int)C.radius.size()),
                C_radius_solve(ptr(C.radius), CudaStatePtr(S),
                               CudaProjectionsPtr(Pc, o["radius"]),
                               ptr(L, ol["radius"]), dt));
            thrust::for_each(
                thrust::device, thrust::make_counting_iterator(0),
                thrust::make_counting_iterator((int)C.touch.size()),
                C_touch_solve(ptr(C.touch), CudaStatePtr(S),
                              CudaProjectionsPtr(Pc, o["touch"]),
                              ptr(L, ol["touch"]), dt));
        }
        thrust::for_each(
            thrust::device, thrust::make_counting_iterator(0),
            thrust::make_counting_iterator((int)C.collision.size()),
            C_collision_solve(ptr(C.collision), CudaStatePtr(S),
                              CudaProjectionsPtr(Pc, o["collision"])));

        if (!permutation_built) {
            gpu->c_perm.resize(np);
            thrust::sequence(thrust::device, gpu->c_perm.begin(),
                             gpu->c_perm.end());
            auto vals_begin = thrust::make_zip_iterator(
                thrust::make_tuple(Pc.dx.begin(), gpu->c_perm.begin()));
            thrust::sort_by_key(Pc.id.begin(), Pc.id.end(), vals_begin);
            Pt = Pc;
            permutation_built = true;
        } else {
            auto src_begin = thrust::make_zip_iterator(
                thrust::make_tuple(Pc.dx.begin(), Pc.id.begin()));
            auto dst_begin = thrust::make_zip_iterator(
                thrust::make_tuple(Pt.dx.begin(), Pt.id.begin()));
            thrust::gather(thrust::device, gpu->c_perm.begin(),
                           gpu->c_perm.end(), src_begin, dst_begin);
        }
        auto new_end =
            thrust::reduce_by_key(thrust::device, Pt.id.begin(), Pt.id.end(),
                                  Pt.dx.begin(), Pp.id.begin(), Pp.dx.begin())
                .first;
        auto f_start = thrust::lower_bound(Pp.id.begin(), new_end, N);
        int proj_count = new_end - Pp.id.begin();
        int p_count = f_start - Pp.id.begin();
        thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(p_count),
                         apply_projection_particles(ptr(S.X.x), ptr(S.X.r),
                                                    ptr(Pp.dx), ptr(Pp.id),
                                                    ptr(S.xa)));
        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator(p_count),
                         thrust::make_counting_iterator(proj_count),
                         apply_projection_frames(ptr(S.X.q), ptr(Pp.dx),
                                                 ptr(Pp.id), ptr(S.qa), N));

        if (floor)
            thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(N),
                             floor_friction(CudaStatePtr(S)));
    }
    float t_solve = toc();

    // GPU -> CPU
    thrust::copy(S.X.x.begin(), S.X.x.end(), state.x.begin());
    thrust::copy(S.Xp.x.begin(), S.Xp.x.end(), state.xp.begin());
    thrust::copy(S.X.q.begin(), S.X.q.end(), state.q.begin());
    thrust::copy(S.Xp.q.begin(), S.Xp.q.end(), state.qp.begin());
    thrust::copy(S.X.r.begin(), S.X.r.end(), state.r.begin());
    thrust::copy(S.Xp.r.begin(), S.Xp.r.end(), state.rp.begin());
    thrust::copy(C.shape.begin(), C.shape.end(), constraints.shape.begin());
    thrust::copy(C.shape2.begin(), C.shape2.end(), constraints.shape2.begin());

    return t_solve;
}

} // namespace viper
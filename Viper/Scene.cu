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

#include "Scene.h"
#include "Utils.h"
#include "nnsearch.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <thrust/iterator/transform_iterator.h>

int current_frame = 0;

namespace viper {

double Scene::step(float dt, int iterations, bool floor, float damping) {
    int M = pills.size();
    std::vector<int> group(M);
    for (int i = 0; i < M; i++)
        group[i] = pInfo[pills[i][0]].group;

    current_frame++;
    double t = solver.solve(constraints, state, pills, group, dt,
                            -10.f * gravity_strength * upDirection, iterations,
                            floor, damping);

    return t;
}

Scene::Scene() { clear(); }

void Scene::reset() {
    for (int i = 0; i < state.x.size(); i++) {
        state.x[i] = state.xi[i];
        state.xp[i] = state.xi[i];
    }

    for (int i = 0; i < state.r.size(); i++) {
        state.r[i] = state.ri[i];
    }

    for (int i = 0; i < state.q.size(); i++) {
        state.q[i] = state.qi[i];
        state.qp[i] = state.qi[i];
    }

    for (int i = 0; i < constraints.shape.size(); i++)
        constraints.shape[i].q = Quaternion::Identity();

    current_frame = 0;
}

int Scene::addParticle(const Vec3 &p, float r, float w, bool kinematic) {
    state.x.push_back(p);
    state.xp.push_back(p);
    state.v.push_back(Vec3::Zero());
    state.w.push_back(w);
    state.wr.push_back(1.0);
    state.r.push_back(r);
    state.rp.push_back(r);

    state.xi.push_back(p);
    state.ri.push_back(r);

    state.xa.push_back(!kinematic);
    state.xai.push_back(!kinematic);

    ParticleInfo info;
    pInfo.push_back(info);

    return state.x.size() - 1;
}

Id Scene::addPill(Id a, Id b, bool kinematic) {
    pills.push_back(Vec2i(a, b));
    Id id = pills.size() - 1;
    pInfo[a].isFiber = true;
    pInfo[b].isFiber = true;

    Vec3 ab = (state.x[b] - state.x[a]).normalized();
    Quaternion q = Quaternion::FromTwoVectors(Vec3::UnitZ(), ab);

    state.q.push_back(q);
    state.qp.push_back(q);
    state.qi.push_back(q);
    state.wq.push_back(1.0f);
    state.vq.push_back(Vec3::Zero());

    state.qa.push_back(!kinematic);
    state.qai.push_back(!kinematic);

    return id;
}

Scene::~Scene() { clear(); }

void Scene::clear() {
    state.clear();
    pInfo.clear();
    pills.clear();

    constraints.distancemax.clear();
    constraints.distance.clear();
    constraints.skinning.clear();
    constraints.volume.clear();
    constraints.volume2.clear();
    constraints.bend.clear();
    constraints.stretch.clear();
    constraints.radius.clear();
    constraints.bilap.clear();
    constraints.shape.clear();
    constraints.touch.clear();
    constraints.shape2.clear();
}

Id Scene::getNewGroup() {
    int maxGroup = -1;
    for (ParticleInfo &i : pInfo) {
        if (i.group > maxGroup)
            maxGroup = i.group;
    }
    return maxGroup + 1;
}

void Scene::addRod(int n, Vec3 start, Vec3 step, float r, bool volume,
                   IntArray &pIds, IntArray &rIds, float stretch_eta,
                   float volume_eta, float bilap_eta) {
    pIds.clear();
    rIds.clear();
    for (int i = 0; i < n; i++) {
        Vec3 p = start + i * step;
        Id id = addParticle(p, r);
        pIds.push_back(id);
    }

    for (int i = 1; i < n; i++) {
        Id a = pIds[i - 1];
        Id b = pIds[i];
        Id id = addPill(a, b);
        rIds.push_back(id);
        float d = step.norm();
        constraints.stretch.push_back(C_stretch(a, b, id, d, stretch_eta));
        if (volume) {
            constraints.volume2.push_back(
                C_volume2(a, b, id, state, volume_eta));
        }
    }

    for (int i = 1; i < n - 1; i++) {
        constraints.bend.push_back(C_bend(rIds[i - 1], rIds[i], state, 0.0f));
    }

    for (int i = 1; i < n - 1; i++) {
        constraints.bilap.push_back(C_bilap(pIds, i, bilap_eta));
    }
}

void Scene::setUpDirection(const Vec3 &dir) { upDirection = dir; }

} // namespace viper
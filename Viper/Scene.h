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

#include <unordered_map>

#include "Common.h"
#include "CudaConstraints.h"
#include "CudaSolver.h"
#include "Mesh.h"
#include "Particle.h"
#include "Rod.h"

namespace viper {

struct Scene {
    Scene();
    ~Scene();
    Id addParticle(const Vec3 &p = Vec3::Zero(), float r = PARTICLE_RADIUS,
                   float w = 1.0f, bool kinematic = false);
    Id addPill(Id a, Id b, bool kinematic = false);
    void addRod(int n, Vec3 start, Vec3 step, float r, bool volume,
                IntArray &pIds, IntArray &rIds, float stretch_eta = 0.0f,
                float volume_eta = 0.0f, float bilap_eta = 0.0f);
    double step(float timestep, int iterations = 20, bool floor = false,
                float damping = 0.9999f);
    void clear();
    void reset();
    virtual void initialize(){};
    virtual void kinematicStep(float time){};
    Id getNewGroup();
    void setUpDirection(const Vec3 &dir);

    SimulationState state;

    std::vector<Vec2i> pills;
    std::vector<ParticleInfo> pInfo;

    Vec3 upDirection = Vec3::UnitY();
    float gravity_strength = 1.0;
    CudaSolver solver;

    ConstraintsCPU constraints;
};

} // namespace viper
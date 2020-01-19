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

namespace viper {

class SimulationState {
  public:
    void swap() {
        x.swap(xp);
        q.swap(qp);
        r.swap(rp);
    }
    void clear() {
        x.clear();
        q.clear();
        r.clear();

        xp.clear();
        qp.clear();
        rp.clear();

        xi.clear();
        qi.clear();
        ri.clear();

        w.clear();
        wr.clear();

        v.clear();
        vq.clear();
        vr.clear();
    }

    // Current state
    Vec3Array x;       // current position
    QuaternionArray q; // current orientation
    FloatArray r;      // current radii

    // Previous state
    Vec3Array xp;       // previous position
    QuaternionArray qp; // previous orientations
    FloatArray rp;      // previous scale

    // Initial state
    Vec3Array xi;       // initial position
    QuaternionArray qi; // initial orientation
    FloatArray ri;      // initial radii

    std::vector<uint8_t> xa;
    std::vector<uint8_t> xai;
    std::vector<uint8_t> qa;
    std::vector<uint8_t> qai;

    std::vector<Matrix4> b;
    std::vector<Matrix4> bp;
    std::vector<Matrix4> bi;

    // Weights
    FloatArray w;  // position weight (inverse mass)
    FloatArray wq; // orientation weight
    FloatArray wr; // radius weight

    // Velocities
    Vec3Array v;   // positional velocity
    Vec3Array vq;  // angular velocity
    FloatArray vr; // radii velocity
};

} // namespace viper
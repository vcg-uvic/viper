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
#include "State.h"

namespace viper {

struct CudaSolverData;

class CudaSolver {
  public:
    CudaSolver();
    ~CudaSolver();
    double solve(ConstraintsCPU &constraints, SimulationState &state,
                 const std::vector<Vec2i> &pills, const std::vector<int> &group,
                 float dt, const Vec3 &g, int iterations, bool floor = false,
                 float damping = 0.9999f);

  private:
    CudaSolverData *gpu;
};

} // namespace viper
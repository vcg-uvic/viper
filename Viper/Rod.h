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

struct RodBundle {
    RodBundle() {}
    RodBundle(const std::vector<IntArray> &pIds,
              const std::vector<IntArray> &rIds)
        : particles(pIds), pillsSlices(rIds) {
        for (const IntArray &p : rIds) {
            for (int idx : p)
                pills.push_back(idx);
        }
    }

    std::vector<IntArray> particles;
    std::vector<IntArray> pillsSlices;
    IntArray pills;
    std::vector<std::vector<float>> skin_weights;
    std::vector<std::vector<int>> skin_bone_ids;
};

} // namespace viper
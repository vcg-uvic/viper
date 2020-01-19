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

#include <thrust/device_vector.h>

#include "Common.h"

namespace viper {

class CollisionGrid {
  public:
    static Vec4 b_sphere;

    const thrust::device_vector<Vec2i> &
    test_particles(const thrust::device_vector<Vec3> &c,
                   const thrust::device_vector<float> &r, float margin);

  private:
    thrust::device_vector<Vec3> c;
    thrust::device_vector<float> r;
    thrust::device_vector<int> i;

    thrust::device_vector<int> part_cell_ids;

    thrust::device_vector<int> cell_starts;
    thrust::device_vector<int> cell_ends;

    thrust::device_vector<int> parts_per_cell;

    thrust::device_vector<int> neighbours_per_pair;

    thrust::device_vector<int> pair_collision_starts;

    thrust::device_vector<int> pairs_per_coll_group;

    thrust::device_vector<int> collision_pair_ids;

    thrust::device_vector<Vec2i> collisions;
};

} // namespace viper
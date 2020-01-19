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

#include <chrono>
#include <iomanip>

#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>

#include "CudaUtils.cuh"

#define SK_DEVICE __host__ __device__

namespace viper {

Vec4 CollisionGrid::b_sphere;

const thrust::device_vector<Vec2i> &
CollisionGrid::test_particles(const thrust::device_vector<Vec3> &c_in,
                              const thrust::device_vector<float> &r_in,
                              float eps) {

    float total_ms = 0;
    float time_ms = 0;

    auto zero_begin = thrust::make_counting_iterator(0);

    int n_particles = c_in.size();

    // Input particle positions
    c = c_in;
    // Input particles radii
    r.resize(r_in.size());
    auto add_margin = [=] SK_DEVICE(float x) -> float {
        return x + 0.5f * eps;
    };
    thrust::transform(thrust::device, r_in.begin(), r_in.end(), r.begin(),
                      add_margin);

    // Particle IDs
    i.clear();
    i.insert(i.end(), zero_begin, zero_begin + n_particles);

    int max_i =
        thrust::max_element(thrust::device, r.begin(), r.end()) - r.begin();

    float max_radius = r_in[max_i];

    auto part_to_sphere =
        [=] SK_DEVICE(thrust::tuple<Vec3, float, int> part) -> Vec4 {
        return Vec4(part.get<0>()[0], part.get<0>()[1], part.get<0>()[2],
                    part.get<1>());
    };

    auto particles_begin = thrust::make_zip_iterator(
        thrust::make_tuple(c.begin(), r.begin(), i.begin()));

    auto spheres_begin =
        thrust::make_transform_iterator(particles_begin, part_to_sphere);

    using Vec6 = Eigen::Matrix<float, 6, 1>;

    Vec3 c_init = c_in[0];
    float r_init = r_in[0];

    auto sphere_bbox = [=] SK_DEVICE(Vec4 s) -> Vec6 {
        Vec6 bbox;
        bbox.segment<3>(0) = s.head<3>() - Vec3::Ones() * s[3];
        bbox.segment<3>(3) = s.head<3>() + Vec3::Ones() * s[3];
        return bbox;
    };

    auto bbox_union = [=] SK_DEVICE(Vec6 b0, Vec6 b1) -> Vec6 {
        Vec6 result;
        result.segment<3>(0) = b0.segment<3>(0).cwiseMin(b1.segment<3>(0));
        result.segment<3>(3) = b0.segment<3>(3).cwiseMax(b1.segment<3>(3));
        return result;
    };

    auto bboxes_begin =
        thrust::make_transform_iterator(spheres_begin, sphere_bbox);

    Vec6 bbox_init = sphere_bbox(Vec4(c_init[0], c_init[1], c_init[2], r_init));

    Vec6 bbox =
        thrust::reduce(thrust::device, bboxes_begin, bboxes_begin + n_particles,
                       bbox_init, bbox_union);

    Vec3 bbox_min = bbox.segment<3>(0);
    Vec3 bbox_max = bbox.segment<3>(3);

    b_sphere.head<3>() = (bbox_max + bbox_min) / 2;
    b_sphere[3] = (bbox_max - bbox_min).norm() / 2;

    float grid_width = bbox_max[0] - bbox_min[0];
    int grid_res = grid_width / (2 * max_radius);
    grid_res = max(min(grid_res, 32), 1);

    int n_cells = grid_res * grid_res * grid_res;

    float cell_size = grid_width / grid_res;

    auto center_to_cell_id = [=] SK_DEVICE(Vec3 c) {
        Vec3i i = ((c - bbox_min) / cell_size)
                      .cast<int>()
                      .cwiseMax(Vec3i::Zero())
                      .cwiseMin(Vec3i::Ones() * (grid_res - 1));
        return i[0] + grid_res * i[1] + grid_res * grid_res * i[2];
    };

    auto part_cell_ids_gen =
        thrust::make_transform_iterator(c.begin(), center_to_cell_id);

    // ID of cell containing each particle
    part_cell_ids.clear();
    part_cell_ids.insert(part_cell_ids.end(), part_cell_ids_gen,
                         part_cell_ids_gen + n_particles);

    // Sort particle list by id of cells
    thrust::sort_by_key(thrust::device, part_cell_ids.begin(),
                        part_cell_ids.end(), particles_begin);

    // Indices of first particle for each cell
    cell_starts.clear();
    cell_starts.resize(n_cells, 0);
    cell_ends.clear();
    cell_ends.resize(n_cells, 0);

    auto part_cell_ids_begin = thrust::raw_pointer_cast(part_cell_ids.data());
    auto cell_starts_begin = thrust::raw_pointer_cast(cell_starts.data());
    auto cell_ends_begin = thrust::raw_pointer_cast(cell_ends.data());

    auto write_cell_starts = [=] SK_DEVICE(int i) {
        int id1 = part_cell_ids_begin[i];

        if (i == 0) {
            cell_starts_begin[id1] = i;
        } else {
            int id0 = part_cell_ids_begin[i - 1];
            if (id0 != id1) {
                cell_starts_begin[id1] = i;
                cell_ends_begin[id0] = i;
            }
        }
        if (i == n_particles - 1) {
            cell_ends_begin[id1] = n_particles;
        }
    };

    // Write out start indices of each cell in sorted particle list
    thrust::for_each(thrust::device, zero_begin, zero_begin + n_particles,
                     write_cell_starts);

    // Number of particles in each cell
    parts_per_cell.clear();
    parts_per_cell.resize(n_cells, 0);

    auto count_cell_parts = [=] SK_DEVICE(int cell) {
        return cell_ends_begin[cell] - cell_starts_begin[cell];
    };

    thrust::transform(thrust::device, zero_begin, zero_begin + n_cells,
                      parts_per_cell.begin(), count_cell_parts);

    auto parts_per_cell_begin = thrust::raw_pointer_cast(parts_per_cell.data());

    auto n_neighbours = [=] SK_DEVICE(int pair_id) {
        int nhbr_off = pair_id % 27;
        int particle_id = pair_id / 27;
        int cell_id = part_cell_ids_begin[particle_id];

        int c_i = cell_id % grid_res;
        int c_j = (cell_id / grid_res) % grid_res;
        int c_k = cell_id / (grid_res * grid_res);

        int i = (nhbr_off % 3) - 1;
        int j = ((nhbr_off / 3) % 3) - 1;
        int k = (nhbr_off / 9) - 1;

        int n_i = c_i + i;
        int n_j = c_j + j;
        int n_k = c_k + k;

        int nhbr_cell_id = n_i + grid_res * n_j + grid_res * grid_res * n_k;

        bool inside = n_i >= 0 && n_i < grid_res && n_j >= 0 &&
                      n_j < grid_res && n_k >= 0 && n_k < grid_res;

        int count = 0;
        if (inside) {
            count = parts_per_cell_begin[nhbr_cell_id];
        }

        return count;
    };

    auto neighbour_count_begin =
        thrust::make_transform_iterator(zero_begin, n_neighbours);

    int n_pairs = n_particles * 27;

    // Count potential collisions for each particle-cell pair
    // NOTE: promote from 27 threads per particle to 32 for warp coherence if
    // slow
    neighbours_per_pair.clear();
    neighbours_per_pair.insert(neighbours_per_pair.end(), neighbour_count_begin,
                               neighbour_count_begin + n_pairs);

    // Index of first collision for each particle-cell pair
    pair_collision_starts.resize(n_pairs);

    // tic();

    // Find start indices for cell-particle_pairs
    thrust::exclusive_scan(thrust::device, neighbours_per_pair.begin(),
                           neighbours_per_pair.end(),
                           pair_collision_starts.begin());

    int n_collisions =
        pair_collision_starts.back() + neighbours_per_pair.back();

    // Number of particle-cell pairs pointing to a group of collisions
    pairs_per_coll_group.clear();
    pairs_per_coll_group.resize(n_pairs, 1);

    // In-place scan to compute above count
    thrust::inclusive_scan_by_key(thrust::device, pair_collision_starts.begin(),
                                  pair_collision_starts.end(),
                                  pairs_per_coll_group.begin(),
                                  pairs_per_coll_group.begin());

    // Compute vector with number of pairs pointing to the start of each
    // collision group
    collision_pair_ids.clear();
    collision_pair_ids.resize(n_collisions, 0);
    thrust::scatter_if(thrust::device, pairs_per_coll_group.begin(),
                       pairs_per_coll_group.end(),
                       pair_collision_starts.begin(),
                       neighbours_per_pair.begin(), collision_pair_ids.begin());

    // Compute pair id of each collision
    thrust::inclusive_scan(thrust::device, collision_pair_ids.begin(),
                           collision_pair_ids.end(),
                           collision_pair_ids.begin());

    // Subtract one to get indices
    thrust::for_each(thrust::device, collision_pair_ids.begin(),
                     collision_pair_ids.end(),
                     [] SK_DEVICE(int &i) { i -= 1; });

    auto collision_pair_ids_begin =
        thrust::raw_pointer_cast(collision_pair_ids.data());
    auto pair_collision_starts_begin =
        thrust::raw_pointer_cast(pair_collision_starts.data());

    auto collision_generator = [=] SK_DEVICE(int collision_id) {
        int pair_id = collision_pair_ids_begin[collision_id];
        int collision_start = pair_collision_starts_begin[pair_id];
        int part_offset = collision_id - collision_start;

        int nhbr_off = pair_id % 27;
        int particle_id = pair_id / 27;
        int cell_id = part_cell_ids_begin[particle_id];

        int c_i = cell_id % grid_res;
        int c_j = (cell_id / grid_res) % grid_res;
        int c_k = cell_id / (grid_res * grid_res);

        int i = (nhbr_off % 3) - 1;
        int j = ((nhbr_off / 3) % 3) - 1;
        int k = (nhbr_off / 9) - 1;

        int n_i = c_i + i;
        int n_j = c_j + j;
        int n_k = c_k + k;

        int nhbr_cell_id = n_i + grid_res * n_j + grid_res * grid_res * n_k;

        int cell_start = cell_starts_begin[nhbr_cell_id];

        int other_particle_id = cell_start + part_offset;

        return Vec2i(particle_id, other_particle_id);
    };

    auto collisions_gen =
        thrust::make_transform_iterator(zero_begin, collision_generator);

    // Generate particle ids for all potential collisions
    collisions.clear();
    collisions.insert(collisions.end(), collisions_gen,
                      collisions_gen + n_collisions);

    auto c_begin = thrust::raw_pointer_cast(c.data());
    auto r_begin = thrust::raw_pointer_cast(r.data());

    auto not_colliding = [=] SK_DEVICE(Vec2i collision) {
        Vec3 c0 = c_begin[collision[0]];
        Vec3 c1 = c_begin[collision[1]];
        float r0 = r_begin[collision[0]];
        float r1 = r_begin[collision[1]];

        return (c1 - c0).norm() > (r0 + r1) || (collision[0] >= collision[1]);
    };

    auto valid_coll_end = thrust::remove_if(thrust::device, collisions.begin(),
                                            collisions.end(), not_colliding);

    collisions.erase(valid_coll_end, collisions.end());

    auto i_begin = thrust::raw_pointer_cast(i.data());

    auto permute_indices = [=] SK_DEVICE(Vec2i & coll) {
        coll[0] = i_begin[coll[0]];
        coll[1] = i_begin[coll[1]];
    };

    thrust::for_each(thrust::device, collisions.begin(), collisions.end(),
                     permute_indices);

    return collisions;
}

} // namespace viper
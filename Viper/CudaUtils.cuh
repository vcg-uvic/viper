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
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace viper {

template <class T> T *ptr(thrust::device_vector<T> &v, int i = 0) {
    return thrust::raw_pointer_cast(v.data()) + i;
}

template <class T> const T *ptr(const thrust::device_vector<T> &v, int i = 0) {
    return thrust::raw_pointer_cast(v.data()) + i;
}

template <class T>
__host__ inline void printVector(const thrust::device_vector<T> &gpu_v) {
    thrust::host_vector<T> v = gpu_v;
    for (auto item : v)
        std::cout << " | " << item;
    std::cout << " |" << std::endl;
}

template <>
inline __host__ void
printVector<Vec3>(const thrust::device_vector<Vec3> &gpu_v) {
    thrust::host_vector<Vec3> v = gpu_v;
    for (auto item : v)
        std::cout << " | " << item.transpose();
    std::cout << " |" << std::endl;
}

template <>
inline __host__ void
printVector<Vec6>(const thrust::device_vector<Vec6> &gpu_v) {
    thrust::host_vector<Vec6> v = gpu_v;
    for (auto item : v)
        std::cout << " | " << item.transpose() << std::endl;
    std::cout << " |" << std::endl;
}

void tic();

double toc();

} // namespace viper
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

#include "CudaUtils.cuh"
#include <chrono>

namespace viper {

decltype(std::chrono::system_clock::now()) start_time;

void tic() {
    cudaDeviceSynchronize();
    start_time = std::chrono::system_clock::now();
}

double toc() {
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();

    auto dur = end - start_time;
    auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(dur).count();

    return (double)us / 1000.0;
}

} // namespace viper
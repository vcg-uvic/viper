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

#include <map>
#include <vector>

#include <Eigen/Dense>

namespace cow {

using Vec2 = Eigen::Vector2f;
using Vec3 = Eigen::Vector3f;
using Vec4 = Eigen::Vector4f;

using Vec2i = Eigen::Vector2i;
using Vec3i = Eigen::Vector3i;
using Vec4i = Eigen::Vector4i;

inline void get_octopus(std::vector<Vec4> &spheres, std::vector<Vec2i> &pills,
                        std::vector<int> &control_pills,
                        std::vector<float> &masses,
                        std::vector<float> &compliances) {

    spheres.clear();
    pills.clear();
    control_pills.clear();
    masses.clear();
    compliances.clear();

    std::map<std::string, Vec4> sphere_vals = {
        {"knee", Vec4(-0.35362565517425537, -0.813965916633606,
                      0.017165500670671463, 0.14786656200885773)},
        {"knee.001", Vec4(0.3370053768157959, -0.8209866285324097,
                          0.017165495082736015, 0.14786659181118011)},
        {"knee.002", Vec4(-0.8256120681762695, -0.3255097270011902,
                          0.017165491357445717, 0.14786656200885773)},
        {"knee.003", Vec4(-0.8139659762382507, 0.3536257743835449,
                          0.017165502533316612, 0.14786644279956818)},
        {"knee.004", Vec4(-0.30087363719940186, 0.8349052667617798,
                          0.017165515571832657, 0.14786657691001892)},
        {"knee.005", Vec4(0.4012768268585205, 0.7915606498718262,
                          0.017165493220090866, 0.14786645770072937)},
        {"knee.006", Vec4(0.85039883852005, 0.25379884243011475,
                          0.01716550998389721, 0.14786656200885773)},
        {"knee.007", Vec4(0.7984437942504883, -0.3874009847640991,
                          0.017165498808026314, 0.14786657691001892)},
        {"ankle", Vec4(-0.516542911529541, -1.1639161109924316,
                       -0.2540038824081421, 0.11486519873142242)},
        {"ankle.001", Vec4(0.47430869936943054, -1.1817569732666016,
                           -0.2540039122104645, 0.11486519873142242)},
        {"ankle.002", Vec4(-1.1882641315460205, -0.4577622413635254,
                           -0.2540039122104645, 0.11486519873142242)},
        {"ankle.003", Vec4(-1.1639163494110107, 0.5165430307388306,
                           -0.2540039122104645, 0.11486509442329407)},
        {"ankle.004", Vec4(-0.42230913043022156, 1.2013213634490967,
                           -0.2540038824081421, 0.11486519873142242)},
        {"ankle.005", Vec4(0.584661602973938, 1.1312336921691895,
                           -0.2540039122104645, 0.11486520618200302)},
        {"ankle.006", Vec4(1.223022222518921, 0.3545911908149719,
                           -0.2540038824081421, 0.11486519873142242)},
        {"ankle.007", Vec4(1.1412649154663086, -0.5648297071456909,
                           -0.2540039122104645, 0.11486519873142242)},
        {"hip", Vec4(0.20626629889011383, -0.4769735634326935,
                     0.06736935675144196, 0.14786657691001892)},
        {"hip.001", Vec4(-0.479807049036026, -0.19958660006523132,
                         0.06736935675144196, 0.14786657691001892)},
        {"hip.002", Vec4(-0.1981457769870758, -0.4804038107395172,
                         0.06736935675144196, 0.14786656200885773)},
        {"hip.003", Vec4(-0.480403870344162, 0.198145791888237,
                         0.06736935675144196, 0.14786644279956818)},
        {"hip.004", Vec4(-0.18526466190814972, 0.48551687598228455,
                         0.06736935675144196, 0.14786654710769653)},
        {"hip.005", Vec4(0.2262880653142929, 0.46780699491500854,
                         0.06736935675144196, 0.14786657691001892)},
        {"hip.006", Vec4(0.495101660490036, 0.1578734964132309,
                         0.06736935675144196, 0.14786642789840698)},
        {"hip.007", Vec4(0.4716850221157074, -0.2180892825126648,
                         0.06736935675144196, 0.14786657691001892)},
        {"foot", Vec4(-0.7747266888618469, -1.7187798023223877,
                      -0.636471152305603, 0.07699611037969589)},
        {"foot.001", Vec4(0.6920993328094482, -1.7536826133728027,
                          -0.636471152305603, 0.07699618488550186)},
        {"foot.002", Vec4(-1.7631759643554688, -0.6675468683242798,
                          -0.6364713907241821, 0.07699617743492126)},
        {"foot.003", Vec4(-1.7187803983688354, 0.7747265100479126,
                          -0.636471152305603, 0.07699616998434067)},
        {"foot.004", Vec4(-0.6149460077285767, 1.7822026014328003,
                          -0.636471152305603, 0.07699616998434067)},
        {"foot.005", Vec4(0.8752979636192322, 1.6698089838027954,
                          -0.636471152305603, 0.07699623703956604)},
        {"foot.006", Vec4(1.8137513399124146, 0.5145020484924316,
                          -0.6364708542823792, 0.07699617743492126)},
        {"foot.007", Vec4(1.6848303079605103, -0.8460220098495483,
                          -0.636471152305603, 0.07699617743492126)},
        {"eye_l", Vec4(0.24251265823841095, -0.38848432898521423,
                       0.8807172179222107, 0.24920815229415894)},
        {"eye_r", Vec4(-0.22834379971027374, -0.38848432898521423,
                       0.8807172179222107, 0.24920815229415894)},
        {"root", Vec4(0.0, 0.0, 0.17928314208984375, 0.35808438062667847)},
        {"head_b", Vec4(0.0, 0.0, 0.8439158797264099, 0.5126206874847412)},
        {"head_tf", Vec4(0.0, -0.019088756293058395, 1.1476688385009766,
                         0.5920186638832092)},
        {"head_tb", Vec4(0.0, 0.18671098351478577, 1.2015498876571655,
                         0.5920186638832092)}};

    std::map<std::string, float> sphere_masses = {
        {"foot", 0.3f},      {"foot.001", 0.3f},  {"foot.002", 0.3f},
        {"foot.003", 0.3f},  {"foot.004", 0.3f},  {"foot.005", 0.3f},
        {"foot.006", 0.3f},  {"foot.007", 0.3f},  {"knee", 0.3f},
        {"knee.001", 0.3f},  {"knee.002", 0.3f},  {"knee.003", 0.3f},
        {"knee.004", 0.3f},  {"knee.005", 0.3f},  {"knee.006", 0.3f},
        {"knee.007", 0.3f},  {"ankle", 0.3f},     {"ankle.001", 0.3f},
        {"ankle.002", 0.3f}, {"ankle.003", 0.3f}, {"ankle.004", 0.3f},
        {"ankle.005", 0.3f}, {"ankle.006", 0.3f}, {"ankle.007", 0.3f},
        {"hip", 0.3f},       {"hip.001", 0.3f},   {"hip.002", 0.3f},
        {"hip.003", 0.3f},   {"hip.004", 0.3f},   {"hip.005", 0.3f},
        {"hip.006", 0.3f},   {"hip.007", 0.3f},   {"eye_l", 0.1f},
        {"eye_r", 0.1f},     {"root", 1.0f},      {"head_b", 1.0f},
        {"head_tf", 1.0f},   {"head_tb", 1.0f}};

    std::map<std::string, int> sphere_ids;
    int i = 0;
    for (auto pair : sphere_vals) {
        sphere_ids[pair.first] = i++;
        spheres.push_back(pair.second);
        masses.push_back(sphere_masses[pair.first]);
    }

    using tuple = std::tuple<Vec2i, bool, float>;
    std::vector<tuple> pill_flags = {
        tuple(Vec2i(sphere_ids["root"], sphere_ids["head_b"]), true, 0.0f),
        tuple(Vec2i(sphere_ids["head_b"], sphere_ids["eye_l"]), true, 0.0f),
        tuple(Vec2i(sphere_ids["head_b"], sphere_ids["eye_r"]), true, 0.0f),
        tuple(Vec2i(sphere_ids["head_b"], sphere_ids["head_tf"]), true, 0.0f),
        tuple(Vec2i(sphere_ids["head_b"], sphere_ids["head_tb"]), true, 0.0f),
        tuple(Vec2i(sphere_ids["head_tf"], sphere_ids["head_tb"]), true, 0.0f),
        tuple(Vec2i(sphere_ids["root"], sphere_ids["hip.001"]), true, 1e-4f),
        tuple(Vec2i(sphere_ids["hip"], sphere_ids["knee.001"]), true, 1e-3f),
        tuple(Vec2i(sphere_ids["hip.001"], sphere_ids["knee.002"]), true,
              1e-3f),
        tuple(Vec2i(sphere_ids["hip.002"], sphere_ids["knee"]), true, 1e-3f),
        tuple(Vec2i(sphere_ids["hip.003"], sphere_ids["knee.003"]), true,
              1e-3f),
        tuple(Vec2i(sphere_ids["hip.004"], sphere_ids["knee.004"]), true,
              1e-3f),
        tuple(Vec2i(sphere_ids["hip.005"], sphere_ids["knee.005"]), true,
              1e-3f),
        tuple(Vec2i(sphere_ids["hip.006"], sphere_ids["knee.006"]), true,
              1e-3f),
        tuple(Vec2i(sphere_ids["hip.007"], sphere_ids["knee.007"]), true,
              1e-3f),
        tuple(Vec2i(sphere_ids["knee"], sphere_ids["ankle"]), true, 9e-3f),
        tuple(Vec2i(sphere_ids["knee.001"], sphere_ids["ankle.001"]), true,
              9e-3f),
        tuple(Vec2i(sphere_ids["knee.002"], sphere_ids["ankle.002"]), true,
              9e-3f),
        tuple(Vec2i(sphere_ids["knee.003"], sphere_ids["ankle.003"]), true,
              9e-3f),
        tuple(Vec2i(sphere_ids["knee.004"], sphere_ids["ankle.004"]), true,
              9e-3f),
        tuple(Vec2i(sphere_ids["knee.005"], sphere_ids["ankle.005"]), true,
              9e-3f),
        tuple(Vec2i(sphere_ids["knee.006"], sphere_ids["ankle.006"]), true,
              9e-3f),
        tuple(Vec2i(sphere_ids["knee.007"], sphere_ids["ankle.007"]), true,
              9e-3f),
        tuple(Vec2i(sphere_ids["ankle"], sphere_ids["foot"]), true, 9e-3f),
        tuple(Vec2i(sphere_ids["ankle.001"], sphere_ids["foot.001"]), true,
              9e-3f),
        tuple(Vec2i(sphere_ids["ankle.002"], sphere_ids["foot.002"]), true,
              9e-3f),
        tuple(Vec2i(sphere_ids["ankle.003"], sphere_ids["foot.003"]), true,
              9e-3f),
        tuple(Vec2i(sphere_ids["ankle.004"], sphere_ids["foot.004"]), true,
              9e-3f),
        tuple(Vec2i(sphere_ids["ankle.005"], sphere_ids["foot.005"]), true,
              9e-3f),
        tuple(Vec2i(sphere_ids["ankle.006"], sphere_ids["foot.006"]), true,
              9e-3f),
        tuple(Vec2i(sphere_ids["ankle.007"], sphere_ids["foot.007"]), true,
              9e-3f),
        tuple(Vec2i(sphere_ids["root"], sphere_ids["hip.002"]), true, 1e-4f),
        tuple(Vec2i(sphere_ids["root"], sphere_ids["hip.003"]), true, 1e-4f),
        tuple(Vec2i(sphere_ids["root"], sphere_ids["hip.004"]), true, 1e-4f),
        tuple(Vec2i(sphere_ids["root"], sphere_ids["hip.005"]), true, 1e-4f),
        tuple(Vec2i(sphere_ids["root"], sphere_ids["hip.006"]), true, 1e-4f),
        tuple(Vec2i(sphere_ids["root"], sphere_ids["hip.007"]), true, 1e-4f),
        tuple(Vec2i(sphere_ids["root"], sphere_ids["hip"]), true, 1e-4f),
    };

    for (int i = 0; i < pill_flags.size(); ++i) {
        pills.push_back(std::get<0>(pill_flags[i]));
        compliances.push_back(std::get<2>(pill_flags[i]));
        if (std::get<1>(pill_flags[i])) {
            control_pills.push_back(i);
        }
    }
}

} // namespace cow
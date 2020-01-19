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

#include "SphereWeights.h"

#include <fstream>
#include <iostream>
#include <thread>

#include "Viper_json.h"

using namespace OpenGP;
using Vec2i = Eigen::Vector2i;
using Vec3i = Eigen::Vector3i;
using Vec4i = Eigen::Vector4i;

int main(int argc, char **argv) {

    std::clog << "Starting SphereWeights:" << std::endl;

    std::istream *stream_ptr = &std::cin;

    std::ifstream file_stream;
    if (argc > 1) {
        file_stream = std::ifstream(argv[1]);
        stream_ptr = &file_stream;
    }

    std::istream &in_stream = *stream_ptr;

    std::string input, line;

    while (std::getline(in_stream, line)) {
        input += line + "\n";
    }

    rapidjson::Document j;
    j.Parse(input.c_str());

    auto verts = viper::from_json<std::vector<Vec3>>(j["vertices"]);
    auto tris = viper::from_json<std::vector<Vec3i>>(j["triangles"]);
    auto spheres = viper::from_json<std::vector<Vec4>>(j["spheres"]);
    auto pills = viper::from_json<std::vector<Vec2i>>(j["pills"]);

    SurfaceMesh mesh;
    SphereMesh smesh;

    for (auto vert : verts) {
        mesh.add_vertex(vert);
    }
    for (auto tri : tris) {
        using V = SurfaceMesh::Vertex;
        mesh.add_triangle(V(tri[0]), V(tri[1]), V(tri[2]));
    }

    for (auto sphere : spheres) {
        smesh.add_vertex(sphere);
    }
    for (auto pill : pills) {
        using V = SphereMesh::Vertex;
        smesh.add_edge(V(pill[0]), V(pill[1]));
    }

    calc_weights(smesh, mesh);

    auto &weights =
        mesh.get_vertex_property<std::vector<float>>("v:skinweight").vector();
    auto &bone_ids =
        mesh.get_vertex_property<std::vector<int>>("v:boneid").vector();

    {
        rapidjson::MemoryPoolAllocator<> alloc;
        rapidjson::Document j(&alloc);
        j.SetObject();
        rapidjson::Document::AllocatorType &allocator = j.GetAllocator();

        rapidjson::Value weights_array(rapidjson::kArrayType);
        for (auto &w : weights) {
            weights_array.PushBack(viper::to_json(w, allocator), allocator);
        }
        j.AddMember("weights", weights_array, allocator);

        rapidjson::Value ids_array(rapidjson::kArrayType);
        for (auto &i : bone_ids) {
            ids_array.PushBack(viper::to_json(i, allocator), allocator);
        }
        j.AddMember("bone_ids", ids_array, allocator);

        rapidjson::StringBuffer strbuf;
        strbuf.Clear();
        rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
        j.Accept(writer);

        std::cout << strbuf.GetString() << std::endl;
    }

    return 0;
}
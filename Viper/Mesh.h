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

class Mesh {
  public:
    Vec3Array vertices;
    std::vector<Vec3i> triangles;
    Vec3Array normals; // face normals

    Vec3Array vertex_normals;
    Vec3Array edge_normals;
    Vec2iArray edges;

    IntArray faceTags;

    void boundingBox(Vec3 &min, Vec3 &max) const;
    void computeEdges();
    void computeVertexNormals();
    void computeEdgeNormals();
    Vec3 getSmoothNormal(int tIdx, const Vec3 &p);
    Mesh applyTransform(const Transform &T) const;
    void addQuad(int a, int b, int c, int d);
    static Mesh loadFromOBJ(std::string path);
    static Mesh fromPill(const Vec3 &a, const Vec3 &b, float ra, float rb,
                         int n = 300, int m = 100);
    void exportToOBJ(const std::string &path);

  private:
    void addEdgeUnique(const Vec2i &e, int tIdx, int k);
    int getEdgeIndex(int tIdx, int i);
    IntArray triangleToEdge;
};

} // namespace viper
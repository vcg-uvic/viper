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

#include "Mesh.h"
#include <algorithm>
#include <set>
#define TINYOBJLOADER_IMPLEMENTATION
#include "Utils.h"
#include "tiny_obj_loader.h"

#include <fstream>
#include <iostream>

namespace viper {

bool vec2iEquals(Vec2i a, Vec2i b) {
    return (a(0) == b(0) && a(1) == b(1)) || (a(0) == b(1) && a(1) == b(0));
}

struct less_vec2i {
    inline bool operator()(const Vec2i &a, const Vec2i &b) {
        int amax = std::max(a(0), a(1));
        int bmax = std::max(b(0), b(1));
        if (amax < bmax)
            return true;
        if (amax > bmax)
            return false;
        int amin = std::min(a(0), a(1));
        int bmin = std::min(b(0), b(1));
        return amin < bmin;
    }
};

Mesh Mesh::loadFromOBJ(std::string path) {
    Mesh m;
    tinyobj::attrib_t attrib;
    std::string err;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> mat;
    bool r = tinyobj::LoadObj(&attrib, &shapes, &mat, &err, path.c_str());
    tinyobj::shape_t &s = shapes[0];
    const int nVerts = attrib.vertices.size() / 3;

    for (int i = 0; i < nVerts; i++) {
        float x = attrib.vertices[i * 3 + 0];
        float y = attrib.vertices[i * 3 + 1];
        float z = attrib.vertices[i * 3 + 2];
        m.vertices.push_back(Vec3(x, y, z));
    }

    size_t index_offset = 0;
    for (size_t f = 0; f < s.mesh.num_face_vertices.size(); f++) {
        int fv = s.mesh.num_face_vertices[f];
        if (fv != 3)
            std::cerr << "ERROR: OBJ FILE CONTAINS FACES OF " << fv
                      << " VERTICES" << std::endl;

        int a = s.mesh.indices[index_offset + 0].vertex_index;
        int b = s.mesh.indices[index_offset + 1].vertex_index;
        int c = s.mesh.indices[index_offset + 2].vertex_index;
        m.triangles.push_back(Vec3i(a, b, c));

        index_offset += fv;

        if (attrib.normals.size() > 0) {
            int nIdx = s.mesh.indices[f * 3].normal_index;
            float nx = attrib.normals[nIdx * 3 + 0];
            float ny = attrib.normals[nIdx * 3 + 1];
            float nz = attrib.normals[nIdx * 3 + 2];
            m.normals.push_back(Vec3(nx, ny, nz).normalized());
        }
    }

    std::cout << "vertices: " << m.vertices.size()
              << " faces: " << m.triangles.size()
              << " normals: " << m.normals.size() << std::endl;

    return m;
}

void Mesh::boundingBox(Vec3 &min, Vec3 &max) const {
    if (vertices.size() == 0) {
        min = Vec3::Zero();
        max = Vec3::Zero();
        return;
    }

    min = vertices[0];
    max = vertices[0];

    for (int i = 0; i < vertices.size(); i++) {
        min = min.cwiseMin(vertices[i]);
        max = max.cwiseMax(vertices[i]);
    }
}

Mesh Mesh::applyTransform(const Transform &T) const {
    Mesh m = *this;
    // m.triangles = triangles;
    for (int i = 0; i < vertices.size(); i++)
        m.vertices[i] = T * vertices[i];
    for (int i = 0; i < normals.size(); i++)
        m.normals[i] = T.rotation() * normals[i];

    return m;
}

void Mesh::addEdgeUnique(const Vec2i &a, int tIdx, int k) {
    int foundIdx = -1;
    for (int i = 0; i < edges.size(); i++) {
        Vec2i &b = edges[i];
        if ((a(0) == b(0) && a(1) == b(1)) || (a(0) == b(1) && a(1) == b(0))) {
            foundIdx = i;
            break;
        }
    }

    if (foundIdx == -1) {
        edges.push_back(a);
        foundIdx = edges.size() - 1;
    }
    triangleToEdge[3 * tIdx + k] = foundIdx;
}

void Mesh::computeEdges() {
    triangleToEdge.resize(3 * triangles.size());
    for (int i = 0; i < triangles.size(); i++) {
        int a = triangles[i](0);
        int b = triangles[i](1);
        int c = triangles[i](2);
        addEdgeUnique(Vec2i(a, b), i, 0);
        addEdgeUnique(Vec2i(b, c), i, 1);
        addEdgeUnique(Vec2i(c, a), i, 2);
    }
}

void Mesh::computeVertexNormals() {
    vertex_normals.resize(vertices.size());
    for (int i = 0; i < vertex_normals.size(); i++)
        vertex_normals[i] = Vec3::Zero();

    for (int i = 0; i < triangles.size(); i++) {
        Vec3 ab = vertices[triangles[i][1]] - vertices[triangles[i][0]];
        Vec3 ac = vertices[triangles[i][2]] - vertices[triangles[i][0]];
        Vec3 n = ab.cross(ac).normalized();
        for (int k = 0; k < 3; k++)
            vertex_normals[triangles[i][k]] += n;
    }

    for (int i = 0; i < vertex_normals.size(); i++)
        vertex_normals[i].normalize();
}

void Mesh::computeEdgeNormals() {
    if (edges.size() == 0)
        computeEdges();

    edge_normals.resize(edges.size());
    for (int i = 0; i < edges.size(); i++)
        edge_normals[i] = Vec3::Zero();

    for (int i = 0; i < normals.size(); i++) {
        for (int k = 0; k < 3; k++) {
            int eIdx = getEdgeIndex(i, k);
            edge_normals[eIdx] += normals[i];
        }
    }

    for (int i = 0; i < edge_normals.size(); i++)
        edge_normals[i].normalize();
}

Vec3 Mesh::getSmoothNormal(int tIdx, const Vec3 &p) {
    if (vertex_normals.size() == 0)
        computeVertexNormals();
    if (edge_normals.size() == 0)
        computeEdgeNormals();

    int a = triangles[tIdx][0];
    int b = triangles[tIdx][1];
    int c = triangles[tIdx][2];
    Vec3 pa = vertices[a];
    Vec3 pb = vertices[b];
    Vec3 pc = vertices[c];

    // 0. Compute the uv coordinates
    Vec3 uvw = Utils::barycentricCoordinates(pa, pb, pc, p);

    float eps = 1e-6f;
    int bCount = 0;
    for (int i = 0; i < 3; i++)
        if (uvw[i] <= eps)
            bCount++;

    // if not on the boundary, return the triangle normal
    if (bCount == 0)
        return normals[tIdx];

    // if on the edge, return the edge normal
    if (bCount == 1) {
        if (uvw[0] <= eps)
            return edge_normals[getEdgeIndex(tIdx, 1)];
        else if (uvw[1] <= eps)
            return edge_normals[getEdgeIndex(tIdx, 2)];
        return edge_normals[getEdgeIndex(tIdx, 0)];
    }

    // if on a vertex, return the vertex normal
    if (uvw[0] >= 1.f - eps)
        return vertex_normals[a];
    if (uvw[1] >= 1.f - eps)
        return vertex_normals[b];
    return vertex_normals[c];
}

int Mesh::getEdgeIndex(int tIdx, int i) { return triangleToEdge[tIdx * 3 + i]; }

void Mesh::addQuad(int a, int b, int c, int d) {
    triangles.push_back(Vec3i(a, b, c));
    triangles.push_back(Vec3i(a, c, d));
}

Mesh Mesh::fromPill(const Vec3 &a, const Vec3 &b, float ra, float rb, int n,
                    int m) {
    Mesh mesh;

    Vec3 ab = b - a;
    Vec3 abn = ab.normalized();
    float L = ab.norm();
    float e = (rb - ra) / L;
    float beta = std::asin(e);

    float ha = ra * std::cos(beta);
    float hb = rb * std::cos(beta);
    float ga = -ra * e;
    float gb = -rb * e;
    Vec3 ca = a + ga * abn;
    Vec3 cb = b + gb * abn;
    Transform T = Utils::getOrthogonalFrame(abn);

    // cylinder
    for (int i = 0; i < n; i++) {
        float t = 2.0f * M_PIf * i / (float)n;
        Vec3 v = T * Vec3(0.0f, cos(t), sin(t));
        Vec3 v1 = ca + v * ha;
        Vec3 v2 = cb + v * hb;

        mesh.vertices.push_back(v1);
        mesh.vertices.push_back(v2);

        mesh.normals.push_back((v1 - a).normalized());
        mesh.normals.push_back((v2 - b).normalized());

        if (i == 0)
            mesh.addQuad(0, 1, n * 2 - 1, n * 2 - 2);
        else
            mesh.addQuad(i * 2, i * 2 + 1, i * 2 - 1, i * 2 - 2);
    }

    // caps
    int ob = 2 * n;
    float capAngleB = 0.5f * M_PIf + beta;
    int mn = m * n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            float alpha = 2.0f * M_PIf * i / (float)n;
            float beta = capAngleB * j / (float)(m - 1);

            Vec3 v = T * Vec3(cos(beta), cos(alpha) * sin(beta),
                              sin(alpha) * sin(beta));
            mesh.vertices.push_back(b + rb * v);
            mesh.normals.push_back(v);

            if (j < m - 1) {
                int idx = i * m + j;
                mesh.addQuad(ob + idx, ob + (idx + m) % mn,
                             ob + (idx + m + 1) % mn, ob + idx + 1);
            }
        }
    }

    int oa = 2 * n + m * n;
    float capAngleA = 0.5f * M_PIf - beta;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            float alpha = 2.0f * M_PIf * i / (float)n;
            float beta = capAngleA * j / (float)(m - 1);

            Vec3 v = T * Vec3(-cos(beta), cos(alpha) * sin(beta),
                              sin(alpha) * sin(beta));
            mesh.vertices.push_back(a + ra * v);
            mesh.normals.push_back(v);

            if (j < m - 1) {
                int idx = i * m + j;
                mesh.addQuad(oa + idx, oa + (idx + m) % mn,
                             oa + (idx + m + 1) % mn, oa + idx + 1);
            }
        }
    }

    return mesh;
}

void Mesh::exportToOBJ(const std::string &path) {
    std::ofstream file;
    file.open(path);

    for (int i = 0; i < vertices.size(); i++)
        file << "v " << vertices[i][0] << " " << vertices[i][1] << " "
             << vertices[i][2] << std::endl;
    for (int i = 0; i < triangles.size(); i++)
        file << "f " << triangles[i][0] + 1 << " " << triangles[i][1] + 1 << " "
             << triangles[i][2] + 1 << std::endl;
    file.close();
}

} // namespace viper
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

#include "Utils.h"
#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/make_surface_mesh.h>
#include <iostream>
#include <map>
#include <random>

namespace viper {

typedef CGAL::Surface_mesh_default_triangulation_3 Tr;
typedef CGAL::Complex_2_in_triangulation_3<Tr> C2t3;
typedef Tr::Geom_traits GT;
typedef GT::Sphere_3 Sphere_3;
typedef GT::Point_3 Point_3;
typedef GT::FT FT;

template <typename function>
using Surface_3 = CGAL::Implicit_surface_3<GT, function>;

struct SceneSDF {
    SceneSDF(const Scene *scene, float offset) : scene(scene), offset(offset) {}
    FT operator()(Point_3 p3) const {
        Vec3 p = Vec3(p3.x(), p3.y(), p3.z());
        float minDist = 1e10f;
        for (int i = 0; i < scene->state.x.size(); i++) {
            float d = (p - scene->state.x[i]).norm() - scene->state.r[i];
            minDist = std::min(minDist, d);
        }
        for (const Vec2i &f : scene->pills) {

            Vec3 c;
            float r;
            float t;
            float d = Utils::closestPtPointPill(
                p, scene->state.x[f(0)], scene->state.x[f(1)],
                scene->state.r[f(0)], scene->state.r[f(1)], t, c, r);
            minDist = std::min(minDist, d - r);
        }

        return minDist - offset;
    }

    float offset;
    const Scene *scene;
};

namespace Utils {
Quaternion randomQuaternion() {
    return Quaternion(
               Rotation(randomFloat(0.0f, 2.0 * M_PIf), randomDirection()))
        .normalized();
}

Vec3 safeNormal(const Vec3 &a, const Vec3 &b) {
    Vec3 n = b - a;
    float d = n.norm();
    const float eps = 1e-8f;
    if (d < eps)
        n = Utils::randomDirection();
    else
        n = n / d;

    return n;
}

void generateMesh(const Scene *scene, Mesh &mesh, float offset) {
    mesh.vertices.clear();
    mesh.triangles.clear();

    Tr tr3d;
    C2t3 c2t3(tr3d);
    SceneSDF sdf = SceneSDF(scene, offset);
    Surface_3<SceneSDF> surface(sdf, Sphere_3(Point_3(0.0, 1.0, 0.0), 4.));
    CGAL::Surface_mesh_default_criteria_3<Tr> criteria(20.f, 0.03f, 0.03f);
    CGAL::make_surface_mesh(c2t3, surface, criteria, CGAL::Manifold_tag());

    Tr tr = c2t3.triangulation();
    std::map<Tr::Vertex_handle, int> V;
    int i = 0;
    for (auto it = tr.finite_vertices_begin(); it != tr.finite_vertices_end();
         it++) {
        Tr::Vertex_handle vh = it;
        V[vh] = i++;
        Point_3 p = it->point();
        mesh.vertices.push_back(Vec3(p.x(), p.y(), p.z()));
    }

    for (auto it = tr.finite_facets_begin(); it != tr.finite_facets_end();
         it++) {
        Tr::Cell_handle cell = it->first;
        int index = it->second;
        if (!c2t3.is_in_complex(cell, index))
            continue;
        Vec3i t = Vec3i(-1, -1, -1);
        for (int i = 0; i < 3; i++) {
            Tr::Vertex_handle vh =
                cell->vertex(tr.vertex_triple_index(index, i));
            auto vIt = V.find(vh);
            if (vIt == V.end())
                std::cout << "WOOOOOOOOOOO" << std::endl;
            else
                t[i] = vIt->second;
        }
        mesh.triangles.push_back(t);
    }

    for (int i = 0; i < mesh.triangles.size(); i++) {
        Vec3i &T = mesh.triangles[i];
        Vec3 ab = mesh.vertices[T[1]] - mesh.vertices[T[0]];
        Vec3 ac = mesh.vertices[T[2]] - mesh.vertices[T[0]];
        Vec3 n = ab.cross(ac).normalized();
        Vec3 p0 =
            (mesh.vertices[T[0]] + mesh.vertices[T[0]] + mesh.vertices[T[0]]) /
            3.0f;
        Vec3 p1 = p0 + 0.00001f * n;
        float f0 = sdf(Point_3(p0[0], p0[1], p0[2]));
        float f1 = sdf(Point_3(p1[0], p1[1], p1[2]));
        if (f0 > f1) {
            int tmp = T[1];
            T[1] = T[2];
            T[2] = tmp;
        }
    }

    std::cout << "nverts: " << mesh.vertices.size() << std::endl;
    std::cout << "nFaces: " << mesh.triangles.size() << std::endl;
}

Transform getOrthogonalFrame(const Vec3 &a) {
    Vec3 helper = Vec3(1.f, 0.f, 0.f);
    if (abs(a.dot(helper)) > 0.9f)
        helper = Vec3(0.f, 1.f, 0.f);
    Vec3 u = a.cross(helper).normalized();
    Vec3 v = a.cross(u);

    Matrix3 M;
    M.col(0) = a;
    M.col(1) = u;
    M.col(2) = v;

    return Transform(M);
}

void closestPtPointSegment(Vec3 c, Vec3 a, Vec3 b, float &t, Vec3 &d) {
    Vec3 ab = b - a;
    t = (c - a).dot(ab);
    if (t < 0.f) {
        t = 0.f;
        d = a;
    } else {
        float denom = ab.dot(ab);
        if (t >= denom) {
            t = 1.0f;
            d = b;
        } else {
            t = t / denom;
            d = a + t * ab;
        }
    }
}

float randomFloat(float min, float max) {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<float> dist(min, max);
    float r = dist(e2);
    return r;
}

Vec3 randomVector(const Vec3 &min, const Vec3 &max) {
    float x = randomFloat(min.x(), max.x());
    float y = randomFloat(min.y(), max.y());
    float z = randomFloat(min.z(), max.z());
    return Vec3(x, y, z);
}

float clamp(float n, float min, float max) {
    if (n < min)
        return min;
    if (n > max)
        return max;
    return n;
}

float closestPtSegmentSegment(Vec p1, Vec q1, Vec p2, Vec q2, float &s,
                              float &t, Vec &c1, Vec &c2) {
    const float eps = 1e-10f;
    Vec d1 = q1 - p1;
    Vec d2 = q2 - p2;
    Vec r = p1 - p2;
    float a = d1.squaredNorm();
    float e = d2.squaredNorm();
    float f = d2.dot(r);

    if (a <= eps && e <= eps) {
        s = t = 0.0f;
        c1 = p1;
        c2 = p2;
        return (c1 - c2).squaredNorm();
    }
    if (a <= eps) {
        s = 0.0f;
        t = f / e;
        t = clamp(t, 0.0f, 1.0f);
    } else {
        float c = d1.dot(r);
        if (e <= eps) {
            t = 0.0f;
            s = clamp(-c / a, 0.0f, 1.0f);
        } else {
            float b = d1.dot(d2);
            float denom = a * e - b * b;

            if (denom != 0.0f)
                s = clamp((b * f - c * e) / denom, 0.0f, 1.0f);
            else
                s = 0.5f;

            t = (b * s + f) / e;

            if (t < 0.0f) {
                t = 0.0f;
                s = clamp(-c / a, 0.0f, 1.0f);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = clamp((b - c) / a, 0.0f, 1.0f);
            }
        }
    }

    c1 = p1 + d1 * s;
    c2 = p2 + d2 * t;
    return (c1 - c2).norm();
}

IntArray range(int start, int end) {
    IntArray ids;
    ids.reserve(end - start + 1);
    for (int i = start; i <= end; i++)
        ids.push_back(i);
    return ids;
}

Vec3 randomDirection() {
    // https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d
    float t = randomFloat(0.0f, 2.0f * M_PIf);
    float z = randomFloat(-1.0f, 1.0f);
    float a = sqrt(1.f - z * z);
    return Vec(a * cos(t), a * sin(t), z);
}

Vec3 randomDirectionCircle(const Vec3 &n) {
    float t = randomFloat(0.f, 2 * M_PIf);
    Transform T = getOrthogonalFrame(n);
    return T * Vec3(0.f, cos(t), sin(t));
}

Vec3 randomDirectionHalfSphere(const Vec3 &n) {
    Vec3 v = randomDirection();
    float p = v.dot(n);
    if (p < 0.f)
        v -= 2.f * p * n;
    return v;
}

float closestPtPointPill(const Vec3 &p, const Vec3 &a, const Vec3 &b, float ra,
                         float rb, float &t, Vec &c, float &r) {
    Vec3 ab = b - a;
    float L = ab.norm();
    Vec3 abn = ab / L;
    Vec3 ap = p - a;
    Vec3 p_proj = ap - abn * ap.dot(abn);
    float sigma = tan(asin((rb - ra) / L));
    if (std::isnan(sigma)) {
        return std::min(ap.norm() - ra, (p - b).norm() - rb);
    }
    Vec3 offset = abn * p_proj.norm() * sigma;
    t = std::min(1.f, std::max(0.f, abn.dot(ap + offset) / L));
    c = (1.f - t) * a + t * b;
    r = (1.f - t) * ra + t * rb;
    return (c - p).norm() - r;
}

float coneSphereDistance(const Vec3 &a1, const Vec3 &a2, const Vec3 &b1,
                         const Vec3 &b2, float ra1, float ra2, float rb1,
                         float rb2, Vec2 uv) {
    Vec3 a = a1 + uv(0) * (a2 - a1);
    Vec3 b = b1 + uv(1) * (b2 - b1);
    float ra = ra1 + uv(0) * (ra2 - ra1);
    float rb = rb1 + uv(1) * (rb2 - rb1);
    return (a - b).norm() - ra - rb;
}

float pillDistanceU(const Vec3Array &x, const FloatArray &r, const Vec2i &a,
                    const Vec2i &b, float u, float &v, Vec3 &pa, Vec3 &pb) {
    pa = (1.f - u) * x[a(0)] + u * x[a(1)];
    float ra = (1.f - u) * r[a(0)] + u * r[a(1)];
    float rc;
    return closestPtPointPill(pa, x[b(0)], x[b(1)], r[b(0)], r[b(1)], v, pb,
                              rc) -
           ra;
}

float closestPtPills(const Vec3Array &x, const FloatArray &r, const Vec2i &a,
                     const Vec2i &b, Vec2 &uv, Vec &pa, Vec &pb) {
    const int iter = 14;
    Vec2 range = Vec2(0.f, 1.f);
    for (int j = 0; j < iter; j++) {
        float eps = M_PIf * 1e-7f;
        float mid = 0.5f * range.sum();
        float ua = mid - eps;
        float ub = mid + eps;
        if (ua > ub)
            break;
        float v;
        Vec3 pa, pb;
        float fa = pillDistanceU(x, r, a, b, ua, v, pa, pb);
        float fb = pillDistanceU(x, r, a, b, ub, v, pa, pb);
        if (fa == fb)
            range = Vec2(ua, ub);
        else if (fa > fb)
            range(0) = ua;
        else
            range(1) = ub;
    }
    uv(0) = 0.5f * range.sum();
    return pillDistanceU(x, r, a, b, uv(0), uv(1), pa, pb);
}

float closestPtPillsNumerical(const Vec3Array &x, const FloatArray &r,
                              const Vec2i &a, const Vec2i &b, Vec2 &uv, Vec &pa,
                              Vec &pb) {
    int n = 1000;
    float min_dist = 1e20f;
    for (int i = 0; i < n; i++) {
        float u = ((float)i) / (n - 1);
        Vec3 pa = (1.f - u) * x[a(0)] + u * x[a(1)];
        float ra = (1.f - u) * r[a(0)] + u * r[a(1)];
        for (int j = 0; j < n; j++) {
            float v = ((float)j) / (n - 1);
            Vec3 pb = (1.f - v) * x[b(0)] + v * x[b(1)];
            float rb = (1.f - v) * r[b(0)] + v * r[b(1)];

            float d = (pa - pb).norm() - ra - rb;
            if (d < min_dist) {
                min_dist = d;
                uv = Vec2(u, v);
            }
        }
    }

    pa = (1.f - uv[0]) * x[a(0)] + uv[0] * x[a(1)];
    pb = (1.f - uv[1]) * x[b(0)] + uv[1] * x[b(1)];
    return min_dist;
}

float pillDistanceNumerical(const Vec3 &p, const Vec3 &a, const Vec3 &b,
                            float rA, float rB, int n) {
    float min_dist = 1e20f;
    for (int i = 0; i < n; i++) {
        float t = ((float)i) / (n - 1);
        Vec3 c = (1.f - t) * a + t * b;
        float rc = (1.f - t) * rA + t * rB;
        float d = (c - p).norm() - rc;
        min_dist = std::min(d, min_dist);
    }

    return min_dist;
}

float pillVolumeNumerical(const Vec3 &a, const Vec3 &b, float rA, float rB,
                          int n) {
    Vec3 gMin = (a - Vec3(rA, rA, rA)).cwiseMin(b - Vec3(rB, rB, rB));
    Vec3 gMax = (a + Vec3(rA, rA, rA)).cwiseMax(b + Vec3(rB, rB, rB));
    Vec3 gDiff = gMax - gMin;
    float maxDiff = gDiff.maxCoeff();
    float s = maxDiff / (float)n;
    Vec3i gDims = (gDiff / s).array().ceil().cast<int>();

    Vec3 gSize = gDims.cast<float>() * s;
    float Vgrid = gSize.prod();
    int cell_count = gDims.prod();

    int count = 0;
    for (int i = 0; i < gDims[0]; i++) {
        for (int j = 0; j < gDims[1]; j++) {
            for (int k = 0; k < gDims[2]; k++) {
                Vec3 p = gMin + s * Vec3(i, j, k);
                float t, r;
                Vec3 c;
                // float d = closestPtPointPill(p, a, b, rA, rB, t, c, r);
                float d = pillDistanceNumerical(p, a, b, rA, rB, 100);
                // closestPtPointSegment(p, a, b, t, c);
                bool isInside = d < 0.0f;
                if (isInside)
                    count++;
            }
        }
    }
    float r = (float)count / (float)cell_count;
    return Vgrid * r;
}

float pillVolumeAnalytical(const Vec3 &a, const Vec3 &b, float rA, float rB) {
    float d = (a - b).norm();
    float beta = asin((rB - rA) / d);
    float sinBeta = sin(beta);
    float cosBeta = cos(beta);
    float L = d + (rA - rB) * sinBeta;
    float ha = rA * cosBeta;
    float hb = rB * cosBeta;
    float da = rA * (1.f - sinBeta);
    float db = rB * (1.f + sinBeta);

    float v_cyl = M_PIf / 3.f * L * (ha * ha + ha * hb + hb * hb);
    float v_capA = M_PIf / 3.f * da * da * (3.f * rA - da);
    float v_capB = M_PIf / 3.f * db * db * (3.f * rB - db);
    float v = v_cyl + v_capA + v_capB;
    return v;
}

Vec3 pointInterp(const Vec3 &a, const Vec3 &b, float t) {
    return (1.f - t) * a + t * b;
}

Vec3 safeNormalCapsules(const Vec3 &a, const Vec3 &b, const Vec3 &a1,
                        const Vec3 &a2, const Vec3 &b1, const Vec3 &b2) {
    Vec3 n = b - a;
    float d = n.norm();
    const float eps = 1e-8f;
    if (d < eps)
        n = (a2 - a1).cross(b2 - b1).normalized();
    if (n.norm() < eps)
        n = Utils::randomDirection();
    else
        n = n / d;

    return n;
}

float triArea2D(float x1, float y1, float x2, float y2, float x3, float y3) {
    return (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2);
}

Vec3 barycentricCoordinates(const Vec3 &a, const Vec3 &b, const Vec3 &c,
                            const Vec3 &p) {
    Vec3 m = (b - a).cross(c - a);
    float nu, nv, ood;
    float x = std::abs(m.x());
    float y = std::abs(m.y());
    float z = std::abs(m.z());

    if (x >= y && x >= z) {
        nu = triArea2D(p.y(), p.z(), b.y(), b.z(), c.y(), c.z());
        nv = triArea2D(p.y(), p.z(), c.y(), c.z(), a.y(), a.z());
        ood = 1.f / m.x();
    } else if (y >= x && y >= z) {
        nu = triArea2D(p.x(), p.z(), b.x(), b.z(), c.x(), c.z());
        nv = triArea2D(p.x(), p.z(), c.x(), c.z(), a.x(), a.z());
        ood = 1.f / -m.y();
    } else {
        nu = triArea2D(p.x(), p.y(), b.x(), b.y(), c.x(), c.y());
        nv = triArea2D(p.x(), p.y(), c.x(), c.y(), a.x(), a.y());
        ood = 1.f / m.z();
    }

    float u = nu * ood;
    float v = nv * ood;
    float w = 1.f - u - v;
    return Vec3(u, v, w);
}

float pillVolume(const Vec3 &xa, const Vec3 &xb, float ra, float rb, bool capA,
                 bool capB) {
    float V = 0.0f;
    float d = (xa - xb).norm();
    if (d > 1e-7f) {
        float e = (rb - ra) / d;
        float L = d + (ra - rb) * e;
        V += M_PIf / 3.0f *
             ((ra * ra * ra - rb * rb * rb) * (e * e * e - 3.f * e) +
              L * (1.f - e * e) * (ra * ra + ra * rb + rb * rb));
    }
    if (capA)
        V += 2.0f * M_PIf / 3.0f * ra * ra * ra;
    if (capB)
        V += 2.0f * M_PIf / 3.0f * rb * rb * rb;

    return V;
}
} // namespace Utils

} // namespace viper
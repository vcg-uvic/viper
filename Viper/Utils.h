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
#include "Scene.h"

namespace viper {

namespace Utils {
Transform getOrthogonalFrame(const Vec3 &a);
float randomFloat(float min = 0.0f, float max = 1.0f);
Vec3 randomVector(const Vec3 &min, const Vec3 &max);
Quaternion randomQuaternion();
Vec3 randomDirection();
Vec3 randomDirectionCircle(const Vec3 &n);
Vec3 randomDirectionHalfSphere(const Vec3 &n);
IntArray range(int start, int end);
void closestPtPointSegment(Vec3 c, Vec3 a, Vec3 b, float &t, Vec3 &d);
float closestPtSegmentSegment(Vec p1, Vec q1, Vec p2, Vec q2, float &s,
                              float &t, Vec &c1, Vec &c2);
float closestPtPointPill(const Vec3 &p, const Vec3 &a, const Vec3 &b, float ra,
                         float rb, float &t, Vec &c, float &r);
float closestPtPills(const Vec3Array &x, const FloatArray &r, const Vec2i &a,
                     const Vec2i &b, Vec2 &uv, Vec &pa, Vec &pb);
float closestPtPillsNumerical(const Vec3Array &x, const FloatArray &r,
                              const Vec2i &a, const Vec2i &b, Vec2 &uv, Vec &pa,
                              Vec &pb);
void generateMesh(const Scene *scene, Mesh &mesh, float offset);
float pillVolumeNumerical(const Vec3 &a, const Vec3 &b, float rA, float rB,
                          int n);
float pillVolumeAnalytical(const Vec3 &a, const Vec3 &b, float rA, float rB);
Vec3 safeNormal(const Vec3 &a, const Vec3 &b);
Vec3 safeNormalCapsules(const Vec3 &a, const Vec3 &b, const Vec3 &a1,
                        const Vec3 &a2, const Vec3 &b1, const Vec3 &b2);
Vec3 pointInterp(const Vec3 &a, const Vec3 &b, float t);
Vec3 barycentricCoordinates(const Vec3 &a, const Vec3 &b, const Vec3 &c,
                            const Vec3 &p);
float pillVolume(const Vec3 &xa, const Vec3 &xb, float ra, float rb,
                 bool capA = false, bool capB = false);
} // namespace Utils

} // namespace viper
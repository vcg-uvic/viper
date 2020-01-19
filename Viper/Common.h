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

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

#define PARTICLE_RADIUS 0.05f
#define M_PI 3.14159265358979323846
#define M_PIf 3.14159265358979323846f

namespace viper {

typedef Eigen::Vector2f Vec2;
typedef Eigen::Vector3f Vec3;
typedef Vec3 Vec;
typedef Eigen::Vector4f Vec4;
typedef Eigen::Matrix<float, 5, 1> Vec5;
typedef Eigen::Matrix<float, 6, 1> Vec6;
typedef Eigen::Vector2i Vec2i;
typedef Eigen::Vector3i Vec3i;
typedef Eigen::Vector4i Vec4i;
typedef int Id;
typedef std::vector<Vec3> Vec3Array;
typedef std::vector<Vec3i> Vec3iArray;
typedef std::vector<Vec4> Vec4Array;
typedef std::vector<Vec4i> Vec4iArray;
typedef std::vector<float> FloatArray;
typedef std::vector<int> IntArray;
typedef std::vector<Id> IdArray;
typedef std::vector<Vec2i> Vec2iArray;
typedef Eigen::VectorXf VectorX;
typedef Eigen::Triplet<float> Triplet;
typedef Eigen::Matrix3f Matrix3;
typedef Eigen::Matrix3Xf Matrix3X;
typedef Eigen::SparseMatrix<float, Eigen::ColMajor> SparseMatrix;
typedef Eigen::Matrix4f Matrix4;
typedef Eigen::Affine3f Transform;
typedef Eigen::AngleAxisf Rotation;
typedef Eigen::Translation3f Translation;
typedef Eigen::Quaternionf Quaternion;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::MatrixXi Matrixi;
typedef std::vector<Quaternion> QuaternionArray;

struct Constraint;
typedef std::vector<Constraint *> ConstraintList;

class ProjectionData {
  public:
    virtual void set(Id id, const Vec3 &dx, float w = 1.f) = 0;
    virtual void setScale(Id id, float diff_s, float w = 1.f) = 0;
    virtual void setOrientation(Id id, const Quaternion &dq, float w = 1.f) = 0;
};

} // namespace viper
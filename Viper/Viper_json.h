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

#include <type_traits>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "Common.h"

namespace viper {

// Quaternion -> JSON
inline rapidjson::Value to_json(const Quaternion &q,
                                rapidjson::Document::AllocatorType &allocator) {
    rapidjson::Value o;
    o.SetArray();
    o.PushBack(q.w(), allocator);
    o.PushBack(q.x(), allocator);
    o.PushBack(q.y(), allocator);
    o.PushBack(q.z(), allocator);

    return o;
}

// Vector -> JSON
template <typename T>
rapidjson::Value to_json(const Eigen::MatrixBase<T> &v,
                         rapidjson::Document::AllocatorType &allocator) {
    static_assert(T::RowsAtCompileTime > 0 && T::ColsAtCompileTime == 1,
                  "JSON -> Matrix not implemented yet");
    rapidjson::Value o;
    o.SetArray();
    for (int i = 0; i < v.size(); ++i) {
        o.PushBack(v[i], allocator);
    }

    return o;
}

// Number -> JSON
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, rapidjson::Value>::type
to_json(T v, rapidjson::Document::AllocatorType &allocator) {
    rapidjson::Value o(v);
    return o;
}

// std::vector -> JSON
template <typename T>
typename std::enable_if<
    std::is_same<typename std::decay<T>::type,
                 std::vector<typename T::value_type>>::value,
    rapidjson::Value>::type
to_json(const T &v, rapidjson::Document::AllocatorType &allocator) {
    rapidjson::Value o;
    o.SetArray();
    for (int i = 0; i < v.size(); i++)
        o.PushBack(to_json(v[i], allocator), allocator);

    return o;
}

// String -> JSON
inline rapidjson::Value to_json(const std::string &s,
                                rapidjson::Document::AllocatorType &allocator) {
    return rapidjson::Value(s.c_str(), allocator);
}

// Transform -> JSON
inline rapidjson::Value to_json(const Transform &T,
                                rapidjson::Document::AllocatorType &allocator) {
    Matrix4 m = T.matrix();
    rapidjson::Value o;
    o.SetArray();
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            o.PushBack(m(i, j), allocator);

    return o;
}

// JSON -> Integer
template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
from_json(const rapidjson::Value &j) {
    return static_cast<T>(j.GetInt64());
}

// JSON -> Float
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
from_json(const rapidjson::Value &j) {
    return static_cast<T>(j.GetDouble());
}

// JSON -> Vector
template <typename T>
typename std::enable_if<std::is_base_of<Eigen::MatrixBase<T>, T>::value,
                        T>::type
from_json(const rapidjson::Value &j) {
    static_assert(T::RowsAtCompileTime > 0 && T::ColsAtCompileTime == 1,
                  "JSON -> Matrix not implemented yet");
    T result;
    for (int i = 0; i < result.size(); ++i) {
        result[i] = from_json<typename T::Scalar>(j[i]);
    }
    return result;
}

// JSON -> std::vector
template <typename T>
typename std::enable_if<
    std::is_same<typename std::decay<T>::type,
                 std::vector<typename T::value_type>>::value,
    T>::type
from_json(const rapidjson::Value &j) {
    assert(j.IsArray());
    T v;
    for (const rapidjson::Value &a : j.GetArray())
        v.push_back(from_json<typename T::value_type>(a));
    return v;
}

////////////////////////////////// DEPRECATED //////////////////////////////////

inline Vec3 getVec3(const rapidjson::Value &j) {
    return Vec3(j[0].GetDouble(), j[1].GetDouble(), j[2].GetDouble());
}

inline Vec4 getVec4(const rapidjson::Value &j) {
    return Vec4(j[0].GetDouble(), j[1].GetDouble(), j[2].GetDouble(),
                j[3].GetDouble());
}

inline Vec2i getVec2i(const rapidjson::Value &j) {
    return Vec2i(j[0].GetInt(), j[1].GetInt());
}

inline Vec3i getVec3i(const rapidjson::Value &j) {
    return Vec3i(j[0].GetInt(), j[1].GetInt(), j[2].GetInt());
}

inline Vec4i getVec4i(const rapidjson::Value &j) {
    return Vec4i(j[0].GetInt(), j[1].GetInt(), j[2].GetInt(), j[3].GetInt());
}

inline Quaternion getQuaternion(const rapidjson::Value &j) {
    return Quaternion(j[0].GetDouble(), j[1].GetDouble(), j[2].GetDouble(),
                      j[3].GetDouble());
}

inline Transform getTransform(const rapidjson::Value &j) {
    Matrix4 m;
    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 4; k++)
            m(i, k) = j[i * 4 + k].GetDouble();
    }
    return Transform(m);
}

inline Vec3Array getVec3Array(const rapidjson::Value &j) {
    assert(j.IsArray());
    Vec3Array v;
    for (const rapidjson::Value &a : j.GetArray())
        v.push_back(getVec3(a));
    return v;
}

inline Vec4Array getVec4Array(const rapidjson::Value &j) {
    assert(j.IsArray());
    Vec4Array v;
    for (const rapidjson::Value &a : j.GetArray())
        v.push_back(getVec4(a));
    return v;
}

inline Vec4iArray getVec4iArray(const rapidjson::Value &j) {
    assert(j.IsArray());
    Vec4iArray v;
    for (const rapidjson::Value &a : j.GetArray())
        v.push_back(getVec4i(a));
    return v;
}

inline Vec3iArray getVec3iArray(const rapidjson::Value &j) {
    assert(j.IsArray());
    Vec3iArray v;
    for (const rapidjson::Value &a : j.GetArray())
        v.push_back(getVec3i(a));
    return v;
}

inline Vec2iArray getVec2iArray(const rapidjson::Value &j) {
    assert(j.IsArray());
    Vec2iArray v;
    for (const rapidjson::Value &a : j.GetArray())
        v.push_back(getVec2i(a));
    return v;
}

inline IntArray getIntArray(const rapidjson::Value &j) {
    assert(j.IsArray());
    IntArray v;
    for (const rapidjson::Value &a : j.GetArray())
        v.push_back(a.GetInt());
    return v;
}

} // namespace viper
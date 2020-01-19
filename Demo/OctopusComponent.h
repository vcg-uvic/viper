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

#include <cfloat>

#include <OpenGP/GL/Components/TrackballComponent.h>
#include <OpenGP/GL/Components/WorldRenderComponent.h>
#include <OpenGP/GL/Scene.h>
#include <OpenGP/SphereMesh/GL/SphereMeshRenderer.h>
#include <OpenGP/SurfaceMesh/GL/SurfaceMeshRenderer.h>

#include "tiny_obj_loader.h"

#include "Octopus.h"
#include "Scene.h"
#include "Subprocess.h"
#include "Viper_json.h"

#define VIPER_TEXTURE 0
#define VIPER_GREY 1
#define VIPER_GOLD 2
#define VIPER_CLEAR_RED 3

namespace OpenGP {

struct OctopusData {
    std::vector<int> radius_constraints;
    std::vector<int> volume_constraints;
    std::vector<int> stretch_constraints;
    std::vector<int> distance_constraints;
    std::vector<int> bend_constraints;
    std::vector<int> cannonball_constraints;

    void set_enabled(bool enabled, viper::Scene &scene) {
        for (auto i : radius_constraints)
            scene.constraints.radius[i].enabled = enabled;
        for (auto i : volume_constraints)
            scene.constraints.volume[i].enabled = enabled;
        for (auto i : stretch_constraints)
            scene.constraints.stretch[i].enabled = enabled;
        for (auto i : distance_constraints)
            scene.constraints.distance[i].enabled = enabled;
        for (auto i : bend_constraints)
            scene.constraints.bend[i].enabled = enabled;
    }

    void set_cannonball_enabled(bool enabled, viper::Scene &scene) {
        for (auto i : cannonball_constraints)
            scene.constraints.distance[i].enabled = enabled;
    }
};

class OctopusComponent : public Component {
  public:
    static viper::Scene *v_scene;

    const int n_cows = 90;
    int n_active = 90;

    int scene_index = 0;

    bool cannonballs_active = false;

    WorldRenderComponent *render_comp;
    SurfaceMeshRenderer *renderer;
    WorldRenderComponent *sphere_render_comp;
    SphereMeshRenderer *sphere_renderer;
    WorldRenderComponent *tsphere_render_comp;
    SphereMeshRenderer *tsphere_renderer;
    WorldRenderComponent *cannonball_render_comp;
    SphereMeshRenderer *cannonball_renderer;
    WorldRenderComponent *pillar_render_comp;
    SphereMeshRenderer *pillar_renderer;

    std::vector<std::vector<int>> v_ids, p_ids;
    std::vector<int> cannonball_ids;
    std::vector<int> pillar_ids;
    std::vector<OctopusData> cow_data;

    SurfaceMesh mesh;
    OpenGP::SphereMesh smesh, cannonball_smesh;

    std::vector<Vec4> spheres;
    std::vector<Vec2i> all_pills;
    std::vector<int> control_pills;
    std::vector<float> compliances;
    std::vector<float> masses;

    std::vector<std::vector<Mat4x4>> init_transforms;

    using V = OpenGP::SphereMesh::Vertex;

    void init() {
        render_comp = &(require<WorldRenderComponent>());
        renderer = &(render_comp->set_renderer<SurfaceMeshRenderer>());

        Material octomat(R"GLSL(

            flat out int gid;

            void vertex_shade() {

                gid = gl_InstanceID;

            }

        )GLSL",
                        R"GLSL(

            flat in int gid[];
            flat out int fid;

            void geometry_vertex_shade(int v) {
                fid = gid[v];
            }

        )GLSL",
                        R"GLSL(

            uniform sampler2D diffuse;
            uniform sampler2D shadow_map;
            uniform int material;

            uniform vec3 light_pos;
            uniform mat4 shadow_matrix;
            uniform float shadow_near;
            uniform float shadow_far;

            flat in int fid;

            vec3 world2uvdepth(vec3 pos, mat4 mat) {
                vec4 a = mat * vec4(pos, 1);
                vec3 b = a.xyz / a.w;
                return (b + vec3(1)) / 2;
            }

            float get_shadow_mask(vec2 uv) {
                return 1 - smoothstep(0.3, 0.5, length(uv - vec2(0.5, 0.5)));
            }

            vec3 get_ambient(vec3 pos) {
                vec3 ambient = vec3(0.14, 0.14, 0.18);

                vec3 uvd = world2uvdepth(pos, shadow_matrix);

                return ambient + vec3(0.2) * get_shadow_mask(uvd.xy);
            }

            float linear_shadow_depth(float d) {
                return shadow_near * shadow_far / (shadow_far + d * (shadow_near - shadow_far));
            }

            float get_shadow(vec3 pos) {
                ivec2 dim = textureSize(shadow_map, 0);
                vec3 uvd = world2uvdepth(pos, shadow_matrix);

                vec2 base_coord = uvd.xy * dim;
                ivec2 base_coord_i = ivec2(floor(base_coord));
                vec2 inter = fract(base_coord);

                mat4 shadow_depths;
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        shadow_depths[i][j] = linear_shadow_depth(texelFetch(shadow_map, base_coord_i + ivec2(i-1, j-1), 0).r);
                    }
                }

                float threshold = linear_shadow_depth(uvd.z) - 0.4;

                mat2 pcf_vals = mat2(0);
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        for (int x = 0; x < 3; ++x) {
                            for (int y = 0; y < 3; ++y) {
                                pcf_vals[i][j] += (shadow_depths[x + i][y + j] < threshold) ? 0 : (1.0 / 9.0);
                            }
                        }
                    }
                }

                float a = mix(pcf_vals[0][0], pcf_vals[1][0], inter.x);
                float b = mix(pcf_vals[0][1], pcf_vals[1][1], inter.x);

                return mix(a, b, inter.y) * get_shadow_mask(uvd.xy);
            }

            // http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl

            vec3 rgb2hsv(vec3 c) {
                vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

                float d = q.x - min(q.w, q.y);
                float e = 1.0e-10;
                return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
            }

            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }

            vec4 fragment_shade() {

                vec3 pos = get_position();

                vec3 lightdir = normalize(light_pos - pos);

                vec3 colors[7];
                colors[0] = vec3(1.00f, 0.98f, 0.28f);
                colors[1] = vec3(0.00f, 0.68f, 0.49f);
                colors[2] = vec3(0.14f, 0.49f, 0.76f);
                colors[3] = vec3(0.92f, 0.40f, 0.00f);
                colors[4] = vec3(0.00f, 0.80f, 1.00f);
                colors[5] = vec3(0.89f, 0.52f, 0.71f);
                colors[6] = vec3(1.00f, 0.68f, 0.00f);

                float spec_power = 80;
                vec3 diffuse_color = vec3(1, 1, 1);
                float alpha = 1.0;
                if (material == 0) {
                    vec3 base_color = colors[fid % 7];

                    vec2 uv = get_texcoord();
                    diffuse_color = texture(diffuse, vec2(uv.x, 1 - uv.y)).rgb;

                    vec3 diffuse_hsv = rgb2hsv(diffuse_color);
                    diffuse_hsv.r = rgb2hsv(base_color).r;
                    diffuse_color = hsv2rgb(diffuse_hsv);
                }
                if (material == 2) {
                    diffuse_color = vec3(220, 200, 120) / 255.0;
                    spec_power = 200;
                }
                if (material == 3) {
                    diffuse_color = vec3(220, 60, 60) / 255.0;
                    alpha = 0.5;
                }

                vec3 ambient = get_ambient(pos);
                ambient *= (1 + abs(dot(get_normal(), get_forward()))) / 2;

                float shadow = get_shadow(pos);

                vec3 out_color = shadow * 0.85 * clamp(dot(get_normal(), normalize(lightdir)), 0, 1) * diffuse_color;
                out_color += vec3(1) * shadow * pow(clamp(dot(get_forward(), reflect(lightdir, get_normal())), 0, 1), spec_power);

                out_color += ambient * diffuse_color;

                return vec4(out_color, alpha);
            }

        )GLSL");
        octomat.set_property("material", VIPER_TEXTURE);
        octomat.set_property("diffuse", 5);
        octomat.set_property("ao_map", 6);
        octomat.set_property("shadow_map", 7);

        renderer->set_material(octomat);
        renderer->rebuild();
        render_comp->visible = true;

        sphere_render_comp =
            &(get_scene().create_entity_with<WorldRenderComponent>());
        sphere_renderer =
            &(sphere_render_comp->set_renderer<SphereMeshRenderer>());

        sphere_renderer->set_material(octomat);
        sphere_renderer->get_material().set_property("material", VIPER_GOLD);
        sphere_renderer->rebuild();
        sphere_render_comp->visible = false;

        tsphere_render_comp =
            &(get_scene().create_entity_with<WorldRenderComponent>());
        tsphere_renderer =
            &(tsphere_render_comp->set_renderer<SphereMeshRenderer>());

        tsphere_renderer->set_material(octomat);
        tsphere_renderer->no_spheres = true;
        tsphere_renderer->get_material().set_property("material",
                                                      VIPER_CLEAR_RED);
        tsphere_renderer->rebuild();
        tsphere_render_comp->visible = false;

        cannonball_render_comp =
            &(get_scene().create_entity_with<WorldRenderComponent>());
        cannonball_renderer =
            &(cannonball_render_comp->set_renderer<SphereMeshRenderer>());

        cannonball_renderer->set_material(octomat);
        cannonball_renderer->get_material().set_property("material",
                                                         VIPER_GREY);
        cannonball_renderer->rebuild();

        pillar_render_comp =
            &(get_scene().create_entity_with<WorldRenderComponent>());
        pillar_renderer =
            &(pillar_render_comp->set_renderer<SphereMeshRenderer>());

        pillar_renderer->set_material(octomat);
        pillar_renderer->get_material().set_property("material", VIPER_GREY);
        pillar_renderer->rebuild();

        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {

                float r = 0.5;
                Vec3 c0 = Vec3(4 * (i - 2), 0, 4 * (j - 2));
                Vec3 c1 = Vec3(4 * (i - 2), 4, 4 * (j - 2));

                int id0 = v_scene->addParticle(c0, r, 1.0, true);
                int id1 = v_scene->addParticle(c1, r, 1.0, true);

                pillar_ids.push_back(id0);
                pillar_ids.push_back(id1);

                v_scene->addPill(id0, id1, true);
            }
        }

        {
            SurfaceMesh tempmesh;
            std::ifstream istream("mesh.bin", std::ios::binary);

            std::vector<Vec3> pointvec(24842);
            istream.read(reinterpret_cast<char*>(&pointvec[0]), 24842 * sizeof(Vec3));
            
            std::vector<Vec3> normalvec(24842);
            istream.read(reinterpret_cast<char*>(&normalvec[0]), 24842 * sizeof(Vec3));
            
            std::vector<Vec2> texvec(24842);
            istream.read(reinterpret_cast<char*>(&texvec[0]), 24842 * sizeof(Vec2));
            
            std::vector<int> pidxvec(24842);
            istream.read(reinterpret_cast<char*>(&pidxvec[0]), 24842 * sizeof(int));
            
            std::vector<int> trivec(12312 * 3);
            istream.read(reinterpret_cast<char*>(&trivec[0]), 12312 * 3 * sizeof(int));

            for (int i = 0; i < 24842; ++i) {
                auto vert = tempmesh.add_vertex(pointvec[i]);
            }

            auto vtexcoord = tempmesh.add_vertex_property<Vec2>("v:texcoord");
            auto vnormal = tempmesh.add_vertex_property<Vec3>("v:normal");
            auto vpindex = tempmesh.add_vertex_property<int>("v:pindex");

            for (int i = 0; i < 24842; ++i) {
                SurfaceMesh::Vertex vert(i);
                vnormal[vert] = normalvec[i];
                vtexcoord[vert] = texvec[i];
                vpindex[vert] = pidxvec[i];
            }
            for (int i = 0; i < 12312; ++i) {
                tempmesh.add_triangle(SurfaceMesh::Vertex(trivec[3*i]),
                                SurfaceMesh::Vertex(trivec[3*i + 1]),
                                SurfaceMesh::Vertex(trivec[3*i + 2]));
            }

            mesh = tempmesh;
        }

        cow::get_octopus(spheres, all_pills, control_pills, masses,
                         compliances);

        for (int i = 0; i < spheres.size(); ++i) {
            auto &v = spheres[i];
            smesh.add_vertex(v);
        }

        for (int i : control_pills) {
            auto pill = all_pills[i];
            smesh.add_edge(V(pill[0]), V(pill[1]));
        }

        std::string output;

        auto vpindex = mesh.get_vertex_property<int>("v:pindex");
        std::map<int, int> ind_verts;
        {
            // build watertight mesh
            std::vector<Vec3> verts;
            for (auto vert : mesh.vertices()) {
                int ind = vpindex[vert];
                if (ind_verts.find(ind) == ind_verts.end()) {
                    ind_verts[ind] = verts.size();
                    verts.push_back(mesh.position(vert));
                }
            }

            std::vector<Vec3i> tris;
            for (auto face : mesh.faces()) {
                Vec3i tri;
                int i = 0;
                for (auto vert : mesh.vertices(face))
                    tri[i++] = ind_verts[vpindex[vert]];
                tris.push_back(tri);
            }

            // compute weights

            rapidjson::MemoryPoolAllocator<> alloc;
            rapidjson::Document j(&alloc);
            j.SetObject();
            rapidjson::Document::AllocatorType &allocator = j.GetAllocator();

            j.AddMember("vertices", viper::to_json(verts, allocator),
                        allocator);
            j.AddMember("triangles", viper::to_json(tris, allocator),
                        allocator);

            rapidjson::Value spheres_array(rapidjson::kArrayType);
            rapidjson::Value pills_array(rapidjson::kArrayType);

            j.AddMember("spheres",
                        viper::to_json(
                            smesh.get_vertex_property<Vec4>("v:point").vector(),
                            allocator),
                        allocator);
            j.AddMember(
                "pills",
                viper::to_json(
                    smesh.get_edge_property<Vec2i>("e:connectivity").vector(),
                    allocator),
                allocator);

            rapidjson::StringBuffer strbuf;
            strbuf.Clear();
            rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
            j.Accept(writer);

            int proc_return =
                viper::run_process("sphereweights", strbuf.GetString(), output);
            std::cout << "Sphereweights exited with code: " << proc_return
                      << std::endl;
        }

        {
            rapidjson::Document j;
            j.Parse(output.c_str());

            auto all_weights =
                viper::from_json<std::vector<std::vector<float>>>(j["weights"]);
            auto all_bones =
                viper::from_json<std::vector<std::vector<int>>>(j["bone_ids"]);

            std::vector<Vec4> weights(all_weights.size());
            std::vector<Vec4i> bones(all_bones.size());

            for (int i = 0; i < weights.size(); ++i) {
                while (all_weights[i].size() < 4) {
                    all_weights[i].push_back(0.f);
                    all_bones[i].push_back(0);
                }
                weights[i] = Vec4(all_weights[i][0], all_weights[i][1],
                                  all_weights[i][2], all_weights[i][3]);
                float sum = weights[i].sum();
                weights[i] /= sum;
                bones[i] = Vec4i(all_bones[i][0], all_bones[i][1],
                                 all_bones[i][2], all_bones[i][3]);
            }

            auto weights_prop = mesh.add_vertex_property<Vec4>("v:skinweight");
            auto bone_ids_prop = mesh.add_vertex_property<Vec4i>("v:boneid");

            for (auto vert : mesh.vertices()) {
                weights_prop[vert] = weights[ind_verts[vpindex[vert]]];
                bone_ids_prop[vert] = bones[ind_verts[vpindex[vert]]];
            }
        }

        v_ids.resize(n_cows);
        p_ids.resize(n_cows);
        cannonball_ids.resize(n_cows);

        for (int cow_id = 0; cow_id < n_cows; ++cow_id) {

            OctopusData data;

            Vec4 cannonball_s(0, 0, -1.5, 0.7);

            cannonball_ids[cow_id] = v_scene->addParticle(
                cannonball_s.head<3>(), cannonball_s[3], 0.05);
            v_scene->pInfo[cannonball_ids[cow_id]].group = 1 + cow_id * 2;
            v_scene->addPill(cannonball_ids[cow_id], cannonball_ids[cow_id]);
            cannonball_smesh.add_sphere(
                cannonball_smesh.add_vertex(cannonball_s));

            for (int i = 0; i < spheres.size(); ++i) {
                auto &v = spheres[i];
                float w = 1.0 / masses[i];
                v_ids[cow_id].push_back(
                    v_scene->addParticle(v.head<3>(), v[3], w));
                v_scene->pInfo[v_ids[cow_id].back()].group = cow_id * 2;

                data.radius_constraints.push_back(
                    v_scene->constraints.radius.size());
                v_scene->constraints.radius.push_back(
                    viper::C_radius(v_ids[cow_id].back(), v[3], 1e-3));

                if (i >= 10 && i < 18) {

                    data.cannonball_constraints.push_back(
                        v_scene->constraints.distance.size());
                    v_scene->constraints.distance.push_back(viper::C_distance(
                        v_ids[cow_id].back(), cannonball_ids[cow_id],
                        cannonball_s[3], 0.f));
                }
            }

            for (int i = 0; i < all_pills.size(); i++) {
                auto pill = all_pills[i];
                int p_id = v_scene->addPill(v_ids[cow_id][pill[0]],
                                            v_ids[cow_id][pill[1]]);
                p_ids[cow_id].push_back(p_id);

                float d = (spheres[pill[0]] - spheres[pill[1]]).norm();

                data.volume_constraints.push_back(
                    v_scene->constraints.volume.size());
                v_scene->constraints.volume.push_back(
                    viper::C_volume(v_ids[cow_id][pill[0]],
                                    v_ids[cow_id][pill[1]], v_scene->state));
                float compliance = 1e-4;
                data.stretch_constraints.push_back(
                    v_scene->constraints.stretch.size());
                v_scene->constraints.stretch.push_back(viper::C_stretch(
                    v_ids[cow_id][pill[0]], v_ids[cow_id][pill[1]], p_id, d,
                    compliance));
            }
            for (int i = 0; i < all_pills.size(); ++i) {
                for (int j = i + 1; j < all_pills.size(); ++j) {
                    auto pill_i = all_pills[i];
                    auto pill_j = all_pills[j];
                    if (pill_i[0] == pill_j[0] || pill_i[0] == pill_j[1] ||
                        pill_i[1] == pill_j[0] || pill_i[1] == pill_j[1]) {
                        float compliance =
                            (compliances[i] + compliances[j]) / 2;
                        data.bend_constraints.push_back(
                            v_scene->constraints.bend.size());
                        v_scene->constraints.bend.push_back(
                            viper::C_bend(p_ids[cow_id][i], p_ids[cow_id][j],
                                          v_scene->state, compliance));
                    }
                }
            }

            cow_data.push_back(data);
        }

        init_transforms = get_transforms();
        renderer->upload_mesh(mesh, get_transforms());
        renderer->get_gpu_mesh().set_vtexcoord(
            mesh.get_vertex_property<Vec2>("v:texcoord").vector());

        reset();

        smesh.clear();

        int offset = 0;
        for (int j = 0; j < n_cows; ++j) {
            for (int i = 0; i < spheres.size(); ++i) {
                auto &v = spheres[i];
                smesh.add_vertex(v);
            }

            for (int i : control_pills) {
                auto pill = all_pills[i];
                smesh.add_edge(V(pill[0] + offset), V(pill[1] + offset));
            }
            offset += spheres.size();
        }

        reset();
    }

    void reset() {
        v_scene->reset();

        bool pillars_active;

        switch (scene_index) {
        case 0: {
            cannonballs_active = false;
            pillars_active = false;

            for (int cow_id = 0; cow_id < n_cows; ++cow_id) {
                set_position(cow_id, Vec3::Random() * 10 + Vec3(0, 12, 0),
                             Vec3::Zero());
            }
            break;
        }
        case 1: {
            cannonballs_active = false;
            pillars_active = true;

            for (int cow_id = 0; cow_id < n_cows; ++cow_id) {
                set_position(cow_id, Vec3::Random() * 10 + Vec3(0, 12, 0),
                             Vec3::Zero());
            }
            break;
        }
        case 2: {
            cannonballs_active = true;
            pillars_active = false;

            for (int cow_id = 0; cow_id < n_cows; ++cow_id) {
                set_position(cow_id, Vec3::Random() * 10 + Vec3(0, 12, 0),
                             Vec3::Zero());
            }
            break;
        }
        case 3: {
            cannonballs_active = false;
            pillars_active = false;

            for (int cow_id = 0; cow_id < n_cows; ++cow_id) {
                set_position(cow_id, Vec3::Zero(), Vec3::Zero());
            }
            break;
        }
        }

        cannonball_render_comp->visible = cannonballs_active;
        pillar_render_comp->visible = pillars_active;

        for (auto &data : cow_data) {
            data.set_cannonball_enabled(cannonballs_active, *v_scene);
        }

        for (int id : cannonball_ids) {
            v_scene->state.xa[id] = cannonballs_active;
            v_scene->state.xai[id] = cannonballs_active;

            if (!cannonballs_active) {
                v_scene->state.x[id] = Vec3(0, -2, 0);
            }
        }

        for (int i = 0; i < pillar_ids.size(); ++i) {
            float offset = pillars_active ? 0 : -5;
            v_scene->state.x[pillar_ids[i]][1] =
                offset + ((i % 2 == 0) ? 0 : 4);
        }
    }

    void update() {}

    int intersect(Vec3 eye, Vec3 dir) {
        float best_dist = FLT_MAX;
        int best_id = -1;

        for (auto &ids : v_ids) {
            for (int i : ids) {

                Vec3 c = v_scene->state.x[i];
                float r = v_scene->state.r[i];

                float a = dir.dot(eye - c);
                float b = -(dir.dot(eye - c));
                float det = a * a - (eye - c).squaredNorm() + r * r;

                if (det <= 0)
                    continue;

                float d0 = b + std::sqrt(det);
                float d1 = b - std::sqrt(det);

                if (d0 < 0)
                    d0 = FLT_MAX;
                if (d1 < 0)
                    d1 = FLT_MAX;
                float d = std::min(d0, d1);

                if (d < best_dist) {
                    best_dist = d;
                    best_id = i;
                }
            }
        }

        return best_id;
    }

    void vis_update() {

        if (sphere_render_comp->visible) {

            auto vpoint = smesh.get_vertex_property<Vec4>("v:point");

            for (int j = 0; j < n_cows; ++j) {
                for (int i = 0; i < spheres.size(); ++i) {

                    vpoint[V(i + j * spheres.size())].head<3>() =
                        v_scene->state.x[v_ids[j][i]];
                    vpoint[V(i + j * spheres.size())][3] =
                        v_scene->state.r[v_ids[j][i]];
                }
            }

            SphereMesh new_smesh;
            for (auto vert : smesh.vertices()) {
                auto v = new_smesh.add_vertex(vpoint[vert]);
                new_smesh.add_sphere(v);
            }

            sphere_renderer->upload_mesh(new_smesh);

            for (auto vert : smesh.vertices()) {
                vpoint[vert][3] *= 1.02;
            }

            tsphere_renderer->upload_mesh(smesh);
        }

        if (render_comp->visible) {

            auto transforms = get_transforms();
            renderer->upload_transforms(transforms);
        }

        if (cannonball_render_comp->visible) {
            auto cannonball_vpoint =
                cannonball_smesh.get_vertex_property<Vec4>("v:point");
            for (int i = 0; i < n_cows; ++i) {
                cannonball_vpoint[V(i)].head<3>() =
                    v_scene->state.x[cannonball_ids[i]];
            }

            cannonball_renderer->upload_mesh(cannonball_smesh);
        }

        if (pillar_render_comp->visible) {
            SphereMesh pillar_smesh;
            for (int i = 0; i < pillar_ids.size(); i += 2) {

                Vec3 c0 = v_scene->state.x[i];
                Vec3 c1 = v_scene->state.x[i + 1];

                float r0 = v_scene->state.r[i];
                float r1 = v_scene->state.r[i + 1];

                auto v0 = pillar_smesh.add_vertex(c0, r0);
                auto v1 = pillar_smesh.add_vertex(c1, r1);

                pillar_smesh.add_edge(v0, v1);
            }

            pillar_renderer->upload_mesh(pillar_smesh);
        }
    }

    void set_position(int i, Vec3 pos, Vec3 v) {
        for (auto id : v_ids[i]) {
            v_scene->state.x[id] = v_scene->state.xi[id] + pos;
            v_scene->state.xp[id] = v_scene->state.x[id] - v;
            v_scene->state.r[id] = v_scene->state.ri[id];
            v_scene->state.rp[id] = v_scene->state.ri[id];
        }
        for (auto id : p_ids[i]) {
            v_scene->state.q[id] = v_scene->state.qi[id];
            v_scene->state.qp[id] = v_scene->state.qi[id];
        }
        if (cannonballs_active) {
            v_scene->state.x[cannonball_ids[i]] =
                v_scene->state.xi[cannonball_ids[i]] + pos;
            v_scene->state.xp[cannonball_ids[i]] =
                v_scene->state.x[cannonball_ids[i]] - v;
        }
    }

    std::vector<std::vector<Mat4x4>> get_transforms() const {

        std::vector<std::vector<Mat4x4>> transforms(n_cows);

        for (int j = 0; j < n_cows; ++j) {
            for (int i = 0; i < control_pills.size(); ++i) {
                int k = p_ids[j][control_pills[i]];

                int v0 = v_ids[j][all_pills[control_pills[i]][0]];
                int v1 = v_ids[j][all_pills[control_pills[i]][1]];

                Transform t0;
                t0.set_translation(v_scene->state.x[v0]);
                t0.apply_rotation(v_scene->state.q[k]);
                t0.set_scale(Vec3::Ones() * v_scene->state.r[v0]);

                transforms[j].push_back(t0.get_transformation_matrix());

                Transform t1;
                t1.set_translation(v_scene->state.x[v1]);
                t1.apply_rotation(v_scene->state.q[k]);
                t1.set_scale(Vec3::Ones() * v_scene->state.r[v1]);

                transforms[j].push_back(t1.get_transformation_matrix());
            }
        }

        return transforms;
    }
};

} // namespace OpenGP
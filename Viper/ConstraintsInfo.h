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

struct ConstraintsInfo {
    void add(const std::string &name, int count, int projections_per_constraint,
             int lambda_per_constraint) {
        assert(index.find(name) == index.end());
        int i = np.size();
        index[name] = i;
        if (i == 0) {
            o.push_back(0);
            ol.push_back(0);
        } else {
            o.push_back(o.back() + np.back());
            ol.push_back(ol.back() + nl.back());
        }
        nc.push_back(count);
        np.push_back(count * projections_per_constraint);
        nl.push_back(count * lambda_per_constraint);
    }

    int get_nc() {
        int n = 0;
        for (int i = 0; i < nc.size(); i++) {
            n += nc[i];
        }
        return n;
    }

    int get_np() {
        if (np.size() == 0)
            return 0;
        return o.back() + np.back();
    }

    int get_nl() {
        if (np.size() == 0)
            return 0;
        return ol.back() + nl.back();
    }

    std::map<std::string, int> get_o() {
        std::map<std::string, int> m;
        for (auto const &x : index)
            m[x.first] = o[x.second];
        return m;
    }

    std::map<std::string, int> get_ol() {
        std::map<std::string, int> m;
        for (auto const &x : index)
            m[x.first] = ol[x.second];
        return m;
    }

  private:
    std::map<std::string, int> index;
    std::vector<int> nc; // number of constraints per type
    std::vector<int> np; // number of projections per type
    std::vector<int> nl; // number of lambdas per type
    std::vector<int> o;  // projection offset per type
    std::vector<int> ol; // lambda offset per type
};
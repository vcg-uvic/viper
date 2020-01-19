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

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace viper {

inline int run_process(const std::string &commandline,
                       const std::string &params, std::string &output) {

    int param_pipefd[2];
    int return_pipefd[2];
    pipe(param_pipefd);
    pipe(return_pipefd);

    pid_t pid = fork();

    if (pid == 0) {
        close(0);
        dup(param_pipefd[0]);
        close(1);
        dup(return_pipefd[1]);

        close(return_pipefd[0]);
        close(param_pipefd[1]);

        execl(commandline.c_str(), nullptr);

        assert(false && "FATAL: child process exec failed");
    }

    close(return_pipefd[1]);
    close(param_pipefd[0]);

    const char *out_buf = params.c_str();
    int remaining = params.size();
    while (remaining != 0) {
        int amount = write(param_pipefd[1], out_buf, remaining);
        remaining -= amount;
        out_buf += amount;
    }

    close(param_pipefd[1]);

    output.clear();

    constexpr int BUF_SIZE = 4096;
    char in_buf[BUF_SIZE];

    while (1) {
        int amount = read(return_pipefd[0], in_buf, BUF_SIZE);
        if (amount <= 0)
            break;

        output += std::string(in_buf, amount);
    }

    // std::cout << "waiting" << std::endl;

    int wstatus;
    waitpid(pid, &wstatus, 0);

    // std::cout << "waiting done" << std::endl;

    return WEXITSTATUS(wstatus);
}

} // namespace viper
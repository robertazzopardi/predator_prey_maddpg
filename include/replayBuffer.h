#ifndef __REPLAY_BUFFER_H__
#define __REPLAY_BUFFER_H__

#include "action.h"
#include <array>
#include <stddef.h>
#include <tuple>
#include <vector>

namespace at {
class Tensor;
}

namespace replaybuffer {

using Experience = std::tuple<std::vector<at::Tensor>, std::vector<float>,
                              std::vector<float>, std::vector<at::Tensor>>;

using Sample =
    std::tuple<std::vector<std::vector<at::Tensor>>,
               std::vector<std::vector<float>>, std::vector<std::vector<float>>,
               std::vector<std::vector<at::Tensor>>, std::vector<at::Tensor>,
               std::vector<at::Tensor>, std::vector<at::Tensor>>;

extern std::vector<Experience> buffer;

void push(Experience);

Sample sample(int);

} // namespace replaybuffer

#endif // !__REPLAY_BUFFER_H__

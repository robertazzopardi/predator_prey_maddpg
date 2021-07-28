#ifndef __REPLAY_BUFFER_H__
#define __REPLAY_BUFFER_H__

// #include "action.h"
#include "env.h"
#include <array>
// #include <stddef.h>
#include <tuple>
#include <vector>

namespace at {
class Tensor;
}

namespace replaybuffer {

using Experience = std::tuple<std::vector<at::Tensor>, std::vector<float>,
                              std::vector<float>, std::vector<at::Tensor>>;

using Sample = std::tuple<std::array<std::vector<at::Tensor>, env::hunterCount>,
                          std::array<std::vector<float>, env::hunterCount>,
                          std::array<std::vector<float>, env::hunterCount>,
                          std::array<std::vector<at::Tensor>, env::hunterCount>,
                          std::array<at::Tensor, env::BATCH_SIZE>,
                          std::array<at::Tensor, env::BATCH_SIZE>,
                          std::array<at::Tensor, env::BATCH_SIZE>>;

extern std::vector<Experience> buffer;

void push(Experience);

// Sample sample(int);
Sample sample();

} // namespace replaybuffer

#endif // !__REPLAY_BUFFER_H__

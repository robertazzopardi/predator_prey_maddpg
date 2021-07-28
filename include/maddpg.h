
#ifndef __MADDPG_H__
#define __MADDPG_H__

#include "action.h" // for Action
#include "env.h"
#include "replayBuffer.h"
#include <vector>

namespace at {
class Tensor;
}

namespace maddpg {

std::vector<float> getActions(std::vector<torch::Tensor>);

void update(int);

void run(int, int, int);

} // namespace maddpg

#endif // !__MADDPG_H__

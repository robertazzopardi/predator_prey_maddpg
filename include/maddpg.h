
#ifndef __MADDPG_H__
#define __MADDPG_H__

#include <vector>

namespace at {
class Tensor;
}

namespace maddpg {

std::vector<float> getActions(std::vector<at::Tensor>);

void update();

void run(int, int);

} // namespace maddpg

#endif // !__MADDPG_H__

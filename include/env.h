
#ifndef __ENV_H__
#define __ENV_H__

#include "action.h"
#include <EnvController.h>
#include <RobotMonitor.h>
#include <memory>
#include <tuple>
#include <vector>

namespace at {
class Tensor;
}

namespace prey {
class Prey;
}

namespace env {

enum class Mode { TRAIN, EVAL };
extern enum Mode mode;

constexpr static auto GRID_SIZE = 5;
constexpr static auto BATCH_SIZE = 64;

static inline auto getEnvSize() {
    return static_cast<int>(GRID_SIZE) * robosim::envcontroller::getCellWidth();
};

static constexpr auto hunterCount = 4;
static constexpr auto preyCount = 1;
static constexpr auto agentCount = hunterCount + preyCount;

// extern std::shared_ptr<prey::Prey> prey;
// extern robosim::envcontroller::MonitorVec robots;

std::vector<at::Tensor> reset();

bool isSamePosition(robosim::envcontroller::RobotPtr);

std::tuple<std::vector<at::Tensor>, std::vector<float>, bool>
    step(std::vector<float>);
// step(std::vector<action::Action>);

int getRandomPos();

} // namespace env

#endif // !__ENV_H__

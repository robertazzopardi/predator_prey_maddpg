#include "../include/prey.h"
#include "../include/env.h"
#include "../include/hunter.h"
#include <Colour.h>                                            // for Colour
#include <EnvController.h>                                     // for Monito...
#include <memory>                                              // for shared...
#include <sys/cdefs.h>                                         // for __unused
#include <torch/csrc/autograd/generated/variable_factories.h>  // for tensor
#include <type_traits>                                         // for remove...
#include <vector>                                              // for vector

namespace hunter {
class Hunter;
}

prey::Prey::Prey(bool verbose, colour::Colour colour)
    : agent::Agent(verbose, colour), dist(0, action::ACTION_COUNT) {}

float prey::Prey::getAction(torch::Tensor states __unused) { return dist(mt); }

torch::Tensor prey::Prey::getObservation() {
    return torch::tensor({1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
}

bool prey::Prey::isTrapped() {
    auto x = getGridX();
    auto y = getGridY();

    auto count = 0;

    if (y + 1 == env::GRID_SIZE - 1) {
        count++;
    }
    if (y - 1 == 0) {
        count++;
    }
    if (x + 1 == env::GRID_SIZE - 1) {
        count++;
    }
    if (x - 1 == 0) {
        count++;
    }

    for (auto var : robosim::envcontroller::robots) {
        if (var.get() != this &&
            std::static_pointer_cast<hunter::Hunter>(var)->isAtGoal()) {
            count++;
        }
    }

    return count > 3;
}

float prey::Prey::getReward(action::Action action __unused) { return 0.0f; }

void prey::Prey::update(agent::UpdateData __unused) {}

void prey::Prey::updateTarget() {}


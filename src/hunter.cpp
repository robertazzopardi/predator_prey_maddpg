#include "hunter.h"
#include "direction.h"
#include "env.h"
#include "models.h"
#include <ATen/Functions.h>
#include <ATen/TensorOperators.h>
#include <Colour.h>
#include <c10/core/Scalar.h>
#include <memory>
#include <stddef.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/modules/loss.h>
#include <torch/nn/pimpl.h>
#include <torch/nn/utils/clip_grad.h>
#include <type_traits>

// namespace c10 {
// template <typename T> class ArrayRef;
// }

namespace {} // namespace

hunter::Hunter::Hunter(bool verbose, colour::Colour colour)
    : agent::Agent(verbose, colour),
      criticOptimiser(critic->parameters(), 0.001),
      actorOptimiser(actor->parameters(), 0.0004) {

    for (size_t i = 0; i < critic->parameters().size(); i++) {
        targetCritic->parameters()[i].data().copy_(
            critic->parameters()[i].data());
    }

    for (size_t i = 0; i < actor->parameters().size(); i++) {
        targetActor->parameters()[i].data().copy_(
            actor->parameters()[i].data());
    }
}

float hunter::Hunter::getReward() { return 0.0f; }

// action::Action
float hunter::Hunter::getAction(torch::Tensor states) {
    auto output = actor->forward(states);
    auto nextAction = actor->nextAction(output);
    // auto action = action::ActionVec[nextAction];
    // return action;
    return static_cast<float>(nextAction);
}

torch::Tensor hunter::Hunter::getObservation() {
    return torch::tensor(
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
}

bool hunter::Hunter::isAtGoal() {
    int px = env::prey->getX();
    int py = env::prey->getY();
    int x = getX();
    int y = getY();
    return (x == direction::Direction(direction::Dir::UP).px(px) &&
            y == direction::Direction(direction::Dir::UP).py(py)) ||
           (x == direction::Direction(direction::Dir::DOWN).px(px) &&
            y == direction::Direction(direction::Dir::DOWN).py(py)) ||
           (x == direction::Direction(direction::Dir::LEFT).px(px) &&
            y == direction::Direction(direction::Dir::LEFT).py(py)) ||
           (x == direction::Direction(direction::Dir::RIGHT).px(px) &&
            y == direction::Direction(direction::Dir::RIGHT).py(py));
}

void hunter::Hunter::update(agent::UpdateData updateData) {
    auto [indivRewardBatchI, indivObsBatch, globalStateBatch, globalActionBatch,
          globalNextStateBatch, nextGlobalActions] = updateData;

    auto irb = torch::tensor(indivRewardBatchI);
    irb.view({(int)irb.size(0), 1});

    auto iob = torch::stack(indivObsBatch);

    auto gsb = torch::stack(globalStateBatch);

    auto gab = torch::stack(globalActionBatch);

    auto gnsb = torch::stack(globalNextStateBatch);

    auto nga = nextGlobalActions;

    criticOptimiser.zero_grad();

    // auto currQ = critic->forward(gsb, gab);
}

void hunter::Hunter::updateTarget() {
    for (size_t i = 0; i < actor->parameters().size(); i++) {
        targetActor->parameters()[i].data().copy_(
            actor->parameters()[i].data());
    }

    for (size_t i = 0; i < critic->parameters().size(); i++) {
        targetCritic->parameters()[i].data().copy_(
            critic->parameters()[i].data() * tau *
            targetCritic->parameters()[i].data() * (1.0f - tau));
    }
}

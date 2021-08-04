#include "../include/hunter.h"
#include "../include/direction.h"
#include "../include/models.h"
#include "../include/prey.h"
#include <ATen/Functions.h>
#include <ATen/TensorOperators.h>
#include <Colour.h>
#include <EnvController.h>  // for getCel...
#include <array>            // for array
#include <c10/core/Scalar.h>
#include <memory>
#include <stddef.h>
#include <sys/cdefs.h>  // for __unused
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/modules/loss.h>
#include <torch/nn/pimpl.h>
#include <torch/nn/utils/clip_grad.h>
#include <type_traits>
#include <vector>  // for vector

namespace prey {
class Prey;
}

namespace {

static int getMaxValueIndex(float *values, int count) {
    int maxAt = 0;

    for (int i = 0; i < count; i++) {
        maxAt = values[i] > values[maxAt] ? i : maxAt;
    }

    return maxAt;
}

static inline float normalise(int x, int min, int max) {
    return (2 * (static_cast<float>((x - min)) / (max - min))) - 1;
    // return 1 - (x - min) / (float) (max - min);
    // return x;
}

}  // namespace

hunter::Hunter::Hunter(bool verbose, colour::Colour colour)
    : agent::Agent(verbose, colour),
      criticOptimiser(critic->parameters(), 1e-3),
      actorOptimiser(actor->parameters(), 1e-4), distRand(0, 1),
      randAction(0, action::ACTION_COUNT) {

    for (size_t i = 0; i < critic->parameters().size(); i++) {
        targetCritic->parameters()[i].data().copy_(
            critic->parameters()[i].data());
    }

    for (size_t i = 0; i < actor->parameters().size(); i++) {
        targetActor->parameters()[i].data().copy_(
            actor->parameters()[i].data());
    }
}

float hunter::Hunter::getReward(action::Action action __unused) {
    // if (action == action::Action::NOTHING) {
    //     return 10.0f;
    // } else {
    //     return -10.0f;
    // }

    if (isAtGoal()) {
        return 0.0f;
    }
    return -0.1f;
}

float hunter::Hunter::getAction(torch::Tensor x) {
    // auto output = actor->forward(states);
    // auto nextAction = actor->nextAction(output);
    // return static_cast<float>(nextAction);

    float random = distRand(mt);
    if (random < epsilon) {
        auto action = randAction(mt);
        if (epsilon > 0) epsilon *= 0.997;
        return action;
    }
    if (epsilon > 0) epsilon *= 0.997;

    auto output = actor->forward(x);

    auto outputValues = static_cast<float *>(output.data_ptr());

    auto maxValue = getMaxValueIndex(outputValues, output.size(0));

    return maxValue;
}

torch::Tensor hunter::Hunter::getObservation() {
    static std::array<float, obsDim> observation;
    int count = 0;

    for (auto var : robosim::envcontroller::robots) {
        observation[count++] = normalise(
            var->getGridX(), 0, robosim::envcontroller::getCellWidth());
        observation[count++] = normalise(
            var->getGridY(), 0, robosim::envcontroller::getCellWidth());
    }

    return torch::from_blob(std::move(observation.data()), {obsDim});
}

bool hunter::Hunter::isAtGoal() {
    auto x = getX();
    auto y = getY();

    for (auto var : robosim::envcontroller::robots) {
        auto prey = std::dynamic_pointer_cast<prey::Prey>(var);
        if (prey) {
            auto px = prey->getX();
            auto py = prey->getY();
            return (x == direction::Direction(direction::Dir::UP).px(px) &&
                    y == direction::Direction(direction::Dir::UP).py(py)) ||
                   (x == direction::Direction(direction::Dir::DOWN).px(px) &&
                    y == direction::Direction(direction::Dir::DOWN).py(py)) ||
                   (x == direction::Direction(direction::Dir::LEFT).px(px) &&
                    y == direction::Direction(direction::Dir::LEFT).py(py)) ||
                   (x == direction::Direction(direction::Dir::RIGHT).px(px) &&
                    y == direction::Direction(direction::Dir::RIGHT).py(py));
        }
    }
    return false;
}

void hunter::Hunter::update(agent::UpdateData updateData) {
    auto [indivRewardBatchI, indivObsBatch, globalStateBatch, globalActionBatch,
          globalNextStateBatch, nextGlobalActions] = updateData;

    auto irb = torch::tensor(indivRewardBatchI);
    // irb.view({irb.size(0), 1});

    auto iob = torch::stack(indivObsBatch);

    auto gsb = torch::stack(globalStateBatch);

    auto gab = torch::stack(globalActionBatch);

    auto gnsb = torch::stack(globalNextStateBatch);

    auto nga = nextGlobalActions;

    // update critic
    criticOptimiser.zero_grad();

    auto currQ = critic->forward(gsb, gab);
    auto nextQ = targetCritic->forward(gnsb, nga);
    // auto estimQ = irb + (gamma * nextQ);
    auto estimQ = irb.reshape({irb.size(0), 1}) + (gamma * nextQ);

    auto criticLoss = MSELoss(currQ, estimQ.detach());
    criticLoss.backward();
    torch::nn::utils::clip_grad_norm_(critic->parameters(), 0.5);
    criticOptimiser.step();

    // update actor
    actorOptimiser.zero_grad();

    auto policyLoss = -critic->forward(gsb, gab).mean();
    auto currPolOut = actor->forward(iob);
    policyLoss += -(torch::pow(currPolOut, 2)).mean() * 1e-3;
    policyLoss.backward();
    torch::nn::utils::clip_grad_norm_(critic->parameters(), 0.5);
    actorOptimiser.step();
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

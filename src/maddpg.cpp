#include "maddpg.h"
#include "models.h"               // for Actor
#include <ATen/Functions.h>       // for stack
#include <ATen/TensorOperators.h> // for Tensor...
#include <ATen/core/TensorBody.h> // for Tensor
#include <EnvController.h>        // for isRunning
#include <RobotMonitor.h>         // for Monito...
#include <__tuple>                // for tuple_...
#include <chrono>                 // for millis...
#include <iostream>               // for operat...
#include <memory>
#include <numeric>                                            // for accumu...
#include <stddef.h>                                           // for size_t
#include <thread>                                             // for sleep_for
#include <torch/csrc/autograd/generated/variable_factories.h> // for tensor
#include <type_traits>                                        // for remove...

namespace agent {
class Agent;
}

// std::vector<action::Action>
// maddpg::getActions(std::vector<torch::Tensor> states) {
//     std::vector<action::Action> actions;
std::vector<float> maddpg::getActions(std::vector<torch::Tensor> states) {
    std::vector<float> actions;

    for (size_t i = 0; i < env::hunterCount; i++) {
        auto action = std::static_pointer_cast<agent::Agent>(env::robots[i])
                          ->getAction(states[i]);

        actions.push_back(action);
    }

    return actions;
}

void maddpg::update(int batchSize) {
    auto [obsBatch, indivActionBatch, indivRewardBatch, nextObsBatch,
          globalStateBatch, globalNextStateBatch, globalActionsBatch] =
        replaybuffer::sample(batchSize);

    for (size_t i = 0; i < env::hunterCount; i++) {

        auto obsBatchI = obsBatch[i];
        auto indivActionBatchI = indivActionBatch[i];
        auto indivRewardBatchI = indivRewardBatch[i];
        auto nextObsBatchI = nextObsBatch[i];

        std::vector<torch::Tensor> nextGlobalActions;

        for (size_t j = 0; j < env::hunterCount; j++) {
            auto hunter =
                std::static_pointer_cast<agent::Agent>(env::robots[j]);
            auto arr = hunter->actor->forward(torch::vstack(nextObsBatchI));

            std::vector<float> indexes;
            for (int row = 0; row < arr.size(0); row++) {
                indexes.push_back(
                    static_cast<float>(hunter->actor->nextAction(arr[row])));
            }

            auto n = torch::tensor(indexes);
            nextGlobalActions.push_back(torch::stack(n, 0));
        }

        auto tmp = torch::cat(nextGlobalActions, 0)
                       .reshape({batchSize, env::hunterCount});

        std::static_pointer_cast<agent::Agent>(env::robots[i])
            ->update({indivRewardBatchI, obsBatchI, globalStateBatch,
                      globalActionsBatch, globalNextStateBatch, tmp});
        std::static_pointer_cast<agent::Agent>(env::robots[i])->updateTarget();
    }
}

void maddpg::run(int maxEpisodes, int maxSteps, int batchSize) {
    std::vector<float> rewards;

    for (int episode = 0; episode < maxEpisodes; episode++) {
        // std::cout << "Episode: " << episode << std::endl;
        auto states = env::reset();
        auto epReward = 0;

        int step = 0;
        for (; step < maxSteps; step++) {
            if (!robosim::envcontroller::isRunning()) {
                return;
            }

            auto actions = getActions(states);
            auto [nextStates, rewards, done] = env::step(actions);

            epReward =
                std::accumulate(rewards.begin(), rewards.end(), epReward);

            if (done || step == maxSteps - 1) {
                break;
            }
            if (env::mode == env::Mode::EVAL) {
                states = nextStates;
                // slow down evaluation a bit
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            } else {
                replaybuffer::push({states, actions, rewards, nextStates});

                states = nextStates;

                if (static_cast<int>(replaybuffer::buffer.size()) > batchSize &&
                    step % batchSize == 0) {
                    update(batchSize);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        std::cout << "Episode: " << episode << " | Step: " << step
                  << " | Average: "
                     " | Reward: "
                     " | "
                     "Average: "
                     " | Time "
                  << std::endl;
    }

    robosim::envcontroller::updateRunning();
}

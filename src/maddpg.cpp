#include "maddpg.h"
#include "env.h"
#include "models.h" // for Actor
#include "prey.h"
#include "replayBuffer.h"
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
#include <type_traits>

namespace agent {
class Agent;
}

std::vector<float> maddpg::getActions(std::vector<torch::Tensor> states) {
    std::vector<float> actions;

    for (size_t i = 0; i < env::hunterCount; i++) {
        auto action = std::static_pointer_cast<agent::Agent>(
                          robosim::envcontroller::robots[i])
                          ->getAction(states[i]);

        actions.push_back(action);
    }

    return actions;
}

void maddpg::update() {
    auto [obsBatch, indivActionBatch, indivRewardBatch, nextObsBatch,
          globalStateBatch, globalNextStateBatch, globalActionsBatch] =
        replaybuffer::sample();

    for (size_t i = 0; i < env::hunterCount; i++) {

        auto obsBatchI = obsBatch[i];
        auto indivActionBatchI = indivActionBatch[i];
        auto indivRewardBatchI = indivRewardBatch[i];
        auto nextObsBatchI = nextObsBatch[i];

        std::vector<torch::Tensor> nextGlobalActions;

        for (size_t j = 0; j < env::hunterCount; j++) {
            auto hunter = std::static_pointer_cast<agent::Agent>(
                robosim::envcontroller::robots[j]);
            auto arr = hunter->actor->forward(torch::vstack(nextObsBatchI));

            std::vector<float> indexes;
            for (int row = 0; row < arr.size(0); row++) {
                indexes.push_back(
                    static_cast<float>(hunter->actor->nextAction(arr[row])));
            }

            // std::cout << indexes.size() << std::endl;

            auto n = torch::tensor(indexes);
            nextGlobalActions.push_back(torch::stack(n, 0));
        }

        auto nextGlobalActionsTemp =
            torch::cat(nextGlobalActions, 0)
                .reshape({env::BATCH_SIZE, env::hunterCount});

        std::static_pointer_cast<agent::Agent>(
            robosim::envcontroller::robots[i])
            ->update({indivRewardBatchI, obsBatchI, globalStateBatch,
                      globalActionsBatch, globalNextStateBatch,
                      nextGlobalActionsTemp});
        std::static_pointer_cast<agent::Agent>(
            robosim::envcontroller::robots[i])
            ->updateTarget();
    }
}

void maddpg::run(int maxEpisodes, int maxSteps) {
    // std::vector<float> rewards;

    for (int episode = 0; episode < maxEpisodes; episode++) {
        // std::cout << "Episode: " << episode << std::endl;
        auto states = env::reset();
        auto epReward = 0.0f;

        int step = 0;
        for (; step < maxSteps; step++) {
            if (!robosim::envcontroller::isRunning())
                return;

            auto actions = getActions(states);
            auto [nextStates, rewards, done] = env::step(actions);

            // epReward =
            //     std::accumulate(rewards.begin(), rewards.end(), epReward);
            epReward = std::reduce(rewards.begin(), rewards.end(), epReward);

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

                if (static_cast<int>(replaybuffer::buffer.size()) >
                        env::BATCH_SIZE &&
                    step % env::BATCH_SIZE == 0) {
                    update();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        std::cout << "Episode: " << episode << " | Step: " << step
                  << " | Average: "
                     " | Reward: "
                  << epReward
                  << " | "
                     "Average: "
                     " | Time "
                  << std::endl;
    }

    robosim::envcontroller::updateRunning();
}

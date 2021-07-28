#include "env.h"
#include "agent.h"
#include "prey.h"
#include <ATen/core/TensorBody.h>
#include <RobotMonitor.h>
#include <algorithm>
#include <random>
#include <stddef.h>
#include <thread>
#include <type_traits>

robosim::envcontroller::MonitorVec env::robots;

enum env::Mode env::mode = Mode::TRAIN;

std::vector<torch::Tensor> env::reset() {

    std::vector<torch::Tensor> obs;

    for (size_t i = 0; i < env::hunterCount; i++) {
        do {
            auto randomX = getRandomPos();
            auto randomY = getRandomPos();
            robots[i]->setPose(randomX, randomY, 0);
        } while (isSamePosition(robots[i]));

        auto agent = std::static_pointer_cast<agent::Agent>(robots[i]);

        obs.push_back(agent->getObservation());
    }

    return obs;
}

std::tuple<std::vector<torch::Tensor>, std::vector<float>, bool>
env::step(std::vector<float> actions) {

    std::vector<float> rewards;

    std::vector<torch::Tensor> nextStates;

    std::vector<std::thread> threads;

    for (size_t i = 0; i < env::hunterCount; i++) {
        auto agent = std::static_pointer_cast<agent::Agent>(env::robots[i]);

        threads.push_back(std::thread(&agent::Agent::executeAction, agent,
                                      action::getActionFromIndex(actions[i])));
    }

    // block and allows robots to execute actions at the same time
    for (auto &th : threads) {
        if (th.joinable())
            th.join();
    }

    for (size_t i = 0; i < env::hunterCount; i++) {
        auto agent = std::static_pointer_cast<agent::Agent>(env::robots[i]);
        nextStates.push_back(agent->getObservation());
        auto reward = agent->getReward(action::getActionFromIndex(actions[i]));
        rewards.push_back(reward);
    }

    bool trapped = env::prey->isTrapped();

    return {nextStates, rewards, trapped};
}

bool env::isSamePosition(robosim::envcontroller::RobotPtr robot) {
    return std::any_of(env::robots.begin(), env::robots.end(),
                       [robot](robosim::envcontroller::RobotPtr &agent) {
                           return robot != agent &&
                                  agent->getX() == robot->getX() &&
                                  agent->getY() == robot->getY();
                       });
}

int env::getRandomPos() {
    auto max = (GRID_SIZE * 2 - 3);
    auto min = 3;

    std::mt19937 mt(std::random_device{}());
    std::uniform_int_distribution<int> random(min, max);
    auto rndPos = random(mt);
    rndPos += rndPos % 2 == 0 ? 1 : 0;
    return (rndPos * robosim::envcontroller::getCellWidth()) / 2;
}

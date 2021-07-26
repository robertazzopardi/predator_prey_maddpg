#include "env.h"
#include "agent.h"                // for Agent
#include <ATen/core/TensorBody.h> // for Tensor
#include <RobotMonitor.h>
#include <algorithm>   // for any_of, uniform_int_distribution
#include <random>      // for mt19937, random_device
#include <stddef.h>    // for size_t
#include <thread>      // for thread
#include <type_traits> // for remove_extent_t

robosim::robotmonitor::MonitorVec env::robots;

enum env::Mode env::mode = Mode::TRAIN;
// enum env::Mode env::mode = Mode::EVAL;

std::vector<torch::Tensor> env::reset() {

    std::vector<torch::Tensor> obs;

    // for (auto robot : robots) {
    for (size_t i = 0; i < robots.size() - 1; i++) {
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
    // env::step(std::vector<action::Action> actions) {
    std::vector<float> rewards;

    std::vector<torch::Tensor> nextStates;

    std::vector<std::thread> threads;
    // vthreads threads;

    for (size_t i = 0; i < env::robots.size() - 1; i++) {
        auto agent = std::static_pointer_cast<agent::Agent>(env::robots[i]);

        // threads.push_back(
        //     std::thread(&agent::Agent::executeAction, agent, actions[i]));
        threads.push_back(std::thread(&agent::Agent::executeAction, agent,
                                      action::getActionFromIndex(actions[i])));
    }

    // block and allows robots to execute actions at the same time
    for (auto &th : threads) {
        if (th.joinable())
            th.join();
    }

    for (size_t i = 0; i < env::robots.size() - 1; i++) {
        auto agent = std::static_pointer_cast<agent::Agent>(env::robots[i]);
        nextStates.push_back(agent->getObservation());
        rewards.push_back(agent->getReward());
    }

    bool trapped = env::prey->isTrapped();

    return {nextStates, rewards, trapped};
}

bool env::isSamePosition(robosim::robotmonitor::RobotPtr robot) {
    return std::any_of(env::robots.begin(), env::robots.end(),
                       [robot](robosim::robotmonitor::RobotPtr &agent) {
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

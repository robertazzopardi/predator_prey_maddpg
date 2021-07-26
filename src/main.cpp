#include "env.h"
#include "hunter.h"
#include "maddpg.h"
#include "prey.h"   // for Prey
#include <Colour.h> // for OFF_BLACK, OFF_RED
#include <EnvController.h>
#include <RobotMonitor.h>
#include <memory>   // for make_shared, shared_ptr
#include <stdlib.h> // for EXIT_SUCCESS
#include <thread>   // for thread

std::shared_ptr<prey::Prey> env::prey =
    std::make_shared<prey::Prey>(false, colour::OFF_RED);

int main(void) {
    env::robots = robosim::robotmonitor::getRobots<hunter::Hunter>(
        env::hunterCount, colour::OFF_BLACK);

    env::robots.push_back(env::prey);

    robosim::envcontroller::EnvController(env::robots, env::GRID_SIZE,
                                          env::GRID_SIZE, 50);

    // std::thread delayThread([]() {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    //     env::reset();
    // });

    std::thread th(maddpg::run, 500, 300, env::BATCH_SIZE);

    robosim::envcontroller::startSimulation();

    th.join();

    // delayThread.join();

    return EXIT_SUCCESS;
}

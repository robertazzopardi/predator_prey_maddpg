#include "env.h"
#include "hunter.h"
#include "maddpg.h"
#include "prey.h"
#include <Colour.h>
#include <EnvController.h>
#include <stdlib.h>
#include <thread>

int main(void) {
    robosim::envcontroller::makeRobots<hunter::Hunter>(env::hunterCount,
                                                       colour::OFF_BLACK);
    robosim::envcontroller::makeRobots<prey::Prey>(env::preyCount,
                                                   colour::OFF_RED);

    robosim::envcontroller::EnvController(env::GRID_SIZE, env::GRID_SIZE, 50);

    std::thread th(maddpg::run, 500, 300);

    robosim::envcontroller::startSimulation();

    th.join();

    return EXIT_SUCCESS;
}

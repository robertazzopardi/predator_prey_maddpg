#include "env.h"
#include "hunter.h"
#include "maddpg.h"
#include "prey.h"
#include <Colour.h>
#include <EnvController.h>
#include <stdlib.h>
#include <thread>

using namespace robosim::envcontroller;

int main(void) {
    makeRobots<hunter::Hunter>(env::hunterCount, colour::OFF_BLACK);
    makeRobots<prey::Prey>(env::preyCount, colour::OFF_RED);

    EnvController(env::GRID_SIZE, env::GRID_SIZE, 50);

    std::thread th(maddpg::run, 500, 300);

    startSimulation();

    th.join();

    return EXIT_SUCCESS;
}

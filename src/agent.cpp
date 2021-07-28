#include "agent.h"
#include "action.h"
#include "direction.h"
#include "env.h"
#include "models.h"
#include <Colour.h> // for Colour
#include <EnvController.h>
#include <algorithm>   // for any_of
#include <iostream>    // for operator<<, basic_ostream, endl, cout
#include <sys/cdefs.h> // for __unused
#include <type_traits> // for remove_extent_t

agent::Agent::Agent(bool verbose, colour::Colour colour)
    : robosim::robotmonitor::RobotMonitor(verbose, colour), MSELoss() {

    // actor = std::make_shared<models::actor::Actor>(10, 4);
    // targetActor = std::make_shared<models::actor::Actor>(10, 4);

    // critic = std::make_shared<models::critic::Critic>(10, 1);
    // targetCritic = std::make_shared<models::critic::Critic>(10, 1);

    auto obsDim = 10;
    auto actionDim = 4;

    auto criticInputDim = obsDim * env::hunterCount;
    auto actorInputDim = obsDim;

    critic =
        std::make_shared<models::critic::Critic>(criticInputDim, actionDim);
    targetCritic =
        std::make_shared<models::critic::Critic>(criticInputDim, actionDim);

    actor = std::make_shared<models::actor::Actor>(actorInputDim, actionDim);
    targetActor =
        std::make_shared<models::actor::Actor>(actorInputDim, actionDim);

    // for (size_t i = 0; i < critic->parameters().size(); i++) {
    //     targetCritic->parameters().data()[i].copy_(
    //         critic->parameters()[i].data());
    // }
    // for (size_t i = 0; i < actor->parameters().size(); i++) {
    //     targetActor->parameters().data()[i].copy_(
    //         actor->parameters()[i].data());
    // }

    nextAction = action::Action::NOTHING;
}

void agent::Agent::moveDirection(direction::Direction direction) {
    auto x = getX();
    auto y = getY();

    if (canMove()) {
        gx = direction.px(x);
        gy = direction.py(y);

        if (env::mode == env::Mode::EVAL) {
            travel();
            // setPose(direction.px(x), direction.py(y), getHeading());
        } else {
            setPose(direction.px(x), direction.py(y), getHeading());
        }
    }
}

void agent::Agent::executeAction(action::Action nextAction) {
    if (nextAction != action::Action::NOTHING) {
        switch (nextAction) {

        case action::Action::FORWARD:
            moveDirection(direction::Direction::fromDegree(getHeading()));
            break;

        case action::Action::LEFT:
            if (env::mode == env::Mode::EVAL) {
                rotate(-90);
            } else {
                setPose(getX(), getY(), getHeading() - 90);
            }
            break;

        case action::Action::RIGHT:
            if (env::mode == env::Mode::EVAL) {
                rotate(90);
            } else {
                setPose(getX(), getY(), getHeading() + 90);
            }
            break;

        default:
            break;
        }

        nextAction = action::Action::NOTHING;
    }
}

void agent::Agent::run(bool *running __unused) {
    std::cout << "Starting Robot: " << serialNumber << std::endl;
}

bool agent::Agent::canMove() {
    auto dir = direction::Direction::fromDegree(getHeading());

    auto x = dir.px(getX());
    auto y = dir.py(getY());

    if (std::any_of(env::robots.begin(), env::robots.end(),
                    [&](const robosim::robotmonitor::RobotPtr &r) {
                        return (r.get() != this) &&
                               (r->getX() == x && r->getY() == y);
                    })) {
        return false;
    }

    auto xOffset = env::getEnvSize() - robosim::envcontroller::getCellWidth();

    return (x < xOffset && x > robosim::envcontroller::getCellWidth()) &&
           (y < xOffset && y > robosim::envcontroller::getCellWidth());
}

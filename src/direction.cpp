#include "direction.h"
#include <EnvController.h>

direction::Direction::Direction(enum Dir dir) : dir(dir) {}

direction::Direction direction::Direction::fromDegree(int degree) {
    switch (degree % 360) {
    case 0:
        return Direction(Dir::DOWN);
    case 90:
    case -270:
        return Direction(Dir::RIGHT);
    case 180:
    case -180:
        return Direction(Dir::UP);
    case 270:
    case -90:
        return Direction(Dir::LEFT);
    default:
        return Direction(Dir::NONE);
    }
}

int direction::Direction::px(int x) {
    switch (dir) {
    case Dir::UP:
    case Dir::DOWN:
        return x;
    case Dir::LEFT:
        return x - robosim::envcontroller::getCellWidth();
    case Dir::RIGHT:
        return x + robosim::envcontroller::getCellWidth();
    default:
        return 0;
    }
}

int direction::Direction::py(int y) {
    switch (dir) {
    case Dir::UP:
        return y - robosim::envcontroller::getCellWidth();
    case Dir::DOWN:
        return y + robosim::envcontroller::getCellWidth();
    case Dir::LEFT:
    case Dir::RIGHT:
        return y;
    default:
        return 0;
    }
}

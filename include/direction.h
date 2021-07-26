
#ifndef __DIRECTION_H__
#define __DIRECTION_H__

namespace direction {

enum class Dir { UP, DOWN, LEFT, RIGHT, NONE };

struct Direction {
    enum Dir dir;

    Direction(enum Dir);

    static Direction fromDegree(int);

    int px(int);
    int py(int);
};

} // namespace direction

#endif // !__DIRECTION_H__

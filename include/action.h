
#ifndef __ACTION_H__
#define __ACTION_H__

#include <array>

namespace action {

enum class Action { FORWARD = 0, LEFT = 1, RIGHT = 2, NOTHING = 3 };

const static std::array<Action, 4> ActionVec = {Action::FORWARD, Action::LEFT,
                                                Action::RIGHT, Action::NOTHING};

static inline auto toString(Action ac) {
    switch (ac) {
    case Action::FORWARD:
        return "FORWARD";

    case Action::LEFT:
        return "LEFT";

    case Action::RIGHT:
        return "RIGHT";

    case Action::NOTHING:
        return "NOTHING";

    default:
        return "NULL";
    }
}

static inline float getActionIndexFloat(enum Action action) {
    auto it = std::find(ActionVec.begin(), ActionVec.end(), action);

    // If element was found
    if (it != ActionVec.end()) {
        return it - ActionVec.begin();
    } else {
        return -1.0f;
    }
}

inline Action getActionFromIndex(int index) { return ActionVec[index]; }

} // namespace action

#endif // !__ACTION_H__

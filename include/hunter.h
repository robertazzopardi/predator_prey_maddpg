
#ifndef __HUNTER_H__
#define __HUNTER_H__

// #include "action.h"
#include "agent.h"
#include <torch/optim/adam.h>

namespace colour {
struct Colour;
}

namespace at {
class Tensor;
}

namespace hunter {

class Hunter : public agent::Agent {
  private:
    torch::optim::Adam criticOptimiser;
    torch::optim::Adam actorOptimiser;

    static constexpr auto tau = 0.001f;
    static constexpr auto gamma = 0.99f;

  public:
    Hunter(bool, colour::Colour);

    // action::Action getAction(at::Tensor);
    float getAction(at::Tensor);

    at::Tensor getObservation();

    bool isAtGoal();

    float getReward(action::Action);

    void update(agent::UpdateData);

    void updateTarget();
};

} // namespace hunter

#endif // !__HUNTER_H__


#ifndef __HUNTER_H__
#define __HUNTER_H__

#include "action.h" // for Action
#include "agent.h"
#include <ATen/core/TensorBody.h> // for Tensor
#include <torch/optim/adam.h>

namespace colour {
struct Colour;
}

namespace hunter {

class Hunter : public agent::Agent {
  private:
    torch::optim::Adam criticOptimiser;
    torch::optim::Adam actorOptimiser;

    static constexpr auto tau = 0.001f;
    static constexpr auto gamma = 0.99f;

    std::uniform_real_distribution<float> distRand;

    std::uniform_int_distribution<int> randAction;

    float epsilon = 0.95;

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

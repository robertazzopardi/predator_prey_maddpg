
#ifndef __PREY_H__
#define __PREY_H__

#include "action.h"
#include "agent.h"
#include <ATen/core/TensorBody.h>

namespace colour {
struct Colour;
}

namespace prey {

class Prey : public agent::Agent {
  private:
    std::uniform_real_distribution<float> dist;

  public:
    Prey(bool, colour::Colour);

    // action::Action getAction(at::Tensor);
    float getAction(at::Tensor);

    at::Tensor getObservation();

    bool isTrapped();

    float getReward(action::Action);

    void update(agent::UpdateData);

    void updateTarget();
};

} // namespace prey

#endif // !__PREY_H__

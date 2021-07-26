
#ifndef __AGENT_H__
#define __AGENT_H__

// #include "action.h"
#include "direction.h"
#include <RobotMonitor.h>
#include <memory>
#include <torch/nn/modules/loss.h>
#include <tuple>
#include <vector>

namespace at {
class Tensor;
}

namespace colour {
struct Colour;
}

namespace action {
enum class Action;
}

namespace models {

namespace actor {
struct Actor;
}

namespace critic {
struct Critic;
}

} // namespace models

namespace agent {

using UpdateData = std::tuple<std::vector<float>, std::vector<at::Tensor>,
                              std::vector<at::Tensor>, std::vector<at::Tensor>,
                              std::vector<at::Tensor>, at::Tensor>;

class Agent : public robosim::robotmonitor::RobotMonitor {
  private:
    int gx;
    int gy;

    void moveDirection(direction::Direction);

  protected:
    torch::nn::MSELoss MSELoss;

    void run(bool *);

  public:
    action::Action nextAction;

    std::shared_ptr<models::actor::Actor> actor;
    std::shared_ptr<models::actor::Actor> targetActor;
    std::shared_ptr<models::critic::Critic> critic;
    std::shared_ptr<models::critic::Critic> targetCritic;

    Agent(bool, colour::Colour);

    bool canMove();

    void executeAction(action::Action);

    virtual float getReward() = 0;

    // virtual action::Action getAction(torch::Tensor) = 0;
    virtual float getAction(torch::Tensor) = 0;

    virtual torch::Tensor getObservation() = 0;

    virtual void update(UpdateData) = 0;

    virtual void updateTarget() = 0;
};

using AgentPtr = std::shared_ptr<Agent>;

} // namespace agent

#endif // !__AGENT_H__

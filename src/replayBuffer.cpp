#include "replayBuffer.h"
#include "env.h"
#include <ATen/Functions.h>                                   // for stack
#include <ATen/core/TensorBody.h>                             // for Tensor
#include <__tuple>                                            // for tuple_...
#include <algorithm>                                          // for sample
#include <iterator>                                           // for back_i...
#include <random>                                             // for random...
#include <torch/csrc/autograd/generated/variable_factories.h> // for tensor
#include <vector>

replaybuffer::ReplayBuffer::ReplayBuffer(int agentCount, int maxSize)
    : buffer() {
    this->agentCount = agentCount;
    this->maxSize = maxSize;
}

void replaybuffer::ReplayBuffer::push(replaybuffer::Experience experience) {
    buffer.push_back(experience);
}

replaybuffer::Sample replaybuffer::ReplayBuffer::sample(int batchSize) {
    std::vector<std::vector<torch::Tensor>> obsBatch(
        env::hunterCount, std::vector<torch::Tensor>());
    // std::vector<std::vector<action::Action>> indiviActionBatch(
    //     env::hunterCount, std::vector<action::Action>());
    std::vector<std::vector<float>> indiviActionBatch(env::hunterCount,
                                                      std::vector<float>());
    std::vector<std::vector<float>> indiviRewardBatch(env::hunterCount,
                                                      std::vector<float>());
    std::vector<std::vector<torch::Tensor>> nextObsBatch(
        env::hunterCount, std::vector<torch::Tensor>());

    std::vector<torch::Tensor> globalStateBatch;
    std::vector<torch::Tensor> globalNextStateBatch;
    std::vector<torch::Tensor> globalActionBatch;

    std::vector<Experience> batch;
    std::sample(buffer.begin(), buffer.end(), std::back_inserter(batch),
                batchSize, std::mt19937{std::random_device{}()});

    for (auto experience : batch) {
        auto [state, action, reward, nextState] = experience;

        for (int i = 0; i < env::hunterCount; i++) {
            obsBatch[i].push_back(state[i]);
            indiviActionBatch[i].push_back(action[i]);
            indiviRewardBatch[i].push_back(reward[i]);
            nextObsBatch[i].push_back(nextState[i]);
        }

        globalStateBatch.push_back(torch::cat(state));

        // std::vector<float> af;
        // std::transform(std::begin(action), std::end(action),
        //                std::back_inserter(af), action::getActionIndexFloat);
        // globalActionBatch.push_back(
        //     torch::from_blob(af.data(), {(long)af.size(), 1}));

        // std::cout << torch::tensor(action) << std::endl;
        globalActionBatch.push_back(torch::tensor(action));

        globalNextStateBatch.push_back(torch::cat(nextState));
    }

    // for (auto var : globalActionBatch) {
    //     std::cout << var << std::endl;
    // }

    return Sample(obsBatch, indiviActionBatch, indiviRewardBatch, nextObsBatch,
                  globalStateBatch, globalNextStateBatch, globalActionBatch);
}

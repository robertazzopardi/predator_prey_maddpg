#include "replayBuffer.h"
#include "env.h"
#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <__tuple>
#include <algorithm>
#include <iterator>
#include <random>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <vector>

std::vector<replaybuffer::Experience> replaybuffer::buffer;

void replaybuffer::push(replaybuffer::Experience experience) {
    buffer.push_back(experience);
}

replaybuffer::Sample replaybuffer::sample(int batchSize) {
    std::vector<std::vector<torch::Tensor>> obsBatch(
        env::hunterCount, std::vector<torch::Tensor>());
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
        globalActionBatch.push_back(torch::tensor(action));
        globalNextStateBatch.push_back(torch::cat(nextState));
    }

    return Sample(obsBatch, indiviActionBatch, indiviRewardBatch, nextObsBatch,
                  globalStateBatch, globalNextStateBatch, globalActionBatch);
}

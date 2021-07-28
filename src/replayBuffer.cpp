#include "replayBuffer.h"
#include "env.h"
#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <__tuple>
#include <algorithm>
#include <iterator>
#include <random>
#include <stddef.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <vector>

std::vector<replaybuffer::Experience> replaybuffer::buffer;

void replaybuffer::push(replaybuffer::Experience experience) {
    buffer.push_back(experience);
}

replaybuffer::Sample replaybuffer::sample() {
    // std::vector<std::vector<torch::Tensor>> obsBatch(
    //     env::hunterCount, std::vector<torch::Tensor>());
    std::array<std::vector<at::Tensor>, env::hunterCount> obsBatch;
    // std::vector<std::vector<float>> indiviActionBatch(env::hunterCount,
    //                                                   std::vector<float>());
    std::array<std::vector<float>, env::hunterCount> indiviActionBatch;
    // std::vector<std::vector<float>> indiviRewardBatch(env::hunterCount,
    //                                                   std::vector<float>());
    std::array<std::vector<float>, env::hunterCount> indiviRewardBatch;
    // std::vector<std::vector<at::Tensor>> nextObsBatch(
    //     env::hunterCount, std::vector<at::Tensor>());
    std::array<std::vector<at::Tensor>, env::hunterCount> nextObsBatch;

    // std::vector<at::Tensor> globalStateBatch;
    std::array<at::Tensor, env::BATCH_SIZE> globalStateBatch;
    // std::vector<at::Tensor> globalNextStateBatch;
    // std::vector<at::Tensor> globalActionBatch;
    std::array<at::Tensor, env::BATCH_SIZE> globalNextStateBatch;
    std::array<at::Tensor, env::BATCH_SIZE> globalActionBatch;

    std::vector<Experience> batch;
    std::sample(buffer.begin(), buffer.end(), std::back_inserter(batch),
                env::BATCH_SIZE, std::mt19937{std::random_device{}()});

    for (size_t index = 0; index < batch.size(); index++) {

        auto [state, action, reward, nextState] = batch[index];

        for (int i = 0; i < env::hunterCount; i++) {
            obsBatch[i].push_back(state[i]);
            indiviActionBatch[i].push_back(action[i]);
            indiviRewardBatch[i].push_back(reward[i]);
            nextObsBatch[i].push_back(nextState[i]);
        }

        // globalStateBatch.push_back(torch::cat(state));
        globalStateBatch[index] = at::cat(state);
        globalActionBatch[index] = torch::tensor(action);
        globalNextStateBatch[index] = at::cat(nextState);
    }

    return Sample(obsBatch, indiviActionBatch, indiviRewardBatch, nextObsBatch,
                  globalStateBatch, globalNextStateBatch, globalActionBatch);
}

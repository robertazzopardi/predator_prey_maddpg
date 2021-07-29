#include "models.h"
#include <ATen/Functions.h>       // for relu, cat
#include <ATen/TensorOperators.h> // for Tensor::operator[]
#include <ATen/core/TensorBody.h>
#include <c10/core/Scalar.h> // for Scalar
#include <stdlib.h>          // for rand, RAND_MAX
#include <torch/nn/pimpl.h>  // for ModuleHolder

namespace models::actor {

namespace {

double fRand(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int boltzmannDistribution(torch::Tensor output, int shape) {
    auto exp = output.exp();
    // double sum = exp.sum(shape).getDouble(0);
    auto sum = exp.sum(shape).item().toFloat();

    double picked = fRand(0.0, 1.0) * sum;

    for (int i = 0; i < exp.size(0); i++) {
        if (picked < exp[i].item().toFloat())
            return i;
        picked -= exp[i].item().toFloat();
    }

    return (int)output.size(0) - 1;
}

} // namespace

Actor::Actor(int obsDim, int actionDim) {
    fc1 = register_module("fc1", torch::nn::Linear(obsDim, 512));
    fc2 = register_module("fc2", torch::nn::Linear(512, 128));
    fc3 = register_module("fc3", torch::nn::Linear(128, actionDim));
}

torch::Tensor Actor::forward(torch::Tensor obs) {
    auto x = torch::relu(fc1(obs));
    x = torch::relu(fc2(x));
    x = torch::tanh(fc3(x));

    return x;
}

int Actor::nextAction(torch::Tensor output) {
    return boltzmannDistribution(output, 0);
}

} // namespace models::actor

namespace models::critic {

Critic::Critic(int obsDim, int actionDim) {
    fc1 = register_module("fc1", torch::nn::Linear(obsDim, 1024));
    fc2 = register_module("fc2", torch::nn::Linear(1024 + actionDim, 512));
    fc3 = register_module("fc3", torch::nn::Linear(512, 300));
    fc4 = register_module("fc4", torch::nn::Linear(300, 1));
}

torch::Tensor Critic::forward(torch::Tensor x, torch::Tensor a) {
    x = torch::relu(fc1(x));
    auto xaCat = torch::cat({x, a}, 1);
    auto xa = torch::relu(fc2(xaCat));
    xa = torch::relu(fc3(xa));
    auto qval = fc4(xa);
    return qval;
}

} // namespace models::critic

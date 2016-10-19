require 'nn'
require 'optim'

local class = require 'class'

DQN = class('DQN')

function DQN:__init(model, discount, optimConfig)
  self.model = model
  self.parameters, self.gradParameters = self.model:getParameters()

  self.criterion = nn.MSECriterion()

  self.optimConfig = optimConfig or {
    optimizer = "rmsprop",
    learningRate = 0.001,
  }

  self.targets = torch.Tensor() --A buffer for targets
  self.discount = discount or 0.99
end

function DQN:act(observation)
  local qvalues = self.model:forward(observation)
  local _, indices = torch.max(qvalues, 2)
  return indices
end

function DQN:create_targets(rewards, next_observations, mask)
  local values = torch.max(self.model:forward(next_observations), 2)
  return torch.cmul(mask, values) * self.discount + rewards
end

function DQN:update_params(observations, actions, qtargets)
  local feval = function(x)

    -- reset gradients
    self.gradParameters:zero()

    local qvalues = self.model:forward(observations)

    self.targets:resizeAs(qvalues)
    self.targets:copy(qvalues)

    -- A simple trick to avoid using a mask for other actions.
    -- Set targets only for taken actions, zero gradient for others.
    for i=1,actions:size(1) do
      self.targets[i][actions[i][1]] = qtargets[i]
    end

    local error = self.criterion:forward(qvalues, self.targets)
    local df_do = self.criterion:backward(qvalues, self.targets)

    self.model:backward(observations, df_do)

    return error, self.gradParameters
  end

  optim[self.optimConfig.optimizer](feval, self.parameters, self.optimConfig)
end

function DQN:update(observations, actions, rewards, next_observations, mask)
  local qtargets = self:create_targets(rewards, next_observations, mask)
  self:update_params(observations, actions, qtargets)
end

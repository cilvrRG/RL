require 'nn'
local class = require 'class'

ReplayBuffer = class('ReplayBuffer')

function ReplayBuffer:__init(maxSize)
  self.maxSize = maxSize
  self.currentIndex = 1
  self.replayBuffer = {}
  self.observations = torch.Tensor()
  self.actions = torch.LongTensor()
  self.rewards = torch.Tensor()
  self.next_observations = torch.Tensor()
  self.mask = torch.Tensor()
end

function ReplayBuffer:insert(observation, action, reward, done, next_observation)
  self.replayBuffer[self.currentIndex] = {
    observation = observation:clone(),
    action = action,
    next_observation = next_observation:clone(),
    reward = reward,
    done = done
  }

  self.currentIndex = self.currentIndex + 1
  if self.currentIndex > self.maxSize then
    self.currentIndex = 1
  end
end

function ReplayBuffer:sample(nSamples)
  local observationSize = self.replayBuffer[1].observation:size(1)
  self.observations:resize(nSamples, observationSize)
  self.actions:resize(nSamples, 1)
  self.rewards:resize(nSamples, 1)
  self.next_observations:resize(nSamples, observationSize)
  self.mask:resize(nSamples, 1)

  for i=1,nSamples do
    local index = torch.random(#self.replayBuffer)
    self.observations[i] = self.replayBuffer[index].observation
    self.actions[i] = self.replayBuffer[index].action
    self.rewards[i] = self.replayBuffer[index].reward
    self.next_observations[i] = self.replayBuffer[index].next_observation
    self.mask[i] = self.replayBuffer[index].done and 0 or 1
  end

  return self.observations, self.actions, self.rewards, self.next_observations, self.mask
end

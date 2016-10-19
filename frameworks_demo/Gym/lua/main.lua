require 'dqn'
require 'replay_buffer'
local utils = require 'utils'
local GymClient = require("gym_http_client")
local HttpClient = require("httpclient")

client = GymClient.new('http://127.0.0.1:5000')

-- Parameters
BATCH_SIZE = 32
NUM_EPISODES = 50000
MIN_SAMPLES = 2 * BATCH_SIZE -- Minimal number of samples for an update
EVAL_EPISODE = 10 -- Perform evaluation each EVAL_EPISODE number of episodes
NUM_STEPS = 200
NUM_EVALUATIONS = 10 -- Number of times we run evaluation
INITIAL_EPSILON = 0.9
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.999
BUFFER_SIZE = 100000

-- Set up environment
env_id = 'CartPole-v0'
instance_id = client:env_create(env_id)

ndim_action = client:env_action_space_info(instance_id).n
ndim_obs = client:env_observation_space_info(instance_id).shape[1]
epsilon = INITIAL_EPSILON

agent = DQN(utils.mlp({ndim_obs, 128, 128, ndim_action}))
replayBuffer = ReplayBuffer(BUFFER_SIZE)

for i = 1, NUM_EPISODES do
  -- Training
  observation = torch.Tensor(client:env_reset(instance_id))
  for j = 1, NUM_STEPS do
    -- Epsilon greedy exploration
    if torch.uniform() < epsilon then
      action = torch.random(2)
    else
      action = agent:act(observation:view(1, -1))[1][1]
    end

    -- Perform an action, observe next state and reward
    next_observation, reward, done, info = client:env_step(instance_id, action-1, true)
    next_observation = torch.Tensor(next_observation)

    -- Insert it to replay buffer
    replayBuffer:insert(observation, action, reward, done, next_observation)

    if done == true then
      break
    else
      observation = next_observation
    end

    if #replayBuffer.replayBuffer > MIN_SAMPLES then
      -- Decay epsilon
      epsilon = epsilon * EPSILON_DECAY
      epsilon = math.max(epsilon, MIN_EPSILON)

      -- Sample a batch of samples and then update
      observations, actions, rewards, next_observations, mask = replayBuffer:sample(BATCH_SIZE)
      agent:update(observations, actions, rewards, next_observations, mask)
    end
  end

  -- Evaluation
  if i % EVAL_EPISODE == 0 then
    local total_reward = 0
    for k = 1, NUM_EVALUATIONS do
      observation = torch.Tensor(client:env_reset(instance_id))

      for j = 1, NUM_STEPS do
        action = agent:act(observation:view(1, -1))[1][1]
        next_observation, reward, done, info = client:env_step(instance_id, action-1, true)
        next_observation = torch.Tensor(next_observation)
        total_reward = total_reward + reward

        if done == true then
          break
        else
          observation = next_observation
        end
      end
    end
    print("Eval after episode #".. i .. ", average reward: " .. total_reward/NUM_EVALUATIONS)
  end
end

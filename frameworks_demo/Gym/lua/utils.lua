require 'layers'

local utils = {}

function utils.mlp(dims)
  local model = nn.Sequential()

  for i=1,#dims-2 do
    model:add(nn.Linear(dims[i], dims[i+1]))
    model:add(nn.LayerNormalization(dims[i+1]))
    model:add(nn.ReLU())
  end

  model:add(nn.Linear(dims[#dims-1], dims[#dims]))
  return model
end

return utils

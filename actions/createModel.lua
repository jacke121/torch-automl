local classic = require 'classic'
require 'classic.torch'
require 'nn'
local log = require "lib.log"

local createModel = classic.class("createModel")

function createModel:_init()

end


function createModel:execute(mljob)

  local criterion = nn.MSECriterion()

  if mljob.loadModelFromDisk == true then
      mljob.criterion = criterion
      return mljob
  end
  
  local n_inputs = mljob.data.train.input:size(2)
  local HUs = math.ceil(mljob.data.train.input:size(2) *  .75)
  local n_outputs = 1

  p = mljob.dropoutRate

  model = nn.Sequential()   


  -- model:add(nn.Dropout(p))
  model:add(nn.Linear(n_inputs, HUs)) 
  model:add(nn.ReLU())
  -- model:add(nn.Dropout(p))
  model:add(nn.Linear(HUs, HUs)) 
  model:add(nn.Sigmoid())
  model:add(nn.Linear(HUs, n_outputs))

  mljob.model = model
  mljob.criterion = criterion

  return mljob
end	

return createModel
require 'totem'
local tests = totem.TestSuite()
local tester = totem.Tester()
local dataStore = require 'core.dataStore'
local job = require 'core.job'
local dispatcher = require 'core.dispatcher'

local mljob1 = job{
                  modelName = 'boston-housing',
                  inputFile = 'boston.csv',
                  targetColumn = 'medv',
                  bayesOptEnabled = true,
                  budget = 100,
                  loadTrainedModelAndProcessedData = false,
                  percentageSplit = 90,
                  printLossIterations = 1000,
                  testModelIterations = 10,
                  batchSize = 100,
                  epochs = 1200,
                  optimization = 'sgd'
}

function tests.startDispatcher()
  torch.manualSeed(0)

  -- Push a new job to the redis queue.
  ds = dataStore{}
  ds:push(mljob1:toJSON())

  -- Start the dispatcher to process new jobs.
  d = dispatcher{}
  d:start()

end

return tester:add(tests):run()
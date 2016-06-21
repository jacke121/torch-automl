require 'totem'
local tests = totem.TestSuite()
local tester = totem.Tester()
local job = require 'core.job'

-- TODO: Modify json string as Job object has changed alot.
local json = [[{"dropoutRate":0.1,"modelName":"test-model","data":{"train":[],"test":[],"targetColumn":"medv","percentageSplit":90},"_class":{"_classAttributes":[],"_name":"job","_methods":[],"_finalMethods":[],"static":[],"_requiredMethods":[]},"train":{"enabled":true,"batchSize":100,"completed":false,"inProgress":false,"optimization":"sgd","testLosses":[],"trainLosses":[],"epochs":1000},"actions":[{"_class":{"_classAttributes":[],"_name":"csvToH5","_methods":[],"_finalMethods":[],"static":[],"_requiredMethods":[]}},{"_class":{"_classAttributes":[],"_name":"preprocessing","_methods":[],"_finalMethods":[],"static":[],"_requiredMethods":[]}},{"_class":{"_classAttributes":[],"_name":"createModel","_methods":[],"_finalMethods":[],"static":[],"_requiredMethods":[]}},{"counter":0,"_class":{"_classAttributes":[],"_name":"bayesOptim","_methods":[],"_finalMethods":[],"static":[],"_requiredMethods":[]}},{"counter":0,"_class":{"_classAttributes":[],"_name":"trainRegression","_methods":[],"_finalMethods":[],"static":[],"_requiredMethods":[]}}],"hyperparams":{"learningRate":0.1},"id":1,"paths":{"trainedModel":"trained-models/1.t7","h5":"/Users/ronanmoynihan/dev/hyperk/ml/test-data/boston.h5","csv":"/Users/ronanmoynihan/dev/hyperk/ml/test-data/boston.csv","s3":"Boston.csv","h5DataFolder":"data"},"bayesOpt":{"inProgress":false,"completed":false,"enabled":true,"sampleSize":0.2}}]]

function tests.toJSON()
local mljob = job{
                  id = 1,
                  modelName = 'test-model',
                  targetColumn = 'medv',
                  bayesOptEnabled = true,
                  trainEnabled = true,
                  learningRate = .1,
                  dropoutRate = 0.10,
                  percentageSplit = 90,
                  s3 = "Boston.csv",
                  batchSize = 100,
                  epochs = 1000,
                  optimization = 'sgd'
                }

  o_json = mljob:toJSON()
  tester:asserteq(o_json, json)
end

function tests.fromJSON()
  mljob = job{}
  mljob.modelName = 'test-model'

  m = mljob:loadFromJSON(json)
  print(m:toJSON())

  tester:asserteq(m.modelName, 'test-model')
  tester:asserteq(m.bayesOpt.enabled, true)
end

return tester:add(tests):run()
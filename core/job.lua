local classic = require 'classic'
require 'classic.torch'
local j = require 'json'

local s3Download = require 'actions.s3Download'
local mkOutputDir = require 'actions.mkOutputDir'
local preprocessing = require 'actions.preprocessing'
local csvToH5 = require 'actions.csvToH5'
local createModel = require 'actions.createModel'
local trainRegression = require 'actions.trainRegression'
local bayesOptim = require 'actions.bayesOptim'
local saveToT7 = require 'actions.saveToT7'
local loadFromT7 = require 'actions.loadFromT7'
local predict = require 'actions.predictAction'
require 'config.config'

local job = classic.class("job")

function job:_init(opts)

  self.dataFile = opts.dataFile
  self.modelName = opts.modelName
  self.loadTrainedModelAndProcessedData = opts.loadTrainedModelAndProcessedData
  self.normalise = opts.normalise or true
  self.predictOnly = opts.predictOnly or false

  self.model = nil
  self.criterion = nil
  self.dropoutRate = opts.dropoutRate

  self.bayesOpt = {}
  self.bayesOpt.budget = opts.budget
  self.bayesOpt.sampleSize = .10 
  self.bayesOpt.enabled = opts.bayesOptEnabled
  self.bayesOpt.completed = false
  self.bayesOpt.inProgress = false 

  self.train = {}
  self.train.trainLosses = {}
  self.train.testLosses = {}
  self.train.avgLoss = nil
  self.train.optimization = opts.optimization
  self.train.batchSize = opts.batchSize
  self.train.epochs = opts.epochs
  self.train.enabled = not opts.predictOnly
  self.train.completed = false
  self.train.inProgress = false 
  self.train.testSizePercentage = opts.testSizePercentage
  self.train.printLossIterations = opts.printLossIterations --  100 -- mljob.data.train.input:size(1) / 5 -- mljob.data.train.input:size(2)
  self.train.testModelIterations = opts.testModelIterations 

  self.hyperparams = {}
  self.hyperparams.learningRate = opts.learningRate
  self.hyperparams.learningRateDecay = opts.learningRateDecay
  self.hyperparams.weightDecay = opts.weightDecay
  self.hyperparams.momentum = opts.momentum
  self.hyperparams.dropout = opts.dropout

  self.data = {}
  self.data.percentageSplit = opts.percentageSplit
  self.data.raw = nil
  self.data.normalised = nil
  self.data.train = {}
  self.data.train.input = nil
  self.data.train.target = nil
  self.data.test = {}
  self.data.test.input = nil
  self.data.test.target = nil
  self.data.columns = nil
  self.data.targetColumn = opts.targetColumn
  self.data.mean = nil
  self.data.std = nil

  self.paths = {}
  self.paths.h5InputDataFileExists = opts.h5InputDataFileExists
  self.paths.s3 = opts.s3
  self.paths.h5DataFolder = "data"
  if opts.modelName~=nil then
    self.outputDir = 'output/' .. self.modelName
    self.paths.csv = config.paths.inputData .. opts.inputFile
    self.paths.h5 = self.outputDir .. "/data.h5"
    self.paths.trainedModel = self.outputDir ..  '/model.t7'
  end  

	self.actions = {
      mkOutputDir{},
      -- s3Download{},
      csvToH5{},
      loadFromT7{},
      preprocessing{shuffle=true},
      saveToT7{},
      createModel{},
      bayesOptim{},
      trainRegression{},
      predict{}
    }
end

function job:toJSON()
		return j.encode(self)
end

function job:loadFromJSON(str)
	o = j.decode(str)
	o.actions = self.actions
	o.toJSON = self.toJSON						 
	return o
end

return job
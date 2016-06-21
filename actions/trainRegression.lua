local classic = require 'classic'
require 'classic.torch'
require 'nn'
require 'optim'
require 'math'
local _ = require ('lib.moses')
local log = require "lib.log"
local dataStore = require 'core.dataStore'
local regression = require 'train.regression'

local trainRegression = classic.class("trainRegression")

function trainRegression:_init()
	self.counter = 0
	ds = dataStore{} 
	trainer = regression{}
end


function trainRegression:execute(mljob)

    if mljob.train.enabled == true then
        isBayesOpt = false
        mljob = trainer:train(mljob, isBayesOpt)
    end

	return mljob
end

return trainRegression
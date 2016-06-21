local classic = require 'classic'
require 'classic.torch'
require 'nn'
require 'optim'
require 'math'
local _ = require ('lib.moses')
local log = require "lib.log"
local dataStore = require 'core.dataStore'
local bo = require 'bayesoptim.regressionBayesOptim'

local bayesOptim = classic.class("bayesOptim")

function bayesOptim:_init()
	self.counter = 0
	ds = dataStore{} 

end


function bayesOptim:execute(mljob)

	if mljob.bayesOpt.enabled then
		hyperOpt = bo{mljob = mljob}
		mljob = hyperOpt:run(mljob)
	end	
	
	return mljob
end

return bayesOptim
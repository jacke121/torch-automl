local classic = require 'classic'
require 'classic.torch'
-- local regressionBayesOptim = require 'bayesoptim.regressionBayesOptim'
local log = require "lib.log"
local pipeline = require "core.pipeline"

local worker = classic.class("worker")

function worker:_init(opts)

end	

function worker:doNext(mljob)

	log.info("running job " .. mljob.modelName)

	p = pipeline{ mljob = mljob }
	p:run()
    
end

return worker
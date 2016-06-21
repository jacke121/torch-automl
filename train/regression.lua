local classic = require 'classic'
require 'classic.torch'
require 'nn'
require 'optim'
require 'math'
local _ = require ('lib.moses')
local log = require "lib.log"
local dataStore = require 'core.dataStore'

local regression = classic.class("regression")

function regression:_init(opts)
	_mljob = opts.mljob
	self.counter = 0
	ds = dataStore{} 
end

function regression:train(mljob,isBayesOpt)

	mljob.train.trainLosses = nil
	mljob.train.testLosses = nil

	------------------------------------------------------------------------------
	-- OPTIONS
	------------------------------------------------------------------------------

	local opt = {}

	opt.print_training_loss = mljob.train.printLossIterations
	opt.test_model_iteration = mljob.train.testModelIterations

	-- NOTE: the code below changes the optimization algorithm used, and its settings
	local optimState       -- stores a lua table with the optimization algorithm's settings, and state during iterations
	local optimMethod      -- stores a function corresponding to the optimization routine	

	optimState = {
	  learningRate = mljob.hyperparams.learningRate,
	  learningRateDecay = mljob.hyperparams.learningRateDecay
              ,
	   momentum = mljob.hyperparams.momentum,
	   weightDecay = mljob.hyperparams.weightDecay
	}
	opt.batch_size = mljob.train.batchSize
	opt.epochs = mljob.train.epochs
	if mljob.train.optimization=='sgd' then
		optimMethod = optim.sgd
	end	
		if mljob.train.optimization=='adagrad' then
		optimMethod = optim.adagrad
	end	

	-- weight initialisation
    local method = 'xavier_caffe'
    mljob.model = require('train.weight-init')(mljob.model, method)

	local model, training_losses, test_losses, avgLoss = self:runtrain(mljob,opt,optimMethod,optimState,isBayesOpt)
	
	mljob.train.avgLoss = avgLoss

	mljob.train.trainLosses = training_losses
	mljob.train.testLosses = test_losses

	log.info("Lowest Train MSE: " .. tostring(_.min(training_losses)))

	log.info("Lowest Test MSE: " .. tostring(_.min(test_losses)))  

	return mljob
end	


function regression:runtrain(mljob,opt,optimMethod,optimState,isBayesOpt)

	model = mljob.model
	data = mljob.data
	criterion = mljob.criterion

	------------------------------------------------------------------------
	-- create model and loss/grad evaluation function
	--
	local ninputs = data.train.input:size(2)
	local n_train_data = data.train.input:size(1) -- number of training data

	if isBayesOpt then
        if n_train_data > 10000 then
		    n_train_data = math.ceil(n_train_data * mljob.bayesOpt.sampleSize)
        end
	end

	x, dl_dx = model:getParameters()

	-- Closure
	function feval(x_new)

	   -- set x to x_new, if differnt
	   -- (in this simple example, x_new will typically always point to x,
	   -- so the copy is really useless)
	   if x ~= x_new then
	      x:copy(x_new)
	   end

	  local batch_inputs, batch_targets = getNextBatch(self,opt,data,n_train_data)

	   -- reset gradients (gradients are always accumulated, to accomodate 
	   -- batch methods)
	   dl_dx:zero()

	   -- evaluate the loss function and its derivative wrt x, for that sample
	   local loss_x = criterion:forward(model:forward(batch_inputs), batch_targets)
	   model:backward(batch_inputs, criterion:backward(model.output, batch_targets))

	   -- return loss(x) and dloss/dx
	   return loss_x, dl_dx
	end


	local losses = {}          -- training losses for each iteration/minibatch
	local test_losses = {}
	local total_loss = 0
	local epochs = opt.epochs  -- number of full passes over all the training data
	local iterations = epochs * math.ceil(n_train_data / opt.batch_size) -- integer number of minibatches to process

	log.info(math.ceil(n_train_data / opt.batch_size))
	log.info(iterations)
	-- an epoch is a full loop over our training data
	for i = 1,iterations do

	    -- optim contains several optimization algorithms. 
	    -- All of these algorithms assume the same parameters:
	    --   + a closure that computes the loss, and its gradient wrt to x, 
	    --     given a point x
	    --   + a point x
	    --   + some parameters, which are algorithm-specific
	    local _,minibatch_loss = optimMethod(feval,x,optimState)
	    -- Functions in optim all return two things:
	    --   + the new x, found by the optimization method (here SGD)
	    --   + the value of the loss functions at all points that were used by
	    --     the algorithm. SGD only estimates the function once, so
	    --     that list just contains one value.

	    losses[#losses + 1] = minibatch_loss[1] -- append the new loss
	    total_loss = total_loss +  minibatch_loss[1]
	    avgLoss = total_loss / i

	    if i % opt.print_training_loss == 0 then
				
				log.info(mljob.modelName .. ' -> '.. tostring(i) ..  ' / ' .. tostring(iterations) .. ' : ' .. 'avg loss = ' .. avgLoss)

                -- Uncomment to save job back to redis.
                --ds:saveJob(mljob)
			end	

			if isBayesOpt~=true then
		    if i % opt.test_model_iteration == 0 then
		   		local test_outputs = model:forward(data.test.input)
		    	local test_loss = criterion:forward(test_outputs, data.test.target)
		    	if (lowest_test_loss == nil or test_loss < lowest_test_loss) then
		    		lowest_test_loss = test_loss
			    	test_losses[#test_losses + 1] = test_loss 
			    	torch.save(mljob.paths.trainedModel,model)
			    end	
		 		end
		 	end	

	end  

	return model, losses, test_losses, avgLoss
end

function getNextBatch(self,opt,data,n_train_data)

		-- get start/end indices for our minibatch (in this code we'll call a minibatch a "batch")
	  --           ------- 
	  --          |  ...  |
	  --        ^ ---------<- start index = i * batchsize + 1
	  --  batch | |       |
	  --   size | | batch |       
	  --        v |   i   |<- end index (inclusive) = start index + batchsize
	  --          ---------                         = (i + 1) * batchsize + 1
	  --          |  ...  |                 (except possibly for the last minibatch, we can't 
	  --          --------                   let that one go past the end of the data, so we take a min())
	  local start_index = self.counter * opt.batch_size + 1
	  local end_index = math.min(n_train_data, (self.counter + 1) * opt.batch_size + 1)
	  if end_index == n_train_data then
	    self.counter = 0
	  else
	    self.counter = self.counter + 1
	  end

	  local batch_inputs = data.train.input[{{start_index, end_index}, {}}]
	  local batch_targets = data.train.target[{{start_index, end_index}}]

	  return batch_inputs, batch_targets
end

return regression
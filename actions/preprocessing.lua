local classic = require 'classic'
require 'classic.torch'
local log = require "lib.log"
local _ = require ('lib.moses')

local preprocessing = classic.class("preprocessing")


function preprocessing:_init(opts)
	shuffle = opts.shuffle
end	

function preprocessing:execute(mljob)

    if mljob.loadTrainedModelAndProcessedData ~= true then

        if mljob.predictOnly == false then
            if shuffle then
                mljob = shuffleData(mljob)
            end
        end

        mljob = splitInputAndTarget(mljob)

        if mljob.normalise then
            mljob = standardize(mljob)
        end

        if mljob.predictOnly == false then
            mljob = splitTrainTest(mljob)
        end

        mljob.data.raw = nil

    end

	return mljob

end	

function shuffleData(mljob)
	shuffle = torch.randperm(mljob.data.raw:size(1)):long()
	mljob.data.raw = mljob.data.raw:index(1,shuffle)

	return mljob
end	

function splitInputAndTarget(mljob)
	targetIndex = _.detect(mljob.data.columns,mljob.data.targetColumn)
	target = mljob.data.raw:select(2,targetIndex)
	
	allColumnIndexes = torch.range(1, mljob.data.raw:size(2)):totable()
	inputColumnIndexes = _.reject(allColumnIndexes, function(i,value) 
	  return value== targetIndex
	end)

	mljob.data.normalised = mljob.data.raw:index(2,torch.LongTensor(inputColumnIndexes))

	return mljob
end	


--[[

>>> X -= np.mean(X, axis = 0) # zero-center
>>> X /= np.std(X, axis = 0) # normalize

--]]
function standardize(mljob)

  if mljob.predictOnly == false then
    mean = mean or mljob.data.normalised:mean(1)
    std = std or mljob.data.normalised:std()

    mljob.data.std = std
    mljob.data.mean = mean
  else
      mean = mljob.data.mean
      std = mljob.data.std
  end

  mljob.data.normalised:add(-mean:expandAs(mljob.data.normalised)):mul(1/std)

  return mljob
end

function splitTrainTest(mljob)
	  -- Split into Train & Test.
	train_test_split_index = (mljob.data.raw:size(1) / 100) * mljob.data.percentageSplit
  
  splitInputs = mljob.data.normalised:split(train_test_split_index,1)
  mljob.data.train.input = splitInputs[1]
  mljob.data.test.input = splitInputs[2]
  
  splitTargets = target:split(train_test_split_index,1)
  mljob.data.train.target = splitTargets[1]
  mljob.data.test.target = splitTargets[2]

  return mljob
end

return preprocessing
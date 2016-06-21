s3 = require 's3'
require 'hdf5'
local classic = require 'classic'
require 'classic.torch'
local log = require "lib.log"

local loadFromT7 = classic.class("loadFromT7")

function loadFromT7:_init(opts)

end	

function loadFromT7:execute(mljob)

    if mljob.loadTrainedModelAndProcessedData or mljob.predictOnly then
        mljob.data = torch.load(mljob.outputDir .. "/data.t7")
        mljob.model = torch.load(mljob.outputDir .. "/model.t7")
        mljob.hyperparams = torch.load(mljob.outputDir .. "/hyperparams.t7")
    end
	return mljob
end	

return loadFromT7

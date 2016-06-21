s3 = require 's3'
require 'hdf5'
local classic = require 'classic'
require 'classic.torch'
local log = require "lib.log"

local saveToT7 = classic.class("saveToT7")

function saveToT7:_init(opts)

end	

function saveToT7:execute(mljob)

    torch.save(mljob.outputDir .. "/data.t7", mljob.data)

	return mljob
end	

return saveToT7

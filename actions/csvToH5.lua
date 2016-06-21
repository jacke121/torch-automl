local classic = require 'classic'
require 'classic.torch'
local log = require "lib.log"
require 'config.config'

local csvToH5 = classic.class("csvToH5")

function csvToH5:_init()

end


function csvToH5:execute(mljob)
    if mljob.loadTrainedModelAndProcessedData ~= true then
        -- Call python script to convert csv to h5 nd read columns.
        command  = config.pythonCommand .. ' ' .. mljob.paths.csv .. ' ' .. mljob.paths.h5 .. ' ' .. mljob.paths.h5DataFolder

        local file = io.popen(command, 'r')
        local output = file:read('*all')
        file:close()
        dynamicLuaCode = 'columnNames = {' .. output .. '}'
        loadstring(dynamicLuaCode)()

        -- Read the h5 file data into a tensor.
        local myFile = hdf5.open(mljob.paths.h5, 'r')
        data = myFile:read(mljob.paths.h5DataFolder):all()
        myFile:close()

        mljob.data.columns = columnNames
        mljob.data.raw = data

        log.info(data:size(1))
    end
	return mljob
end	

return csvToH5
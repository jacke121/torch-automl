local classic = require 'classic'
require 'classic.torch'
require 'nn'
require 'optim'
require 'math'
local _ = require ('lib.moses')
local log = require "lib.log"

local predictAction = classic.class("predictAction")

function predictAction:_init()

end


function predictAction:execute(mljob)

    log.info("starting predictions ...")

    -- numberOfTestPreds = mljob.data.test.input:size(1)
    numberOfTestPreds = 20

    final_test_outputs = mljob.model:forward(mljob.data.test.input[{{1,numberOfTestPreds},{}}])

    log.info('\n#   prediction     actual')
    for i = 1,numberOfTestPreds do
        log.info(string.format("%2d    %6.2f      %6.2f",
            i,
            final_test_outputs[i][1],
            mljob.data.test.target[i]
        )
        )
    end

	return mljob
end

return predictAction
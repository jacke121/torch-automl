local classic = require 'classic'
require 'classic.torch'
local log = require "lib.log"
local _ = require ('lib.moses')

local pipeline = classic.class("pipeline")

function pipeline:_init(opts)
	_mljob = opts.mljob
end

function pipeline:run()

	_.each(_mljob.actions, function(i, action)
        _mljob = action:execute(_mljob)
    end)
end

return pipeline
s3 = require 's3'
require 'hdf5'
local classic = require 'classic'
require 'classic.torch'
local log = require "lib.log"

local mkOutputDir = classic.class("mkOutputDir")

function mkOutputDir:_init(opts)

end	

function mkOutputDir:execute(mljob)

    paths.mkdir(mljob.outputDir)

	return mljob
end	

return mkOutputDir

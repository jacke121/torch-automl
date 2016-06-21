local classic = require 'classic'
require 'classic.torch'
local log = require "lib.log"
local async = require 'async'
local dataStore = require 'core.dataStore'
require 'config.config'

local dispatcher = classic.class("dispatcher")

function dispatcher:_init(opts)

end         

function dispatcher:start()
   log.info("starting dispatcher ...")

   -- Nb of jobs:
   local N = config.numParallelJobs

   async.run(function()
      for i = 1,N do
         local code = [[
            local dataStore = require 'core.dataStore'
            local job = require 'core.job'
            require 'config.config'

            local jobid = ${jobid}
            local njobs = ${njobs}

            ds = dataStore{} 

            keepRunning = true
            while keepRunning do

               -- remove the following line to keep application running and polling redis
               keepRunning = false

               local value = ds:pop("torch-auto-ml")

               if value then
                  local mljob = job{}
                  mljob = mljob:loadFromJSON(value)

                  local worker = require 'core.worker'
                  w = worker{}
                  w:doNext(mljob)
               end
            end      
         ]]

         async.process.dispatch(code, {jobid = i, njobs = N}, function(process)
            process.onexit(function(status, signal)
            end)
            process.stdout.ondata(function(data)
               io.write('[process:'..process.pid.. '] ' .. data)
               io.flush()
            end)
         end)
      end
   end)
end

return dispatcher
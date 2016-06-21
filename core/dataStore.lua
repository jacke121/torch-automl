local classic = require 'classic'
require 'classic.torch'
local redis = require 'redis'

local dataStore = classic.class("dataStore")

local params = {
   host = '127.0.0.1',
   port = 6379,
}

local client = redis.connect(params)

function dataStore:_init(opts)

end	

function dataStore:pop(mljob)
	return client:rpop("torch-auto-ml")
end

function dataStore:push(val)	
	client:rpush("torch-auto-ml",val)
end

function dataStore:len()
	return client:llen("torch-auto-ml")
end

function dataStore:saveJob(mljob)
	client:set('job:' .. mljob.modelName, mljob:toJSON())
end

return dataStore
s3 = require 's3'
require 'hdf5'
local classic = require 'classic'
require 'classic.torch'
local log = require "lib.log"

local s3Download = classic.class("s3Download")

-- lua-s3 & dependencies
-- luarocks install https://github.com/gcr/lua-s3
-- luarocks install luacrypto OPENSSL_DIR=/Users/ronanmoynihan/dev/opennssl/openssl-1.0.2d
-- luarocks install s3

local b = s3:connect{
  awsId="",
  awsKey="",
  bucket="",
}

function s3Download:_init(opts)

end	

function s3Download:execute(mljob)
	f = io.open(mljob.paths.csv, "w")
	print(b:get(mljob.paths.s3, ltn12.sink.file(f)))

	return mljob
end	

return s3Download

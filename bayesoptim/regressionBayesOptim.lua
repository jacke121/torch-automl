local classic = require 'classic'
require 'classic.torch'
local bot7  = require('bot7')
local hyperparam = bot7.hyperparam
local regression = require 'train.regression'
local _ = require ('lib.moses')

local regressionBayesOptim = classic.class("regressionBayesOptim")

function regressionBayesOptim:_init(opts)
  mljob = opts.mljob

  -- cmd:option('-bot',       'bo', 'specify which bot to use: {bo, rs}')
  -- cmd:option('-nInitial',   2, 'specify number of initial candidates to sample at random')
  -- cmd:option('-budget',     100,'specify budget (#nominees) for experiment')
  -- cmd:option('-noisy',      false, 'specify observations as noisy')
  -- cmd:option('-grid_size',  20000, 'specify size of candidate grid')
  -- cmd:option('-grid_type',  'random', 'specify type for candidate grid')
  -- cmd:option('-mins', '',   'specify minima for inputs (defaults to 0.0)')
  -- cmd:option('-maxes',      '', 'specify maxima for inputs (defaults to 1.0)')
  -- cmd:option('-score',      'ei', 'specify acquisition function to be used by bot; {ei, ucb}')

  r = regression{}

  opt = {}

  opt.xDim = 4
  opt.yDim = 1
  opt.bot = 'bo'
  opt.nInitial = 2
  opt.budget = mljob.bayesOpt.budget
  opt.noisy = false
  opt.grid_size = 20000
  opt.grid_type = 'random'
  opt.mins =  ''
  opt.maxes = ''
  opt.score = 'ei'

  ---------------- Experiment Configuration
  _expt = 
  {
    xDim   = opt.xDim,
    yDim   = opt.yDim,
    budget = opt.budget,
  }

  _expt.model = {noiseless = not opt.noisy} 
  _expt.grid  = {type = opt.grid_type, size = opt.grid_size}
  _expt.bot   = {type = opt.bot, nInitial = opt.nInitial}

  -- Establish feasible hyperparameter ranges
  if (opt.mins ~= '') then
    loadstring('expt.mins='..opt.mins)()
  else
    _expt.mins = torch.zeros(1, opt.xDim)
  end

  if (opt.maxes ~= '') then
    loadstring('opt.maxes='..opt.maxes)()
  else
    _expt.maxes = torch.ones(1, opt.xDim)
  end

  ---- Choose acquistion function
  _expt['score'] = {}
  if (opt.score == 'ei') then
    _expt.score['type'] = 'expected_improvement'
  elseif (opt.score == 'ucb') then
    _expt.score['type'] = 'confidence_bound'
  end

  ---- Set metatables
  for key, val in pairs(_expt) do
    if type(val) == 'table' then
      setmetatable(val, {__index = _expt})
    end
  end
end 

local regression_objective = function(X)
  
  if (X:dim() == 1 or X:size(1) == X:nElement()) then
      X = X:reshape(1, X:nElement())
    end
    -- assert(X:size(2) == 4)

    mljob.hyperparams.learningRate = X[1][1]
    mljob.hyperparams.learningRateDecay = X[1][2]
     mljob.hyperparams.momentum = X[1][3]
     mljob.hyperparams.weightDecay = X[1][4]

    isBayesOpt = true
    mljob = r:train(mljob,isBayesOpt)

   last10avg = (_.reduce(_.last(mljob.train.trainLosses,10),function(memo,v)
                    return memo+v
                end) ) / 10

  return last10avg

end

function regressionBayesOptim:run(mljob)
  local hypers = {}
  for k = 1, opt.xDim do
    hypers[k] = hyperparam('x'..k,0,1)
  end

  -------- Initialize bot
  bot = bot7.bots.bayesopt(regression_objective, hypers, _expt)

  -------- Perform experiment
  bot:run_experiment()
  
  mljob.hyperparams.learningRate = bot.best.x[1]
  mljob.hyperparams.learningRateDecay = bot.best.x[2]
  mljob.hyperparams.momentum = bot.best.x[3]
  mljob.hyperparams.weightDecay = bot.best.x[4]

  torch.save(mljob.outputDir .. '/hyperparams.t7', mljob.hyperparams)

  print(string.format('\nbot.best.y (Training MSE): %d', bot.best.y[1][1]))

  return mljob
end


return regressionBayesOptim
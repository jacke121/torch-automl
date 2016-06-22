# Torch-Pipeline
Experimental application to automate all aspects of a machine learning task.  
Data pre-processing, Bayesian Optimization of Hyperparameters, Training, Predicting.  

- Jobs are pushed to a redis queue as json.
- Torch dispatcher polls redis to dequeue and run jobs.
- The number of jobs that can run in parallel can be set in the configuration.
- Can be started as a server to continuously poll the queue.
- Supports csv training files > 1GB.   

Contains the following actions,

- Convert csv to H5 format and load data into a torch tensor.
- Load data from t7 format
- Pre-process data (normalize, shuffle, split into train & test)
- Create a Neural Network Regression model
- Run bayesian optimization to automatically find best hyperparams. Uses [bot7](https://github.com/j-wilson/bot7).
- Train a regression model
- Test model on test data.
- Action to download training file from s3 is not currently enabled and requires additional dependencies.

Currently supports regression but new actions can be added for other types of models.  

## Install
Tested on Mac OS X only.

### Dependencies
- Python  (Used for converting csv to H5 format)   
- [Torch](http://torch.ch/docs/getting-started.html#_)  
- [Redis](http://redis.io/download)

### Torch dependencies
```
  luarocks install redis-lua
  luarocks install async
  luarocks install json
  luarocks install classic
  luarocks install totem
  luarocks install moses
  
  # Gaussian Processes
  git clone https://github.com/j-wilson/gpTorch7
  cd gpTorch7
  luarocks make
  
  # Bot7
  git clone https://github.com/j-wilson/bot7
  cd bot7
  luarocks make
``` 

### Clone this repo
```
  git clone https://github.com/ronanmoynihan/torch-pipeline    
```

## Demo

- Start Redis Server in a new terminal window
```
    cd [path to redis folder]
    src/redis-server
```        

- Run Regression test in a new terminal window  
This will queue a new job which takes the boston housing data set as input, pre-processes the data,  
runs bayesian optimzation, trains a model and makes predictions on the test data set.
```
    cd [path to torch-pipeline folder]
    th tests/queue_train_jobs.lua
```

## Creating a new regression job. 
To queue a new job create a job object similar to tests/queue_train_jobs.lua and push it to the redis queue.   
The input file csv needs to be placed in the input folder of this project.
```
  local mljob1 = job{
    modelName = 'boston-housing',
    inputFile = 'boston.csv',
    targetColumn = 'medv',
    bayesOptEnabled = true,
    budget = 100,
    loadTrainedModelAndProcessedData = false,
    percentageSplit = 90,
    printLossIterations = 1000,
    testModelIterations = 10,
    batchSize = 100,
    epochs = 1200,
    optimization = 'sgd'
  }
  
  -- Push a new job to the redis queue.
  ds = dataStore{}
  ds:push(mljob1:toJSON())
  
  -- Start the dispatcher to process new jobs.
  d = dispatcher{}
  d:start()
```

Once the job is finished the model will be written to the output folder of this project.

## Configuration

- Number of jobs to run in parallel  
To change the number of jobs which can be run in parallel modify the following config setting.


```
  -- config/config.lua
  config.numParallelJobs = 4
```

- Start application as server to continuously poll redis queue  
To modify the dispatcher to continuously poll redis for new jobs modify remove the following line from core/dispatcher.lua

```
  -- remove the following line to keep application running and polling redis
  keepRunning = false
```     

**Warning: After stopping the application in terminal you may need to kill the process in Activity Monitor as it may keep running.**              

## Extending
You can create your own custom actions following the same format as the current ones in the actions folder.  

Once you create a custom action the job object (core/job.lua) needs to be modified to include the new action.

```
	self.actions = {
      mkOutputDir{},
      csvToH5{},
      loadFromT7{},
      preprocessing{shuffle=true},
      saveToT7{},
      createModel{},
      bayesOptim{},
      
      -- new custom action.
      trainClassifier{},
      
      predict{}
    }
```    
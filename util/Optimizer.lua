-- Authors: Tomas Kocisky
--
local Optimizer = torch.class('oxnn.Optimizer')

function Optimizer.optimize(dataset, w, dw, module, criterion, optim_func,
                            state, eval_datasets, update_err, get_err,
                            print_stats)
   print_stats = print_stats or Optimizer.PrintStats
   update_err = update_err or Optimizer.SimpleUpdateError
   get_err = get_err or Optimizer.SimpleGetError

   print('len(w)=' .. w:nElement())
   print('|w|=' .. w:norm())
   print('|dw|=' .. dw:norm())

   local iteration = 1
   state.iteration = iteration

   print('#  training')
   local time_training_start = torch.tic()

   while true do
      module:training()
      module:zeroGradParameters()

      local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
      if not true then
         for t = 1,dataset:size() do
            shuffledIndices[t] = t
         end
      end

      -- learning rate decay
      if params.Optimizer_lrDecayIter > 0 then
         assert(params.Optimizer_lrDecay > 0)
         assert(state.learningRate)
         if iteration > params.Optimizer_lrDecayIter then
            state.learningRate = state.learningRate / params.Optimizer_lrDecay
         end
      end

      local time_iteration_start = torch.tic()
      local eval_counter = 0
      local acconly = false
      local err = {}
      local last_dw_norm = nil
      for t = 1,dataset:size() do
         if params.maxiteration >= 0 and iteration > params.maxiteration then
            err = update_err(err, nil)
            break
         end

         local time_minibatch_start = torch.tic()
         if params.Optimizer_progress then xlua.progress(t, dataset:size()) end

         local example = dataset[shuffledIndices[t]]

         -- possibly collect garbage
         if params.Optimizer_collectgarbage > 0 and
            t % params.Optimizer_collectgarbage == 0 then
            if params.Optimizer_collectgarbage_cutorchsync and cutorch then
               cutorch.synchronize()
            end
            collectgarbage()
         end

         if not acconly then
            -- if we did not accumulate only previously
            module:zeroGradParameters()
         end

         acconly = t ~= dataset:size() and t % params.Optimizer_accumulate ~= 0

         local func = function(x)
            local input = example[1]
            local target = example[2]

            -- forward
            local fx = module:forward(input)
            if criterion then fx = criterion:forward(module.output, target) end
            module:backward(input,
                            criterion
                               and criterion:backward(module.output, target)
                               or 0)
            eval_counter = eval_counter + 1

            if not params.fast then
               assert(dw:norm() ~= 0, 'Gradient is zero!')
            end

            -- gradient normalization
            if not params.Optimizer_noGradNormalization then
               if input.perp_output_tokens then
                  dw:div(input.perp_output_tokens)
               else
                  dw:div(input.batch_size)
               end
            end

            if params.Optimizer_maxGradNorm > 0 then
               local norm = dw:norm()
               last_dw_norm = norm
               if norm > params.Optimizer_maxGradNorm then
                  dw:mul(params.Optimizer_maxGradNorm / norm)
               end
            end

            -- l2 regularization
            if params.Optimizer_l2 > 0 then
               dw:add(params.Optimizer_l2, x)
            end

            return fx,dw
         end

         local x,fx
         if acconly then
            x = w
            fx = {func(x)}
         else
            x,fx = optim_func(func, w, state)
         end

         err = update_err(err, {fx=fx, example=example})

         if params.verbose and
               (params.Optimizer_verbose == 0 or
                  t % torch.floor(dataset:size()*params.Optimizer_verbose/100)
                     == 1)
            then
            print_stats{train=true, verbose=true, err=get_err(err), err_t=err,
                        eval_counter=eval_counter, w=w, dw=dw, fx=fx,
                        iteration=iteration, example=example,
                        time_training_start=time_training_start,
                        time_iteration_start=time_iteration_start,
                        time_minibatch_start=time_minibatch_start,
                        t=t, t_max=dataset:size(), state=state,
                        dwnorm=last_dw_norm,
                       }
         end
         if package.loaded.cutorch then cutorch.synchronize() end
      end

      print_stats{train=true, verbose=false, err=get_err(err), err_t=err,
                  eval_counter=eval_counter, w=w, dw=dw, iteration=iteration,
                  time_training_start=time_training_start,
                  time_iteration_start=time_iteration_start,
                  state=state, dwnorm=last_dw_norm,
                 }

      if params.log and iteration % params.modelsavefreq == 0 then
         local tmp = module._module_clones
         module._module_clones = nil
         torch.save(oxnnmc.GetLogDir() .. '/model' .. iteration,
                   {params, w, nil, nil, state, iteration, eval_counter})
         module._module_clones = tmp
      end

      -- EVAL
      if iteration % params.modelevalfreq == 0 then
         for name, dataset in pairs(eval_datasets) do
            Optimizer.f(name, dataset, w, dw, module, criterion, update_err,
                        get_err, print_stats)
         end
      end

      iteration = iteration + 1
      state.iteration = iteration
      if params.maxiteration >= 0 and iteration > params.maxiteration then
          print('# you have reached the maximum number of iterations')
          break
      end
   end
end

function Optimizer.SimpleUpdateError(err, t)
   err = err or {}
   err.currentError = err.currentError or 0  -- cumulative error for this epoch
   err.err_count = err.err_count or 0  -- training examples processed
   if t == nil then return err end
   local fx = t.fx
   local example = t.example
   err.currentError = err.currentError + fx[#fx] * example[1].batch_size
   err.err_count = err.err_count + example[1].batch_size
   return err
end
function Optimizer.SimpleGetError(err)
   return err.currentError / err.err_count
end
function Optimizer.SimplePerplexityUpdateError(err, t)
   err = err or {}
   err.currentError = err.currentError or 0  -- cumulative error for this epoch
   err.err_count = err.err_count or 0  -- training examples processed
   if t == nil then return err end
   local fx = t.fx
   local example = t.example
   err.currentError = err.currentError + fx[#fx] * example[1].perp_output_tokens
   err.err_count = err.err_count + example[1].perp_output_tokens
   return err
end
function Optimizer.SimplePerplexityGetError(err)
   return err.currentError / err.err_count
end

function Optimizer.LoadModel(keep_params)
   keep_params = keep_params or {}
   local loadmodel = nil
   if #params.loadmodel > 0 then
      print('Loading model', params.loadmodel)
      loadmodel = torch.load(params.loadmodel)
      oxnnmc.LoadParams(loadmodel[1], keep_params)
   end
   return loadmodel
end

function Optimizer.f(name, dataset, w, dw, module, criterion, update_err,
                     get_err, print_stats)
   module:evaluate()
   local err = {}
   local eval_counter = 0
   local time_training_start = torch.tic()
   local time_iteration_start = torch.tic()
   for t = 1,dataset:size() do
      local time_minibatch_start = torch.tic()
      if params.Optimizer_progress then xlua.progress(t, dataset:size()) end
      local example = dataset[t]

      -- possibly collect garbage
      if params.Optimizer_collectgarbage > 0 and
         t % params.Optimizer_collectgarbage == 0 then
         if params.Optimizer_collectgarbage_cutorchsync and cutorch then
            cutorch.synchronize()
         end
         collectgarbage()
      end

      local func = function(x)
         module:zeroGradParameters()
         local input = example[1]
         local target = example[2]

         local fx = module:forward(input)
         if criterion then fx = criterion:forward(module.output, target) end
         eval_counter = eval_counter + 1

         return fx,dw
      end

      local fx = {({func(w)})[1]}

      err = update_err(err, {fx=fx, example=example})

      if params.verbose and
            (params.Optimizer_verbose == 0 or
               t % torch.floor(dataset:size()*params.Optimizer_verbose/100)
                  == 1)
         then
         print_stats{train=false, verbose=true, err=get_err(err), err_t=err,
                     eval_counter=eval_counter, w=w, dw=dw, fx=fx,
                     example=example, name=name,
                     time_training_start=time_training_start,
                     time_iteration_start=time_iteration_start,
                     time_minibatch_start=time_minibatch_start,
                     t=t, t_max=dataset:size(),
                    }
      end

      if package.loaded.cutorch then cutorch.synchronize() end
   end

   print_stats{train=false, verbose=false, err=get_err(err), err_t=err,
               eval_counter=eval_counter, w=w, dw=dw, name=name,
               time_training_start=time_training_start,
               time_iteration_start=time_iteration_start,
              }
end

function Optimizer.AddCmdOptions(cmd)
   cmd:option('-maxiteration', 5, '')
   cmd:option('-adadelta', false, '')
   cmd:option('-adagrad', false, '')
   cmd:option('-lbfgs', false, '')
   cmd:option('-sgd', false, '')
   cmd:option('-origsgd', false, '')
   cmd:option('-modelsavefreq', 1, '')
   cmd:option('-modelevalfreq', 1, '')
   cmd:option('-loadmodel', '', 'Path to a model to load.')
   cmd:option('-batch', 20, 'Mini-batch size.')
   cmd:option('-batchtest', 0, 'Mini-batch size for testing.')
   cmd:option('-Optimizer_verbose', 0, 'Show output every given number percent of epoch.')
   cmd:option('-Optimizer_progress', false, 'Show progress bar.')
   cmd:option('-Optimizer_collectgarbage', 0, 'Collect garbage every n minibatches.')
   cmd:option('-Optimizer_collectgarbage_cutorchsync', false, 'Synchronize when collecting garbage.')
   cmd:option('-Optimizer_accumulate', 1, '')
   cmd:option('-Optimizer_l2', 0, '')
   cmd:option('-Optimizer_lrDecay', 0, 'Divisor for the state.learningRate.')
   cmd:option('-Optimizer_lrDecayIter', 0, 'Decays the state.learningRate every n iterations.')
   cmd:option('-Optimizer_maxGradNorm', 0, 'Enforces maximum gradient norm.')
   cmd:option('-Optimizer_noGradNormalization', false, '')
   cmd:option('-adagrad_learningRate', -666, 'adagrad_learningRate', 'number')
   cmd:option('-adagrad_learningRateDecay', -666, 'adagrad_learningRateDecay',
              'number')
   cmd:option('-adam_beta1', -666, 'adam_beta1', 'number')
   cmd:option('-adam_beta2', -666, 'adam_beta2', 'number')
   cmd:option('-adam_epsilon', -666, 'adam_epsilon', 'number')
   cmd:option('-adam_lambda', -666, 'adam_lambda', 'number')
   cmd:option('-adam_learningRate', -666, 'adam_learningRate', 'number')
   cmd:option('-lbfgs_learningRate', -666, 'lbfgs_learningRate', 'number')
   cmd:option('-lbfgs_maxEval', -666, 'lbfgs_maxEval', 'number')
   cmd:option('-lbfgs_maxIter', -666, 'lbfgs_maxIter', 'number')
   cmd:option('-lbfgs_nCorrection', -666, 'lbfgs_nCorrection', 'number')
   cmd:option('-lbfgs_tolFun', -666, 'lbfgs_tolFun', 'number')
   cmd:option('-lbfgs_tolX', -666, 'lbfgs_tolX', 'number')
   cmd:option('-lbfgs_verbose', false, 'lbfgs_verbose', 'boolean')
   cmd:option('-origsgd_learningRate', -666, 'origsgd_learningRate', 'number')
   cmd:option('-sgd_dampening', -666, 'sgd_dampening', 'number')
   cmd:option('-sgd_learningRate', -666, 'sgd_learningRate', 'number')
   cmd:option('-sgd_learningRateDecay', -666, 'sgd_learningRateDecay', 'number')
   cmd:option('-sgd_momentum', -666, 'sgd_momentum', 'number')
   cmd:option('-sgd_nesterov', false, 'sgd_nesterov', 'boolean')
   cmd:option('-sgd_weightDecay', -666, 'sgd_weightDecay', 'number')
   cmd:option('-sgd_weightDecays', -666, 'sgd_weightDecays', 'number')
end

function Optimizer.Optimize(mlp, w, dw, criterion, GetDataset, train, test_sets,
                            loadmodel, update_err, get_err, print_stats)
   if params.origsgd then
      trainer = nn.StochasticGradient(mlp, criterion)
      trainer.learningRate = params.origsgd_learningRate
      trainer.maxIteration = params.maxiteration
      trainer:train(GetDataset(chtrain, params.batch))
   else
      local dataset = GetDataset(train,params.batch)
      local test_datasets = {}
      for k,v in pairs(test_sets) do
         test_datasets[k] = GetDataset(v, params.batchtest ~= 0
                                           and params.batchtest or params.batch)
      end

      mlp:zeroGradParameters()

      local opt = nil
      local state = nil
      local function init_state(pref)
         local state = {}
         for k,v in pairs(params) do
            if string.match(k, '^'..pref..'.*') and v ~= -666 then
               k = string.sub(k, #pref+1, -1)
               print(k,v)
               state[k] = v
            end
         end
         return state
      end
      if params.adadelta then
         opt = optim.adadelta
         state = init_state('adadelta_')
      elseif params.adagrad then
         opt = optim.adagrad
         state = init_state('adagrad_')
      elseif params.lbfgs then
         opt = optim.lbfgs
         state = init_state('lbfgs_')
      elseif params.sgd then
         opt = optim.sgd
         state = init_state('sgd_')
      else
         error('Choose an optimization method.')
      end

      state = loadmodel and loadmodel[5] or state
      oxnn.Optimizer.optimize(dataset, w, dw, mlp, criterion, opt, state,
                              test_datasets, update_err, get_err, print_stats)
   end
end

function Optimizer.PrintStats(stats)
   local train = stats.train
   local verbose = stats.verbose
   local name = stats.name
   local err = stats.err
   local eval_counter = stats.eval_counter
   local w = stats.w
   local dw = stats.dw
   local fx = stats.fx
   local iteration = stats.iteration
   local mb_start = stats.time_minibatch_start
   local mb = mb_start and xlua.formatTime(torch.toc(mb_start)) or nil
   local it_start = stats.time_iteration_start
   local tr_start = stats.time_training_start
   local it = xlua.formatTime(torch.toc(it_start))
   local tr = xlua.formatTime(torch.toc(tr_start))
   local t = stats.t
   local t_max = stats.t_max
   local p = nil
   local eta = nil
   local state = stats.state
   local example = stats.example
   if t then
      p = t / t_max
      eta = xlua.formatTime((torch.tic()-it_start) / p * (1. - p))
   end
   local dwnorm = stats.dwnorm or dw:norm()

   local function f3(f) return string.format('%.3f', f) end
   local function f5(f) return string.format('%.5f', f) end
   local output = ''
   if stats.train then
      output = output .. '# It:' .. iteration
   else
      output = output .. '# EV:' .. name
   end
   if stats.verbose then
      output = output .. ' P '  -- PART
   else
      output = output .. ' F '  -- FULL
   end
   local sep = ' '
   if stats.verbose then
      output = output
            .. f3(p*100) .. '%'
            .. ' Loss=' .. f5(fx[#fx]) .. sep
   end
   output = output
         .. 'Loss=' .. f5(err) .. sep
   if state and state.learningRate and params.Optimizer_lrDecayIter > 0 then
      output = output
            .. 'lr=' .. f5(state.learningRate) .. sep
   end
   output = output
         .. '#ev=' .. eval_counter .. sep
         .. '|w|=' .. f3(w:norm()) .. sep
         .. '|dw|=' .. f3(dwnorm) .. sep
         .. (mb and (mb..'/') or '') .. it .. '/' .. tr .. sep
   if stats.verbose then
      output = output
         .. 'ETA:' .. eta
   end
   print(output)
end

function Optimizer.PrintPerplexityStats(stats)
   local train = stats.train
   local verbose = stats.verbose
   local name = stats.name
   local err = stats.err
   local err_t = stats.err_t
   local eval_counter = stats.eval_counter
   local w = stats.w
   local dw = stats.dw
   local fx = stats.fx
   local iteration = stats.iteration
   local mb_start = stats.time_minibatch_start
   local mb = mb_start and xlua.formatTime(torch.toc(mb_start)) or nil
   local it_start = stats.time_iteration_start
   local tr_start = stats.time_training_start
   local it = xlua.formatTime(torch.toc(it_start))
   local tr = xlua.formatTime(torch.toc(tr_start))
   local t = stats.t
   local t_max = stats.t_max
   local p = nil
   local eta = nil
   local state = stats.state
   local example = stats.example
   if t then
      p = t / t_max
      eta = xlua.formatTime((torch.tic()-it_start) / p * (1. - p))
   end
   local wps = err_t.err_count / torch.toc(it_start)
   local dwnorm = stats.dwnorm or dw:norm()

   local function f3(f) return string.format('%.3f', f) end
   local function f5(f) return string.format('%.5f', f) end
   local function d(f) return string.format('%d', torch.round(f)) end
   local output = ''
   if stats.train then
      output = output .. '# It:' .. iteration
   else
      output = output .. '# EV:' .. name
   end
   if stats.verbose then
      output = output .. ' P '  -- PART
   else
      output = output .. ' F '  -- FULL
   end
   local sep = ' '

   local function perpe(nll, n) return math.exp(nll/n) end
   local perp = perpe

   if stats.verbose then
      assert(example)
      assert(example[1].perp_output_tokens, 'Need perp_output_tokens for loss averaging.')
      local n = 1--example[1].perp_output_tokens
      output = output
            .. f3(p*100) .. '%'
            .. ' Loss=' .. f5(fx[#fx] / n) .. sep
            .. 'Perp=' .. f3(perp(fx[#fx], n)) .. sep
   end
   output = output
         .. 'Loss=' .. f5(err) .. sep
         .. 'Perp=' .. f3(perp(err, 1))
            .. sep
   if state and state.learningRate and params.Optimizer_lrDecayIter > 0 then
      output = output
            .. 'lr=' .. f5(state.learningRate) .. sep
   end
   output = output
         .. '#ev=' .. eval_counter .. sep
         .. '|w|=' .. f3(w:norm()) .. sep
         .. '|dw|=' .. f3(dwnorm) .. sep
         .. (mb and (mb..'/') or '') .. it .. '/' .. tr .. sep
         .. 'w/s:' .. d(wps) .. sep
   if stats.verbose then
      output = output
         .. 'ETA:' .. eta
   end
   print(output)
end

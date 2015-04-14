-- Author: Tomas Kocisky
--
--[[ Example use:
--
--
local oxnnmc = require 'oxnn.util.models_common'
local cmd = oxnnmc.cmd

oxnn.Optimizer.AddCmdOptions(cmd)
cmd:option('-something', 20, '....')
...

oxnnmc.Init()   -- initializes CUDA, OpenMP,...

... load model and change params ...
local loadmodel = oxnn.Optimizer.LoadModel()

oxnnmc.MaybeLog('lstm1')

...
--
--
--]]
require 'oxnn'

oxnn_models_common = {}

oxnn_models_common.cmd = torch.CmdLine()
local cmd = oxnn_models_common.cmd
cmd:text()
cmd:text()
cmd:text('Model for abstract generation on NYT AC dataset')
cmd:text()
cmd:text('Options')
cmd:option('-cuda', false, 'Use cuda.')
cmd:option('-threads', 4, 'OpenMP threads.')
cmd:option('-strict', false, 'Check for global variables.')
cmd:option('-datadir', '', '')
cmd:option('-tmpdir', '', '')
cmd:option('-logdir', '', 'Folder for logs.')
cmd:option('-log', false, 'Should we log?')
cmd:option('-logdirname', '', 'Log folder name for this run. Overrides the '
                              .. 'default forder name based on the input '
                              .. 'arguments.')
cmd:option('-verbose', false, '')
cmd:option('-seed', 0, 'If greater then 0 then sets a manual seed for cu/torch.')

function oxnn_models_common.Init()
   cmd:text()

   --[[global]] params = cmd:parse(arg)

   assert(not params.log or #params.logdir > 0, 'Option -logdir required.')
   assert(#params.datadir > 0, 'Option -datadir required.')
   assert(#params.tmpdir > 0, 'Option -tmpdir required.')
   local function maybe_add_path_sep(path)
      if #path > 0 and path:sub(#path) ~= '/' then
         path = path .. '/'
      end
      return path
   end
   params.datadir = maybe_add_path_sep(params.datadir)
   params.tmpdir = maybe_add_path_sep(params.tmpdir)
   params.logdir = maybe_add_path_sep(params.logdir)

   --torch.setdefaulttensortype('torch.DoubleTensor')
   torch.setdefaulttensortype('torch.FloatTensor')

   -- Init CUDA
   if params.cuda then
      oxnn.InitCuda()
      --cutorch.deviceReset()
      --cutorch.setDevice(1)
      print( cutorch.getDeviceProperties(cutorch.getDevice()) )
   end
   if params.seed > 0 then
      oxnn.manualSeed(params.seed)
   end

   -- Init OpenMP and make sure we have it
   torch.setnumthreads(params.threads)
   assert(torch.getnumthreads() == params.threads, 'No OpenMP?')

   if params.strict then
      --require 'strict'
      require 'trepl'
      monitor_G()
   end
end

function oxnn_models_common.GetLogDir()
   local name = oxnn_models_common.MaybeLog_name
   assert(name, 'Logging not set up! Call MaybeLog(name).')
   params.rundir = cmd:string(name, params, {tmpdir=true, logdir=true,
                                             datadir=true})
   local logdir = params.logdir .. string.gsub(params.rundir,
                                               '/', '⁄'--[[U+2044]])
   if #params.logdirname > 0 then
      logdir = params.logdir .. name .. '_' .. params.logdirname
   end
   return logdir
end

function oxnn_models_common.MaybeLog(name)
   oxnn_models_common.MaybeLog_name = name
   -- create log file
   if params.log then
      local logdir = oxnn_models_common.GetLogDir()
      print('Logging to ' .. logdir .. '/log')
      os.execute('mkdir "' .. logdir .. '"')
      cmd:addTime()
      if paths.filep(logdir .. '/log') then
         error('Log file exists!')
      end
      cmd:log(logdir .. '/log', params)
   end
end

function oxnn_models_common.LoadParams(new_params, keep)
   local keep_ = {'log', 'logdir', 'tmpdir', 'datadir', 'cuda', 'threads',
                         'strict', 'verbose'}
   local keep_set = {}
   for _,v in pairs(keep_) do keep_set[v] = true end
   for _,v in pairs(keep) do keep_set[v] = true end
   for k,v in pairs(params) do   -- is this iteration ok??
      if not keep_set[k] then
         params[k] = new_params[k]
      end
   end
end

return oxnn_models_common

-- from https://github.com/wojciechz/learning_to_execute/blob/master/utils/utils.lua

--[[
   Copyright 2014 Google Inc. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
         http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
]]--

--[[ Creates clones of the given network.
The clones share all weights and gradWeights with the original network.
Accumulating of the gradients sums the gradients properly.
The clone also allows parameters for which gradients are never computed
to be shared. Such parameters must be returns by the parametersNoGrad
method, which can be null.
--]]
function oxnn.cloneManyTimes(net, T)
   local clones = {}
   local params, gradParams = net:parameters()
   if params == nil then
      params = {}
   end
   local paramsNoGrad
   if net.parametersNoGrad then
      paramsNoGrad = net:parametersNoGrad()
   end
   local mem = torch.MemoryFile("w"):binary()
   mem:writeObject(net)
   for t = 1, T do
      -- We need to use a new reader for each clone.
      -- We don't want to use the pointers to already read objects.
      local reader = torch.MemoryFile(mem:storage(), "r"):binary()
      local clone = reader:readObject()
      reader:close()
      local cloneParams, cloneGradParams = clone:parameters()
      local cloneParamsNoGrad
      for i = 1, #params do
         cloneParams[i]:set(params[i])
         cloneGradParams[i]:set(gradParams[i])
      end
      if paramsNoGrad then
         cloneParamsNoGrad = clone:parametersNoGrad()
         for i =1,#paramsNoGrad do
            cloneParamsNoGrad[i]:set(paramsNoGrad[i])
         end
      end
      clones[t] = clone
      collectgarbage()
   end
   mem:close()
   return clones
end

-- Authors: Tomas Kocisky
-- (Modification of above.)
function oxnn.cloneManyTimesFast(net, T)
   local clones = {}
   local params, gradParams = net:parameters()
   if params == nil then
      params = {}
   end
   local paramsNoGrad
   if net.parametersNoGrad then
      paramsNoGrad = net:parametersNoGrad()
   end
   local mem = torch.MemoryFile("w"):binary()
   mem:writeObject(net)

   -- serialize an empty clone for faster cloning
   do
      local reader = torch.MemoryFile(mem:storage(), "r"):binary()
      local clone = reader:readObject()
      reader:close()
      local cloneParams, cloneGradParams = clone:parameters()
      local cloneParamsNoGrad
      for i = 1, #params do
         cloneParams[i]:set(torch.Tensor():typeAs(params[i]))
         cloneGradParams[i]:set(torch.Tensor():typeAs(gradParams[i]))
      end
      if paramsNoGrad then
         cloneParamsNoGrad = clone:parametersNoGrad()
         for i =1,#paramsNoGrad do
            cloneParamsNoGrad[i]:set(torch.Tensor():typeAs(paramsNoGrad[i]))
         end
      end

      mem:close()
      mem = torch.MemoryFile("w"):binary()
      mem:writeObject(clone)
   end
   collectgarbage()

   for t = 1, T do
      -- We need to use a new reader for each clone.
      -- We don't want to use the pointers to already read objects.
      local reader = torch.MemoryFile(mem:storage(), "r"):binary()
      local clone = reader:readObject()
      reader:close()
      local cloneParams, cloneGradParams = clone:parameters()
      local cloneParamsNoGrad
      for i = 1, #params do
         cloneParams[i]:set(params[i])
         cloneGradParams[i]:set(gradParams[i])
      end
      if paramsNoGrad then
         cloneParamsNoGrad = clone:parametersNoGrad()
         for i =1,#paramsNoGrad do
            cloneParamsNoGrad[i]:set(paramsNoGrad[i])
         end
      end
      clones[t] = clone
      collectgarbage()
   end
   mem:close()
   return clones
end

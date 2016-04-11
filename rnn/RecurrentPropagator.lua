-- Authors: Tomas Kocisky
--
-- General class for propagaring inputs and gradients along a graph of
-- nn.Modules.
--
local RecurrentPropagator, parent = torch.class('oxnn.RecurrentPropagator', 'nn.Container')


function RecurrentPropagator:__init(cg)
   parent.__init(self)
   self.modules = {}
   self._cg = cg
   self._type = torch.getdefaulttensortype()
   -- disables some assertions and checks
   self.fast = (params and params.fast) or false
end


-- enables debug output, try 1 first, then 3
local RecurrentPropagator_debug = 0
RecurrentPropagator.debug = RecurrentPropagator_debug

RecurrentPropagator.cloneManyTimes = oxnn.cloneManyTimesFast
RecurrentPropagator.cudaSynchronizeAfterEachModule = false

function RecurrentPropagator:add(module)
   table.insert(self.modules, module)
   return self, 'm' .. #self.modules
end


-----------------------------------------------------------
----------[[ Sequence of Stacks data structure ]]----------
-----------------------------------------------------------

local function ref_find(ref)
   return table.pack(ref:find('([iom])([0-9]+):*([0-9]*)'))
end
ref_find = u.memoize(ref_find)

local function location_parse(ref)
   if (torch.type(ref) == 'oxnn.RecurrentPropagator_StackName'
         and ((ref.name and ref.name:sub(1,1) == 'i')
              or (ref.name2 and ref.name2[1] == 'i')))
      or torch.type(ref) == 'oxnn.RecurrentPropagator_StackElement' then

      ref = ref.name2 or ref.name
   end
   if torch.type(ref) == 'table' then
      return ref
   end

   assert(torch.type(ref) == 'string', 'Bad stack reference.')
   local _, _, what, i, j = unpack(ref_find(ref))
   assert(what)
   i = tonumber(i) or 1
   j = tonumber(j) or 1
   return {what, i, j}
end

function RecurrentPropagator:_s_get(ref, froms)
   if torch.isTypeOf(ref, nn.Module) then return ref end
   if torch.type(ref) == 'table' then
      local ret = {}
      for i, r in ipairs(ref) do ret[i] = self:_s_get(r, froms) end
      return ret
   end
   if self.debug > 1 then print('get', ref) end
   if self.debug > 3 then print(froms) end

   local what, i, j = unpack(location_parse(ref))
   local from = froms[what]
   local ret = from[i]
   if torch.type(ret) ~= 'table' then
      if not self.fast then assert(ret) end
      return ret
   else
      -- it's a stack
      ret = ret[#ret - j + 1]
      if not self.fast then assert(ret) end
      return ret
   end
end

function RecurrentPropagator:GetIO(ref)
   local froms = {i=self._inputs, o=self._outputs}
   return self:_s_get(ref, froms)
end

RecurrentPropagator.NOT_SET = {'not set'}
local NOT_SET = RecurrentPropagator.NOT_SET

local function RecursiveTensorAdd(dest, src)
   oxnn.RecursiveTensorAdd(dest, src, NOT_SET)
end

function RecurrentPropagator:_s_store(ref, froms, data, add, addhelper)
   if torch.type(ref) == 'table' then
      if data ~= NOT_SET then
         assert(#ref == #data, 'Number of store elements mismatch.')
         for i, r in ipairs(ref) do
            self:_s_store(r, froms, data[i], add, addhelper)
         end
      else
         -- data == NOT_SET
         -- we are populating the structure with the same data value
         for i, r in ipairs(ref) do
            self:_s_store(r, froms, data, add, addhelper)
         end
      end
      return
   end

   local what, i, j = unpack(location_parse(ref))
   if self.debug > 3 then print(froms, what) end
   local from = froms[what]
   from[i] = from[i] or {}
   if not add then
      j = 1
      --assert(j == 1, 'can only store at the top of the stack')
      table.insert(from[i], data)
   else
      addhelper[what] = addhelper[what] or {}
      if what == 'i' and (from[i] == NOT_SET
            or torch.type(from[i]) ~= 'table') then
         if from[i] == NOT_SET then
            from[i] = data
            addhelper[what][i] = false
         else
            if not addhelper[what][i] then
               from[i] = from[i]:clone()
               addhelper[what][i] = true
            end
            -- There is a possibility that data will be empty, in which case we ignore it
            if type(data) ~= 'table' and data:nElement() > 0 then
               from[i]:add(data)
            end
         end
      else
         local fro = from[i]
         addhelper[what][i] = addhelper[what][i] or {}
         if fro[ #(fro) - j + 1 ] == NOT_SET then
            fro[ #(fro) - j + 1 ] = data
            addhelper[what][i][ #(fro) - j + 1 ] = false  -- was not cloned yet
         else
            if type(fro[ #(fro) - j + 1 ]) ~= 'number'
                  and not addhelper[what][i][ #(fro) - j + 1 ] then
               fro[ #(fro) - j + 1 ] = oxnn.recursiveClone(fro[ #(fro) - j + 1 ])
               addhelper[what][i][ #(fro) - j + 1 ] = true
            end
            RecursiveTensorAdd(fro[ #(fro) - j + 1 ], data)
            --fro[ #(fro) - j + 1 ]:add(data)
         end
      end
   end
   if self.debug > 1 then
      print(add and 'storeADD' or 'store', ref)
   end
   if self.debug > 3 then print(froms) end
end

function RecurrentPropagator:_s_unstore(ref, froms)
   if torch.type(ref) == 'table' then
      for i,_ in ipairs(ref) do self:_s_unstore(ref[#ref-i+1], froms) end
      return
   end

   local what, i, j = unpack(location_parse(ref))
   j = 1
   --assert(j == 1, 'can only unstore the top of the stack')
   local from = froms[what]
   assert(from[i])
   from[i][ #(from[i]) ] = nil
   if self.debug > 1 then print('unstore', ref) end
   if self.debug > 3 then print(froms) end
end

--------------------------------------------------------------------
----------[[ (end of) Sequence of Stacks data structure ]]----------
--------------------------------------------------------------------


local function print_edge(e)
   local function delendl(s)  -- remove last endl
      if s[#s] == '\n' then s = s:sub(1,#s-1) end
      return s
   end
   local function format_io(io)
      if torch.type(io) == 'table' then
         local str = {}
         for i,v in ipairs(io) do
            if torch.type(v) == 'table' then
               table.insert(str, format_io(v))
            else
               table.insert(str, delendl(tostring(v)))
            end
         end
         return '{'..table.concat(str, ',')..'}'
      else
         return delendl(tostring(io))
      end
   end

   local c = sys.COLORS
   print(c.green..'EDGE:', format_io(e[1]), delendl(tostring(e[2])), format_io(e[3]), c.none)
end

function RecurrentPropagator.PrintCG(cg)
   local tab = '   '
   local line = '\n'
   local str = ''
   for i,v in ipairs(edges) do
      local i,m,o = unpack(v)
      if torch.type(i) == 'table' then i = table.concat(i, ',') end
      if torch.type(o) == 'table' then o = table.concat(o, ',') end
      str = str .. line .. tab .. '{' .. i .. '} -> ' .. tostring(m)
                     .. ' -> {' .. o .. '}'
   end
   print(str)
end

local function outsCheck1(outs)
   -- outs need to be reversible and we can store only at the top of a stack,
   -- therefore is there are N stores to the same stack, with N>1, they need to
   -- be numbered from N to 1 precisely in this order
   outs = torch.type(outs) == 'table' and outs or { outs }
   outs = oxnn.flatten(outs)

   local stacks = {}
   for _,v in ipairs(outs) do
      local what, i, j = unpack(location_parse(v))
      stacks[what..i] = stacks[what] or {}
      table.insert(stacks[what..i], j)
   end
   for k,v in pairs(stacks) do
      for i=1,#v do
         assert(v[#v-i+1] == i, 'Bad outs. See comment above. ' ..
                                '('..k..':'..i..')')
      end
   end
   return true
end

function RecurrentPropagator:updateOutput(input)
   if self.debug > 0 then print('updateOutput') end
   self._GradOutputs = nil
   self._Inputs = nil
   self._inputs = nil
   self._outputs = nil
   self._last_output = nil
   self.output = nil
   self.gradInput = nil
   self._modules_run_count = nil

   -- indexed by "step" in cg
   self._Inputs = {} -- references to self._inputs and self._outputs
   -- references to above indexed by cg annotations
   self._inputs = input
   self._outputs = {}
   self._modules_run_count = {}
   self._module_clones = self._module_clones or {}
   local mcs = self._module_clones
   local modrc = self._modules_run_count
   local cg = torch.type(self._cg) == 'function'
                  and self._cg(input, self._type) or self._cg
   self._cg_last = cg

   -- clone modules
   do
      local rc = {}   -- projected run count, ie clones needed
      for idx = 1, #cg do
         local e = cg[idx]
         local _, mod_, _ = unpack(e)
         rc[mod_] = rc[mod_] and rc[mod_] + 1 or 1
      end

      local clones_needed = 0
      for k,v in pairs(rc) do
         if type(k) == 'string' then
            mcs[k] = mcs[k] or {}
            local new_clone_count = rc[k] - #(mcs[k])
            clones_needed = clones_needed + math.max(0, new_clone_count)
         end
      end

      local clones_made = 0
      if clones_needed > 0 then
         --print('Cloning '..clones_made..'/'..clones_needed)
      end
      for k,v in pairs(rc) do
         if type(k) == 'string' then
            mcs[k] = mcs[k] or {}
            local new_clone_count = rc[k] - #(mcs[k])
            if new_clone_count > 0 then
               --print(k)
               local orig = self:_s_get(k, {m=self.modules})
               if not orig.__oxnn_RP_createCloneAndShareAll then
                  local new_clones =
                     RecurrentPropagator.cloneManyTimes(orig, new_clone_count)
                  for _,clone in ipairs(new_clones) do
                     table.insert(mcs[k], clone)
                  end
               else
                  for i=1,new_clone_count do
                     table.insert(mcs[k], orig:__oxnn_RP_createCloneAndShareAll())
                     collectgarbage()
                  end
               end
               collectgarbage()
               clones_made = clones_made + new_clone_count
               if clones_needed > 0 then
                  --print('Cloning '..clones_made..'/'..clones_needed)
               end
            end
         end
      end
      --if printing then print 'clones created' end
   end

   local froms = {i=self._inputs, o=self._outputs}
   local froms_store = {o=self._outputs}
   for idx = 1, #cg do
      local e = cg[idx]
      local ins_, mod_, outs_ = unpack(e)
      if self.debug > 2 then
         io.write(sys.COLORS.red)
         print(table.concat(u.rep('-', 80), ''))
         io.write(sys.COLORS.none)
      end
      if self.debug > 0 then print_edge(e) end
      if not self.fast then
         assert(outsCheck1(outs_))
      end

      local ins = self:_s_get(ins_, froms)
      modrc[mod_] = modrc[mod_] and modrc[mod_] + 1 or 1
      local mod = mcs[mod_] and mcs[mod_][modrc[mod_]] or mod_
      self._Inputs[idx] = ins

      if self.debug > 2 then
         io.write(sys.COLORS.yellow)
         oxnn.recprint(ins, 'ins')
         io.write(sys.COLORS.none)
         oxnn.recprint(mod)
      end
      ----
      self.output = mod:forward(ins)
      if RecurrentPropagator.cudaSynchronizeAfterEachModule
         and package.loaded.cutorch then cutorch.synchronize() end
      ----
      if self.debug > 2 then
         io.write(sys.COLORS.cyan)
         oxnn.recprint(self.output, 'output')
         io.write(sys.COLORS.none)
      end

      self:_s_store(outs_, froms_store, self.output)
      self._last_output = self.output
   end

   if self.debug > 0 then print('updateOutput end') end
   return self.output
end

local function is_not_set(v)
   if torch.type(v) == 'table' then
      for _,e in pairs(v) do if is_not_set(e) then return true end end
   end
   return v == NOT_SET
end

function RecurrentPropagator:updateGradInput(input, gradOutput)
   if self.debug > 0 then print('updateGradInput') end
   if not self.fast then
      assert(input == self._inputs) -- perhaps too strict
   end
   local cg = self._cg_last
   self._GradOutputs = {}

   local function recursiveCloneStacks(t, depth)
      depth = depth or 0
      local clone
      if depth < 2 and torch.type(t) == 'table' then
         clone = {}
         for i = 1, #t do
            clone[i] = recursiveCloneStacks(t[i], depth+1)
         end
      else
         if torch.typename(t) and torch.typename(t):find('torch%..+Tensor') then
            clone = NOT_SET
         elseif depth >= 1 then
            clone = NOT_SET
         else
            error('Unimplemented feature.')
         end
      end
      return clone
   end

   local gradInputs_i = recursiveCloneStacks(self._inputs)
   local gradInputs_o = {}

   if self.debug > 0 then print'simulate outputs' end

   for idx = 1, #cg do
      local e = cg[idx]
      local ins_, mod_, outs_ = unpack(e)
      if idx < #cg then
         self:_s_store(outs_, {o=gradInputs_o}, NOT_SET)
      else
         self:_s_store(outs_, {o=gradInputs_o}, gradOutput)
      end
   end
   if self.debug > 0 then print'grad input calculation' end
   local modrc = {}   -- local only
   local mcs = self._module_clones
   local froms_unstore = {o=gradInputs_o}
   local froms_store = {i=gradInputs_i, o=gradInputs_o}
   local store_addhelper = {}
   for idx = #cg, 1, -1 do
      local e = cg[idx]
      local ins_, mod_, outs_ = unpack(e)
      if self.debug > 2 then
         io.write(sys.COLORS.red)
         print(table.concat(u.rep('-', 80), ''))
         io.write(sys.COLORS.none)
      end
      if self.debug > 0 then print_edge(e) end

      local ins = self._Inputs[idx]
      modrc[mod_] = modrc[mod_] and modrc[mod_] + 1 or 1
      local mod = mcs[mod_] and
                     mcs[mod_][self._modules_run_count[mod_] - modrc[mod_]+1]
                     or mod_
      if not self.fast then assert(mod) end
      local gradOutput = self:_s_get(outs_, {i=gradInputs_i, o=gradInputs_o})
      if not self.fast then
         assert(not is_not_set(gradOutput)
                or torch.isTypeOf(mod, oxnn.CriterionTable),
                'Missing gradOutput. Expecting already computed gradOutput ' ..
                'or this module to be oxnn.CriterionTable')
      end

      --assert(not is_not_set(gradOutput) or torch.isTypeOf(mod, nn.Criterion)
      --                            or torch.isTypeOf(mod, nn.CriterionTable))
      self._GradOutputs[idx] = gradOutput

      if self.debug > 2 then
         io.write(sys.COLORS.yellow)
         oxnn.recprint(ins, 'ins')
         io.write(sys.COLORS.magenta)
         oxnn.recprint(gradOutput, 'gradOutput')
         io.write(sys.COLORS.none)
      end
      ----
      local gradInput = mod:updateGradInput(ins, gradOutput)
      if RecurrentPropagator.cudaSynchronizeAfterEachModule
         and package.loaded.cutorch then cutorch.synchronize() end
      ----
      if self.debug > 2 then
         io.write(sys.COLORS.cyan)
         oxnn.recprint(gradInput, 'gradInput')
         io.write(sys.COLORS.none)
      end

      self:_s_unstore(outs_, froms_unstore)
      self:_s_store(ins_, froms_store, gradInput, true, store_addhelper)
   end

   local function zero(i, gi, idx)
      if gi[idx] == NOT_SET then
         gi[idx] = oxnn.recursiveClone(i[idx],
                                      function (t)
                                          return type(t)=='number' and 0
                                             or t.new():resizeAs(t):zero() end)
      elseif torch.type(i[idx]) == 'table' then
         for k,_ in pairs(i[idx]) do
            zero(i[idx], gi[idx], k)
         end
      elseif torch.type(i[idx]) == 'number' then
         gi[idx] = 0
      elseif torch.type(i[idx]):match('torch%..*Tensor') then
      else
         error('This shouldn\'t happen')
      end
   end

   for i,_ in ipairs(gradInputs_i) do
      zero(self._inputs, gradInputs_i, i)
   end
   self.gradInput = gradInputs_i
   if self.debug > 0 then print('updateGradInput end') end
   return self.gradInput
end

function RecurrentPropagator:_accGradParameters(input, gradOutput, scaleOrLr,
                                                tocall)
   if self.debug > 0 then print('_acc(Update)GradInput') end
   if not self.fast then
      assert(input == self._inputs) -- perhaps too strict
   end
   local cg = self._cg_last
   if self.debug > 0 then
      print'acc (update) grad parameters calculation'
   end
   local modrc = {}   -- local only
   for idx = #cg, 1, -1 do
      local e = cg[idx]
      local ins_, mod_, outs_ = unpack(e)
      if self.debug > 0 then print_edge(e) end

      local ins = self._Inputs[idx]
      modrc[mod_] = modrc[mod_] and modrc[mod_] + 1 or 1
      local mcs = self._module_clones
      local mod = mcs[mod_] and
                      mcs[mod_][self._modules_run_count[mod_] - modrc[mod_]+1]
                      or mod_
      if not self.fast then assert(mod) end
      local gradOutput = self._GradOutputs[idx]

      ----
      --mod[tocall](mod, ins, gradOutput, scaleOrLr / modrc[mod_] / batch_size)
      mod[tocall](mod, ins, gradOutput, scaleOrLr)
      if RecurrentPropagator.cudaSynchronizeAfterEachModule
         and package.loaded.cutorch then cutorch.synchronize() end
      ----
   end
   if self.debug > 0 then print('_acc(Update)GradInput end') end
end

function RecurrentPropagator:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self:_accGradParameters(input, gradOutput, scale, 'accGradParameters')
end

function RecurrentPropagator:accUpdateGradParameters(input, gradOutput, lr)
   self:_accGradParameters(input, gradOutput, lr, 'accUpdateGradParameters')
end

function RecurrentPropagator:zeroGradParameters()
   for i=1,#self.modules do
      self.modules[i]:zeroGradParameters()
   end
   self._GradOutputs = nil
   --self._Inputs = nil
   --self._inputs = nil
   --self._outputs = nil
   --self._last_output = nil
   --self.output = nil
   self.gradInput = nil
   --self._modules_run_count = nil
end

function RecurrentPropagator:updateParameters(learningRate)
   for i=1,#self.modules do
      self.modules[i]:updateParameters(learningRate)
   end
end

function RecurrentPropagator:training()
   for i=1,#self.modules do
      self.modules[i]:training()
   end
   if self._module_clones then
      for _,v in pairs(self._module_clones) do
         for _,v in ipairs(v) do
            v:training()
         end
      end
   end
end

function RecurrentPropagator:evaluate()
   for i=1,#self.modules do
      self.modules[i]:evaluate()
   end
   if self._module_clones then
      for k,v in pairs(self._module_clones) do
         for i,v in ipairs(v) do
            v:evaluate()
         end
      end
   end
end

function RecurrentPropagator:share(mlp,...)
   assert(false)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

function RecurrentPropagator:reset(stdv)
   for i=1,#self.modules do
      self.modules[i]:reset(stdv)
   end
end

function RecurrentPropagator:parameters()
   local function tinsert(to, from)
      if type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   local w = {}
   local gw = {}
   for i=1,#self.modules do
      local mw,mgw = self.modules[i]:parameters()
      if mw then
         tinsert(w,mw)
         tinsert(gw,mgw)
      end
   end
   return w,gw
end

function RecurrentPropagator:clearState()
   self._GradOutputs = nil
   self._Inputs = nil
   self._inputs = nil
   self._outputs = nil
   self._last_output = nil
   self.output = nil
   self.gradInput = nil
   self._modules_run_count = nil
   self._module_clones = nil
   --self._cg = nil
   self._cg_last = nil
   self._modules_run_count = nil
   for i=1,#self.modules do
      self.modules[i]:zeroGradParameters()
   end
end

function RecurrentPropagator:type(type)
   self:clearState()
   parent.type(self, type)
   self._type = type
   return self
end

function RecurrentPropagator:__tostring__()
   local tab = '   '
   local line = '\n'
   local str = 'oxnn.RecurrentPropagator'
   str = str .. ' {'
   for i=1,#self.modules do
      str = str .. line .. tab .. 'm' .. i .. ': ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   if self._cg_last then
      str = str .. line .. 'last computation graph:'
      for i,v in ipairs(self._cg_last) do
         i,m,o = unpack(v)
         if torch.type(i) == 'table' then i = table.concat(i, ',') end
         if torch.type(o) == 'table' then o = table.concat(o, ',') end
         str = str .. line .. tab .. '{' .. i .. '} -> ' .. m
                        .. ' -> {' .. o .. '}'
      end
   end
   str = str .. line .. '}'
   return str
end

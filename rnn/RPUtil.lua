local RPUtil = torch.class('oxnn.RPUtil')


function RPUtil.CGUtil(edges)
   local r = {}

   function r.E(e)
      table.insert(edges, e)
   end

   local stack_context = oxnn.RecurrentPropagator_StackContext()
   function r.S(comment)
      return oxnn.RecurrentPropagator_StackName(stack_context, comment)
   end
   function r.Ss(n, comment)
      local res = {}
      for i=1,n do
         local c = comment and comment..n or nil
         res[#res+1] = oxnn.RecurrentPropagator_StackName(stack_context, c)
      end
      return res
   end
   function r.i(i)
      assert(i)
      local s = oxnn.RecurrentPropagator_StackName()
      s._context = stack_context
      s.name = nil--'i'..i
      s.name2 = {'i', i, 1}
      s.comment = ''
      return s
   end
   function r.Split(stackElement, count, comment)
      comment = comment or 'split'
      local s = r.S(comment)
      r.E { stackElement, nn.Identity(), s[{count,1}] }
      return s[{count,1}]
   end
   function r.Join(elements, comment)
      comment = comment or 'join'
      local s = r.S(comment)
      r.E { elements, nn.Identity(), s[1] }
      return s[1]
   end

   return r
end

local StackContext = torch.class('oxnn.RecurrentPropagator_StackContext')
function StackContext:__init()
   self._used = {}
   self._cnt = 0
end
function StackContext:get()
   self._cnt = self._cnt + 1
   return self._cnt
end

local StackName = torch.class('oxnn.RecurrentPropagator_StackName')
function StackName:__init(context, comment)
   self._context = context
   self.name = nil
   if context then self.name2 = {'o', context:get(), 1} end
   self.comment = comment
end
function StackName:__index(k)
   if torch.type(k) == 'number' then
      assert(k >= 1)
      return oxnn.RecurrentPropagator_StackElement(self, k)
   elseif torch.type(k) == 'table' then
      local res = {}
      assert(#k==2)
      for i = k[1], k[2], (k[1] < k[2] and 1 or -1) do
         res[#res+1] = oxnn.RecurrentPropagator_StackElement(self, i)
      end
      return res
   end
   return rawget(StackName, k)
end
function StackName:__tostring()
   local n = self.name or (self.name2[1]..self.name2[2])
   return n .. (#self.comment>0 and '('..self.comment..')' or '')
end

local StackElement = torch.class('oxnn.RecurrentPropagator_StackElement')
function StackElement:__init(stackName, j)
   self._stackName = stackName
   self._j = j
   self.name = stackName.name and stackName.name..':'..j or nil
   if stackName.name2 then
      self.name2 = {stackName.name2[1], stackName.name2[2]}
      self.name2[3] = j
   end
   self.comment = stackName.comment
end
function StackElement:__tostring()
   local n = self.name or (self.name2[1]..self.name2[2]..':'..self.name2[3])
   return n .. (#self.comment>0 and '('..self.comment..')' or '')
end




--
-- module should be if output_stack is give:
--  of type { recurrent_stack, next_input } to { recurrent_stack, output_stack }
-- otherwise
--  of type { recurrent_stack, next_input } to recurrent_stack
--
-- last input should be at the TOP of the stack
--
function RPUtil.RecurrentSequence(args)
   assert(type(args) == 'table', "Requires key-value arguments")
   local _, edges, r, module, initial_state, input_length,
         input_stack, recurrent_stack, output_stack, output_stack_masked, masks,
         type = xlua.unpack(
               {args},
               'RecurrentSequence',
               'Generates CG edges to process a sequence',
               {arg='edges', type='table', req=true,
                help='list to append edges to'},
               {arg='r', type='CGUtil', req=true, help=''},
               {arg='module', type='string | nn.Module', req=true,
                help='module to repeat'},
               {arg='initial_state', type='string', req=true,
                help='initial recurrent state'},
               {arg='input_length', type='int', req=true,
                help='maximal input sequence length'},
               {arg='input_stack', type='string', req=true,
                help='stack with inputs'},
               {arg='recurrent_stack', type='string', req=true,
                help='tmp stack for recurrent states'},
               {arg='output_stack', type='string', req=false,
                help='optional stack for outputs'},
               {arg='output_stack_masked', type='string', req=false,
                help='optional stack for outputs'},
               {arg='masks', type='string', req=false,
                help='masks for output'},
               {arg='type', type='string', req=true,
                help='type of tensors used'}
         )

   r.E {initial_state, nn.Identity():type(type), recurrent_stack[1]}
   for i = input_length, 1, -1 do
      local seq_pos = input_length - i + 1   -- 1,..,input_length
      local next_input = input_stack[i]
      --------
      local rec_out = output_stack and {recurrent_stack[1], output_stack[1]}
                                   or recurrent_stack[1]
      r.E {{recurrent_stack[1], next_input}, module, rec_out}
      --------
   end
   if masks then
      -- collect recurrent result
      local outs = {}
      for i,idx in ipairs(masks.endedat) do
         assert(idx >= 1)
         table.insert(outs, recurrent_stack[input_length + 1 - idx])
      end
      -- FIXME: Don't allocate memory.
      r.E {outs, oxnn.MaskedAdd(masks.ended, masks.endedat), recurrent_stack[1]}
      -- possibly mask outputs
      if output_stack_masked then
         -- not the best way of doing this since we allocate new memory in the
         -- modules
         for i = input_length, 1, -1 do
            local seq_pos = input_length - i + 1   -- 1,..,input_length
            if masks.active:select(1,seq_pos):sum()
                  ~= masks.active:size(1) then
               r.E {output_stack[i], oxnn.Mask(masks.active:select(1,seq_pos)),
                    output_stack_masked[1]}
            else
               r.E {output_stack[i], nn.Identity(), output_stack_masked[1]}
            end
         end
      end
   end
end

function RPUtil.DeepRecurrentSequence(args)
   assert(type(args) == 'table', "Requires key-value arguments")
   local _, edges, r, modules, initial_state_stack, input_length,
         input_stack, recurrent_stack, output_stack, layers,
         modules_layer2layer, output_stack_masked, masks, type
         = xlua.unpack(
               {args},
               'DeepRecurrentSequence',
               'Generates CG edges to process a sequence',
               {arg='edges', type='table', req=true,
                help='list to append edges to'},
               {arg='r', type='CGUtil', req=true, help=''},
               {arg='modules', type='table of string | nn.Module', req=true,
                help='table of modules to repeat'},
               {arg='initial_state_stack', type='string', req=true,
                help='initial recurrent state'},
               {arg='input_length', type='int', req=true,
                help='maximal input sequence length'},
               {arg='input_stack', type='string', req=true,
                help='stack with inputs'},
               {arg='recurrent_stack', type='string', req=true,
                help='final recurrent states from each layer'},
               {arg='output_stack', type='string', req=false,
                help='optional stack for outputs'},
               {arg='layers', type='number', req=false,
                help='number of layers'},
               {arg='modules_layer2layer', type='false OR table of string | nn.Module',
                req=false, help='number of layers'},
               {arg='output_stack_masked', type='string', req=false,
                help='optional stack for outputs'},
               {arg='masks', type='string', req=false,
                help='masks for output'},
               {arg='type', type='string', req=true,
                help='type of tensors used'}
         )

   local l = layers or 1
   assert(#modules == l)
   assert(modules_layer2layer == false or #modules_layer2layer == l-1)

   local function s(n)  -- layer suffix
      local len = #(''..(2*l))
      return string.format('%0'..len..'d', n)
   end

   local prev_output_stack = nil
   for i = 1, l do
      local rec = r.S('rec:l'..s(i))
      if i == 1 and i == l then  -- only one layer
         oxnn.RPUtil.RecurrentSequence{
                                   edges=edges,
                                   r=r,
                                   module=modules[i],
                                   initial_state=initial_state_stack[l-i+1],
                                   input_length=input_length,
                                   input_stack=input_stack,
                                   recurrent_stack=rec,
                                   output_stack=output_stack,
                                   output_stack_masked=output_stack_masked,
                                   masks=masks,
                                   type=type,
                                   }
      elseif i == 1 then  -- first layer
         local next_output_stack = r.S('out:l'..s(i))
         oxnn.RPUtil.RecurrentSequence{
                                   edges=edges,
                                   r=r,
                                   module=modules[i],
                                   initial_state=initial_state_stack[l-i+1],
                                   input_length=input_length,
                                   input_stack=input_stack,
                                   recurrent_stack=rec,
                                   output_stack=next_output_stack,
                                   masks=masks,
                                   type=type,
                                   }
         prev_output_stack = next_output_stack
         if modules_layer2layer ~= false then
            local next_output_stack = r.S('out:l'..s(i))
            r.E {prev_output_stack[{input_length,1}],
                 modules_layer2layer[i],
                 next_output_stack[{input_length,1}]}
            prev_output_stack = next_output_stack
         end
      elseif i == l then  -- last layer
         oxnn.RPUtil.RecurrentSequence{
                                   edges=edges,
                                   r=r,
                                   module=modules[i],
                                   initial_state=initial_state_stack[l-i+1],
                                   input_length=input_length,
                                   input_stack=prev_output_stack,
                                   recurrent_stack=rec,
                                   output_stack=output_stack,
                                   output_stack_masked=output_stack_masked,
                                   masks=masks,
                                   type=type,
                                   }
      else  -- other layers
         local next_output_stack = r.S('out:l'..s(i))
         oxnn.RPUtil.RecurrentSequence{
                                   edges=edges,
                                   r=r,
                                   module=modules[i],
                                   initial_state=initial_state_stack[l-i+1],
                                   input_length=input_length,
                                   input_stack=prev_output_stack,
                                   recurrent_stack=rec,
                                   output_stack=next_output_stack,
                                   masks=masks,
                                   type=type,
                                   }
         prev_output_stack = next_output_stack
         if modules_layer2layer ~= false then
            local next_output_stack = r.S('out:l'..s(i))
            r.E {prev_output_stack[{input_length,1}],
                 modules_layer2layer[i],
                 next_output_stack[{input_length,1}]}
            prev_output_stack = next_output_stack
         end
      end
      r.E {rec[1], nn.Identity(), recurrent_stack[1]}
   end
end

function RPUtil.CreateMasks(batch_size, max_len, lens, type)
   assert(type)
   local masks = {}
   masks.active = torch.Tensor():resize(max_len, batch_size):zero()
   local ends = {}   -- len -> index in batch
   assert(batch_size == #lens)
   for i, l in ipairs(lens) do
      if l > 0 then
         masks.active:narrow(2,i,1):narrow(1,1,l):fill(1)
      end
      ends[l] = ends[l] or {}
      table.insert(ends[l], i)
   end
   masks.active = masks.active:type(type)
   local ends_num = 0;
   for l = 1, max_len do if ends[l] then ends_num = ends_num + 1 end end
   masks.ended = torch.Tensor():resize(ends_num, batch_size):zero()
   masks.endedat = {}
   for l = 1, max_len do
      if ends[l] then
         table.insert(masks.endedat, l)
         local idx = #masks.endedat
         for _,i in pairs(ends[l]) do
            masks.ended[idx][i] = 1
         end
      end
   end
   masks.ended = masks.ended:type(type)
   return masks
end

function RPUtil.ClassNLLCriterionMaskedForWords(args)
   assert(type(args) == 'table', "Requires key-value arguments")
   local _, edges, r, lengths, joinedcriterion, input_stack, target_length,
      target_stack, target, module_split, output_stack, masks, type,
      mod_zl, mod_classnll, sumlosses = xlua.unpack(
               {args},
               'ClassNLLCriterionMaskedForWords',
               '',
               {arg='edges', type='table', req=true,
                help='list to append edges to'},
               {arg='r', type='CGUtil', req=true, help=''},
               {arg='lengths', type='table', req=false,
                help='table with sequence lengths'},
               {arg='joinedcriterion', type='boolean', req=true,
                help='whether to use only one ClassNLLCriterionMasked.'},
               {arg='input_stack', type='string', req=true,
                help='optional stack for losses'},
               {arg='target_length', type='int', req=true,
                help='maximal target sequence length'},
               {arg='target_stack', type='string', req=false,
                help='stack with targets'},
               {arg='target', type='string', req=false,
                help='target'},
               {arg='module_split', type='string | nn.Module', req=false,
                help='Split module when using target and not joined criterion.'},
               {arg='output_stack', type='string', req=true,
                help='optional stack for losses'},
               {arg='masks', type='string', req=false,
                help='masks for output'},
               {arg='type', type='string', req=true,
                help='type of tensors used'},
               {arg='module_zero_loss', type='string | nn.Module', req=false,
                help=''},
               {arg='module_classnll', type='string | nn.Module', req=true,
                help=''},
               {arg='sumlosses', type='boolean', req=false, help=''}
         )
   local len = assert(target_length)
   sumlosses = sumlosses==nil and true or sumlosses

   if not joinedcriterion then
      assert((target and module_split) or target_stack)
      -- possibly split, target will be on taget_stack
      if target and module_split then
         assert(target_stack == nil)
         target_stack = r.S('target_stack')
         r.E { target, module_split, target_stack[{len,1}] }
      end
      r.E {{target_stack[len]}, mod_zl, r.S('null')[1]}

      local losses = r.S('losses')
      if masks then
         local help_stack1 = r.S('help_stack1')
         for i = len,2,-1 do
            r.E {{},
                 oxnn.Constant(masks.active:select(1,len-i+1+1)):type(type),
                 help_stack1[1]}
         end
         for i = len,2,-1 do
            r.E {{input_stack[i-1], {target_stack[i-1], help_stack1[i-1]}},
                 mod_classnll,
                 losses[1]}
         end

      else
         for i = len,2,-1 do
            r.E {{input_stack[i-1], target_stack[i-1]}, mod_classnll, losses[1]}
         end
      end

      if sumlosses then
         -- sum the NLL's
         r.E {{losses[{len-1,1}]},
              oxnn.CriterionTable(oxnn.SumLosses(true, lengths)),
              output_stack[1]}
      else
         assert(lengths == nil)
         r.E {losses[{len-1,1}], nn.Identity(), output_stack[{len-1,1}]}
      end

   else
      assert(target and not module_split and not target_stack)
      local loss = r.S('loss')
      if masks then
         local help_stack1 = r.S('help_stack1')
         assert(len > 1)
         r.E {{},
              oxnn.Constant(masks.active:narrow(1,2,len-1):view(-1)):type(type),
              help_stack1[1]}
         r.E {{input_stack[1], {target, help_stack1[1]}}, mod_classnll, loss[1]}
      else
         r.E {{input_stack[1], target}, mod_classnll, loss[1]}
      end

      -- sum the NLL's
      if sumlosses then
         r.E {{{loss[1]}},
              oxnn.CriterionTable(oxnn.SumLosses(true, lengths)),
              output_stack[1]}
      else
         assert(lengths == nil)
         r.E {loss[1], nn.Identity(), output_stack[1]}
      end
   end
end

local SequenceOfWords, parent = torch.class('oxnn.SequenceOfWords', 'nn.Container')

-- input = { initial_recurrent_state, word_sequence, word_sequence_length }
-- output = { last_recurrent_state(s), loss }
--
-- recurrent network:
--    { recurrent_state, next_input } -> { recurrent_state, next_output }
--   or if 'noloss' is given then
--    { recurrent_state, next_input } -> recurrent_state
-- output network:
--    if not joinedoutput then
--       one rec_net_output -> what is passed to one criterion
--    if joinedoutput and not joinedcriterion then
--       many to many module from outputs to what is passes to each criterion
--    if joinedoutput and joinedcriterion then
--       many to one module from outputs to what is passes to joined criterion
-- loss:
--    'noloss', or 'nllloss' which puts class nll loss for the next word
function SequenceOfWords:__init(args)
   assert(type(args) == 'table', "Requires key-value arguments")
   local _, lookuptable, recurrent, out, joinedoutput, joinedcriterion, loss,
      layers, layer2layer, rectangle, lookuptable_type = xlua.unpack(
               {args},
               'SequenceOfWords',
               'Processes sequence of words as RNN',
               {arg='lookuptable', type='nn.LookupTable', req=true,
                help='LookupTable to process and split the input sequence(s)'},
               {arg='recurrent', type='nn.Module | table', req=true,
                help='recurrent network to repeat'},
               {arg='output', type='nn.Module', req=false,
                help='output network that goes from the recurrent output to ' ..
                     ' ClassNLLCriterionMasked'},
               {arg='joinedoutput', type='boolean', req=false,
                help='whether to process all outputs with one modules, ' ..
                     'false by default'},
               {arg='joinedcriterion', type='boolean', req=false,
                help='whether to use only one criterion for all outputs, ' ..
                     'false by default'},
               {arg='loss', type='string', req=true,
                help='whether to use \'nllloss\' or \'noloss\' or \'outputs\''..
                     ' or \'outputsmasked\''},
               -- deep
               {arg='layers', type='number', req=false, help=''},
               {arg='layer2layer', type='nn.Module | table', req=false,
                help=''},
               {arg='rectangle', type='nn.Module', req=false, help=''},
               -- other
               {arg='lookuptable_type', type='string', req=false,
                help='Default it \'one2many\'. Other options are ' ..
                     '\'many2many\' and \'apply2many\''}
                     -- If input is one tensor then use one2many, lookuptable
                     -- shoudld also split it into time step inputs.
                     -- If input is a table (and not a tensor) then many2many
                     -- applys uses one module, and apply2many uses one module
                     -- for each of the elements of the input (each time step).
         )
   parent.__init(self)

   self.lookuptable = lookuptable
   self.recurrent = recurrent
   self.layer2layer = layer2layer
   self.rectangle = rectangle
   self.out = out
   self.loss = loss == nil and 'noloss' or loss
   assert(self.loss=='noloss' or self.loss=='nllloss' or self.loss=='outputs'
          or self.loss=='outputsmasked')
   self.layers = layers or 1
   self.lookuptable_type = lookuptable_type == nil and 'one2many'
                                                   or lookuptable_type
   assert(self.lookuptable_type == 'one2many'
            or self.lookuptable_type == 'many2many'
            or self.lookuptable_type == 'apply2many')

   self.joinedoutput = joinedoutput or false
   self.joinedcriterion = joinedcriterion or false
   if self.joinedcriterion then
      assert(self.joinedoutput, 'Requiring joinedoutput when using joined class.')
   end

   local rp = oxnn.RecurrentPropagator()
   self.rp = rp
   self.modules = {self.rp}

   self.mod = {}
   local function s(a,b) return b end
   self.mod.lt = s(rp:add(self.lookuptable))
   self.mod.rec = {}
   if not self.rectangle then
      if torch.type(self.recurrent) ~= 'table' then
         assert(self.layers == 1)
         table.insert(self.mod.rec, s(rp:add(self.recurrent)))
      else
         assert(#self.recurrent == self.layers)
         for i = 1,layers do
            table.insert(self.mod.rec, s(rp:add(self.recurrent[i])))
         end
      end
      self.mod.l2l = {}
      if self.layers > 1 and self.layer2layer then
         assert(#self.layer2layer == self.layers - 1)
         for i = 1,self.layers-1 do
            table.insert(self.mod.l2l, s(rp:add(self.layer2layer[i])))
         end
      end
   else
      assert(torch.type(self.recurrent) == 'table' and #self.recurrent == 0)
      self.mod.rect = s(rp:add(self.rectangle))
   end

   if self.out then self.mod.out = s(rp:add(self.out)) end
   self.mod.zl1 = s(rp:add(oxnn.CriterionTable(oxnn.ZeroLoss())))
   self.mod.zl2 = s(rp:add(oxnn.CriterionTable(oxnn.ZeroLoss())))
   if self.loss == 'nllloss' then
      self.mod.classnll = s(
         rp:add(oxnn.CriterionTable(oxnn.ClassNLLCriterionMasked())))
   end
   if not self.joinedcriterion then
      self._split_module = nn.SplitTable(2):type('torch.LongTensor')
      self.mod.split = s(rp:add(self._split_module))
   end

   rp._cg = self:_GetBatchToCG()
end

function SequenceOfWords:_GetBatchToCG()
   local m = self.mod
   local function batch_to_cg(batch, type)  -- batch is the input
      local edges = {}
      local r = oxnn.RPUtil.CGUtil(edges)

      local len
      local batch_size
      if torch.type(batch[2]) ~= 'table' then
         len = batch[2]:size(2)
         batch_size = batch[2]:size(1)
         batch.masks = oxnn.RPUtil.CreateMasks(batch_size, len, batch[3], type)
      else
         len = #batch[2]
         batch_size = batch[2][1]:size(1)
         batch.masks = oxnn.RPUtil.CreateMasks(batch_size, len, batch[3], type)
      end

      local inputs = r.S('inputs o10')
      if self.lookuptable_type == 'one2many' then
         r.E {r.i(2), m.lt, inputs[{len,1}]}
      elseif self.lookuptable_type == 'many2many' then
         r.E {r.i(2)[{len,1}], m.lt, inputs[{len,1}]}
      elseif self.lookuptable_type == 'apply2many' then
         for i=len,1,-1 do
            r.E {r.i(2)[i], m.lt, inputs[1]}
         end
      else
         error('This shouldn\'t happen.')
      end

      local recurrent_stack = r.S('rec o11')
      local outputs = r.S('outputs o14')
      local outputs_masked = nil
      if self.loss == 'outputsmasked' then
         outputs_masked = r.S('outputs_masked')
      end

      if not self.mod.rect then
         if self.layers == 1 then
            oxnn.RPUtil.RecurrentSequence{
                             edges=edges,
                             r=r,
                             module=m.rec[1],
                             initial_state=r.i(1),  -- initial recurrent state
                             input_length=len,
                             input_stack=inputs,
                             recurrent_stack=recurrent_stack,
                             output_stack=(self.loss=='nllloss'
                                             or self.loss=='outputs'
                                             or self.loss=='outputsmasked')
                                          and outputs or false,
                             output_stack_masked=outputs_masked,
                             masks=batch.masks,
                             type=type,
                            }
         else
            oxnn.RPUtil.DeepRecurrentSequence{
                          edges=edges,
                          r=r,
                          modules=m.rec,
                          initial_state_stack=r.i(1),  -- initial recurrent states
                          input_length=len,
                          input_stack=inputs,
                          recurrent_stack=recurrent_stack,
                          output_stack=(self.loss=='nllloss'
                                          or self.loss=='outputs'
                                          or self.loss=='outputsmasked')
                                       and outputs or false,
                          output_stack_masked=outputs_masked,
                          layers=self.layers,
                          modules_layer2layer=#m.l2l>0 and m.l2l or false,
                          masks=batch.masks,
                          type=type,
                         }
         end
      else
         local l = params.layers
         r.E {  -- initial state, inputs, lengths
              {r.i(1)[{l,1}], inputs[{len,1}], r.i(3)[{batch_size,1}]},
              self.mod.rect,
              {recurrent_stack[{l,1}], outputs[{len,1}]}}
      end

      local after_out = r.S('after out')
      local function process_outputs(from, to)
         local outs = nil
         local outputs = outputs_masked or outputs

         if not self.joinedoutput then
            outs = {}
            for i = from,to,-1 do
               r.E {outputs[i], m.out, after_out[1]}
            end
            outs = after_out[{from,to}]
         elseif self.joinedoutput and not self.joinedcriterion then
            r.E {outputs[{from,to}], m.out, after_out[{from-to+1,1}]}
            outs = after_out[{from-to+1,1}]
         else  -- self.joinedoutput and self.joinedclass
            local o1 = {}
            r.E {outputs[{from,to}], m.out, after_out[1]}
            outs = after_out[1]
         end
         return outs
      end

      local result = r.S('result')
      if self.loss == 'noloss' then
         r.E { {}, oxnn.Constant(0), result[1]}

      elseif self.loss == 'outputs' or self.loss == 'outputsmasked' then
         assert(m.out)
         local outs = process_outputs(len, 1)
         r.E { outs, nn.Identity(), result[1]}

      elseif self.loss == 'nllloss' then
         assert(m.out)
         process_outputs(len, 2)
         r.E {{outputs[1]}, m.zl1, r.S('null')[1]}

         if not self.joinedcriterion then
            oxnn.RPUtil.ClassNLLCriterionMaskedForWords{
                                         edges=edges,
                                         r=r,
                                         lengths=batch[3],
                                         joinedcriterion=self.joinedcriterion,
                                         input_stack=after_out,
                                         target_length=len,
                                         target=r.i(2),
                                         module_split=m.split,
                                         output_stack=result,
                                         masks=batch.masks,
                                         type=type,
                                         module_zero_loss=m.zl2,
                                         module_classnll=m.classnll,
                                        }
         else
            -- FIXME: This allocates memory every minibatch. It depends on len
            -- in argument to nn.Narrow
            local targets = r.S('targets o101')
            r.E {r.i(2),
                 (nn:Sequential():add(nn.Narrow(2,2,len-1))
                                 :add(nn.Transpose({1,2}))
                                 :add(nn.View(-1)))
                    :type(type=='torch.CudaTensor'
                          and 'torch.CudaTensor' or 'torch.LongTensor'),
                 targets[1]}
            oxnn.RPUtil.ClassNLLCriterionMaskedForWords{
                                         edges=edges,
                                         r=r,
                                         lengths=batch[3],
                                         joinedcriterion=self.joinedcriterion,
                                         input_stack=after_out,
                                         target_length=len,
                                         target=targets[1],
                                         output_stack=result,
                                         masks=batch.masks,
                                         type=type,
                                         module_zero_loss=m.zl2,
                                         module_classnll=m.classnll,
                                        }
         end
      else
         error('This shouldn\'t happen.')
      end

      -- oxnn.RecurrentPropagator output, and so output of this module
      local output = r.S('output')
      if self.layers == 1 then
         r.E {{recurrent_stack[1], result[1]},
              nn.Identity(),
              output[1]}
      else
         r.E {{recurrent_stack[{1,self.layers}], result[1]},
              nn.Identity(),
              output[1]}
      end

      return edges
   end

   return batch_to_cg
end

function SequenceOfWords:updateOutput(input)
   self.output = self.rp:updateOutput(input)
   return self.output
end

function SequenceOfWords:updateGradInput(input, gradOutput)
   self.gradInput = self.rp:updateGradInput(input, gradOutput)
   return self.gradInput
end

function SequenceOfWords:accGradParameters(input, gradOutput, scale)
   self.rp:accGradParameters(input, gradOutput, scale)
end

function SequenceOfWords:accUpdateGradParameters(input, gradOutput, lr)
   self.rp:accUpdateGradParameters(input, gradOutput, lr)
end

function SequenceOfWords:zeroGradParameters()
   self.rp:zeroGradParameters()
end

function SequenceOfWords:updateParameters(learningRate)
   self.rp:updateParameters(learningRate)
end

function SequenceOfWords:training()
   self.rp:training()
end

function SequenceOfWords:evaluate()
   self.rp:evaluate()
end

function SequenceOfWords:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...);
   end
end

function SequenceOfWords:reset(stdv)
   self.rp:reset(stdv)
end

function SequenceOfWords:type(type)
   parent.type(self, type)
   if self._split_module then
      self._split_module:type(type=='torch.CudaTensor' and 'torch.CudaTensor' or 'torch.LongTensor')
   end
end

function SequenceOfWords:parameters()
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

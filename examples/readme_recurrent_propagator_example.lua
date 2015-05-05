local cuda = false
require 'oxnn'
if cuda then oxnn.InitCuda() end

-- We implement a simple LSTM
--
-- Predict  B       C
--          ^       ^
--          |       |
-- init -> Cell -> Cell -> Cell -> representation
--          ^       ^       ^
--          |       |       |
--          A       B       C

lookuptable = nn.Sequential():add(nn.LookupTable(10, 128))
                             :add(nn.SplitTable(2))
recurrent = oxnn.ModelUtil.LSTMCell12cl(128, true)
output = nn.Sequential():add(nn.Linear(128, 10))
                        :add(oxnn.LogSoftMaxInplace(true,true))
criterion = oxnn.CriterionTable(nn.ClassNLLCriterion())
targets = nn.Sequential():add(nn.SplitTable(2))

rp = oxnn.RecurrentPropagator()
rp, mod_lt = rp:add(lookuptable)  -- modifies rp inplace
rp, mod_rec = rp:add(recurrent)
rp, mod_out = rp:add(output)
rp, mod_crit = rp:add(criterion)
rp, mod_targ = rp:add(targets)

rp._cg = function(batch, type)
   -- This function creates a computation graph for the given batch (i.e.
   -- input); it is called for each input. The modules are not executed during
   -- the run of this function.  type is the last type we used to type the
   -- RecurrentPropagator with.

   local edges = {}                                 -- the edges of the CG
   local r = oxnn.RPUtil.CGUtil(edges)              -- helper functions

   -- We will assume the batch has the same format as above example, but for one
   -- layer.
   local len = batch[2]:size(2)

   -- The outputs are stored in virtual stacks. We create a stack. The string
   -- argument shows only when debugging.
   -- (For debugging set RecurrentPropagator.debug to 1 or 3.
   --
   -- When storing we need specify the index, where inputs[1] would be the top
   -- of the stack.
   local inputs = r.S('inputs')
   -- Add the first edge to the edges table.
   r.E { r.i(2), mod_lt, inputs[{len,1}] }
   -- Output of the lookuptable is a table with input for each time step.
   -- inputs[{len,1}] is equivalent to {inputs[len],inputs[len-1]...inputs[1]}.
   -- (Note that we can store only at the top of the stack, however, when we are
   -- storing multiple values, we need to index them with contiguous decreasing
   -- indices up to 1. This is so that on backward pass we can reverse graph.)
   --
   -- Inputs from the batch are accessed similarly. The elements of the input
   -- table correspond the r.i(1), r.i(2),... Each input is regarded also as a
   -- stack and needs to be indexed, e.g. r.i(1)[2]; except in a special case
   -- when batch[j] is a tensor, then we can use simply r.i(j), as we did above.

   local rec_state = r.S('rec_state')
   r.E { r.i(1)[1], nn.Identity(), rec_state[1] }  -- initial LSTM state
   -- As a module in an edge we usually use the module string returned when
   -- adding a module to the RecurrentPropagator (here mod_lt,...); we can
   -- also use a new module, but this module is created anew for each batch, and
   -- it's particularly bad if it allocates memory. This is not an issue with
   -- nn.Identity().

   local outputs = r.S('outputs')
   for i=len,1,-1 do
      r.E { { rec_state[1], inputs[i] },
            mod_rec,
            { rec_state[1], outputs[1] } }
      -- Each time mod_rec is used, a clone that shares the parameters of the
      -- original is used (and reused for subsequent batches).
      -- We take the appropriate input and store the output on the top of the
      -- stack. Note that rec_state[1] on both input and output is a table of
      -- two elements: h and c (hidden layers of the LSTM).
   end

   local after_out = r.S('after_out')
   for i=len,2,-1 do
      r.E { outputs[i], mod_out, after_out[1] }
   end

   local targets_all = r.S('targets_all')
   -- To demonstrate r.Split
   r.E { r.i(2), mod_targ, targets_all[1] }
   local targets = r.Split(targets_all[1], len)
   -- this is equivalent to
   -- r.E { targets_all[1], nn.Identity(), targets[{len,1}] }
   -- where targets is a new stack. We could have "saved" to multiple stack
   -- places directly.

   local losses = r.S('loss')
   --target of ouput of first time step (after_out[len-1]) is the input of 
   --second timestep(target[len-1]), and so forth
   for i=len-1,1,-1 do
      r.E { { after_out[i], targets[i] },
            mod_crit,
            losses[1] }
   end


   -- Some of the computed values are not used and expect a gradient flowing
   -- back so we put zero loss on them. The only module allowed without a
   -- gradient flowing back is oxnn.CriterionTable .
   r.E { {outputs[1],nil}, oxnn.CriterionTable(oxnn.ZeroLoss()), r.S('0')[1] }
   r.E { {rec_state[1],nil}, oxnn.CriterionTable(oxnn.ZeroLoss()), r.S('0')[1] }
   r.E { {targets[len],nil}, oxnn.CriterionTable(oxnn.ZeroLoss()), r.S('0')[1] }
   -- Ideally, we would add the above modules to the RecurrentPropagator since
   -- they allocate memory; this way they do it for each batch.

   local lengths = {}
   for i=1,batch[2]:size(1) do table.insert(lengths, len) end
   r.E { {losses[{len-1,1}], nil},  -- table of table of losses since
                                    -- CriterionTable expects a table.
         oxnn.CriterionTable(oxnn.SumLosses(true, lengths)),  -- sum the losses
                                                              -- and average
         r.S('final output')[1] }
   -- output from RecurrentPropagator is the output of the last edge/module.

   return edges
end

-- batch of 2 sequences of lengths 4 and 4
input = { { { torch.zeros(2,128), torch.zeros(2,128) } },
          torch.Tensor{ { 1, 7, 9, 8 },   -- sentence 1
                        { 2, 3, 5, 6 } }, -- sentence 2
          { 4, 3 }  -- sentence lengths
        }

if cuda then
   rp:cuda()
   input[1][1][1] = input[1][1][2]:cuda()
   input[1][1][2] = input[1][1][2]:cuda()
   input[2] = input[2]:cuda()
end

print(rp:forward(input))
print(rp:backward(input, 0))  -- 0 since output if only a number and the loss
                              -- does not require a gradient.

# oxnn -- Oxford NN Library

This library contains extensions to the Torch nn and cunn libraries. This is
development code and is not fully tested.

## Highlights

  * RNNs
    * [oxnn.SequenceOfWords](rnn/SequenceOfWords.lua) - Deep RNN class for
      sentences. It can handle batches where sentence lengths vary across and
      within batches, with output and loss appropriately masked.
    * Optimized LSTM cell [oxnn.ModelUtil.LSTM12cl](util/ModelUtil.lua),
      [oxnn.LSTM12Part2](nn/LSTM12Part2.lua).
    * [oxnn.RecurrentPropagator](rnn/RecurrentPropagator.lua) - Module for
      executing custom computations graphs, which is useful for RNNs. It handles
      cloning and weight sharing of modules that are used multiple times.  Each
      batch can have a different computation graph,
  * NN
    * [oxnn.LinearBlockDiagonal](nn/LinearBlockDiagonal.lua)
    * [oxnn.LinearCAddInplace](nn/LinearCAddInplace.lua)
    * [oxnn.LogSoftMaxInplace](nn/LogSoftMaxInplace.lua)
    * [oxnn.NarrowTable](nn/NarrowTable.lua) - multiple narrows on one tensor.
    * [oxnn.NoAccGradParameters](nn/NoAccGradParameters.lua) - do not train
      parameters of a module.
    * [oxnn.VecsToVecs](nn/VecsToVecs.lua) - maps vectors to vectors, given and
      retuned as a table or a tensor of batch of vectors for each time step.
  * Text
    * [oxnn.Vocabulary](text/Vocabulary.lua)
    * [oxnn.TextUtil](text/TextUtil.lua)

Documentation can be found at the top of the files, and we provide some
(examples)[examples/].

## License

We release this code under the BSD license (see the [LICENSE](LICENSE) file).
Some of the files are modification of files from nn or cunn. File
[util/cloneManyTimes.lua](util/cloneManyTimes.lua) has a separate license.

## Installation

Clone this code into your `$HOME` directory and run the `./build.sh` command.

For your .bashrc:
```bash
# substitute $HOME with the path to the cloned repository
export LUA_PATH="$HOME/?/init.lua;$HOME/?.lua;$LUA_PATH"
export LUA_CPATH="$HOME/oxnn/?.so;$HOME/?.so;$LUA_CPATH"
```

To test the installation try running the tests and the examples.


## Short examples

### oxnn.SequenceOfWords

Two layer LSTM:
```lua
lstm = oxnn.SequenceOfWords{
   lookuptable = nn.Sequential():add(nn.LookupTable(10, 128))
                                :add(nn.SplitTable(2)),
   recurrent = { oxnn.ModelUtil.LSTMCell12cl(128, true),  -- layer 1
                 oxnn.ModelUtil.LSTMCell12cl(128, true) },-- layer 2
   output =
            nn.Sequential()
               :add(nn.Linear(128, 10))
               :add(oxnn.LogSoftMaxInplace(true,true)),
   loss = 'nllloss',
   layers = 2
}

pad = 10
-- batch of 2 sequences of lengths 4 and 3
input = { { { torch.zeros(2,128), torch.zeros(2,128) }, -- initial state layer 1
            { torch.zeros(2,128), torch.zeros(2,128) } }, -- initial state layer 2
          torch.Tensor{ { 1, 7, 9, 8   },   -- sentence 1
                        { 2, 3, 5, pad } }, -- sentence 2
          { 4, 3 }  -- sentence lengths
        }


print(lstm:forward(input))
{
  1 :
    {
      1 :
        {
          1 : DoubleTensor - size: 2x128   -- last recurrent state for layer 1
          2 : DoubleTensor - size: 2x128   -- not including the padding step
        }
      2 :
        {
          1 : DoubleTensor - size: 2x128   -- last recurrent state for layer 2
          2 : DoubleTensor - size: 2x128   -- not including the padding step
        }
    }
  2 : 2.3162038392024                      -- NLL/(3+2)  (per output token);
                                           -- as loss gradOutput pass 0
}
```

For more advanced examples see in the `examples/` directory.

### oxnn.RecurrentPropagator

We will implement a one layer LSTM that takes batches of sequences. Sequences
within a batch need to have the same lengths, but can change across batches.

```lua
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
```

For more advanced example see the implementation of
[oxnn.SequenceOfWords](rnn/SequenceOfWords.lua).

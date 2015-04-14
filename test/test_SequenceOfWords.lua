local precision = 1e-5
local t = oxnn.mytester

local function counting_net(worddim, V, seqs, double)
   local mlp = nn.Sequential()
   local function simple_rec()
      local prev_rec = nn.Identity()()
      local i = nn.Identity()()
      local rec = prev_rec
      if double then
         local lin = nn.Linear(worddim,worddim)
         lin.bias:zero()
         lin.weight:copy(torch.eye(worddim)*2)
         rec = lin(prev_rec)
      end
      local next_rec = nn.CAddTable()({rec, i})
      local o = nn.Identity()(next_rec)
      return nn.gModule({prev_rec, i}, {next_rec, o})
   end
   local function lookup_table()
      local lt = nn.LookupTable(V, worddim)
      lt.weight:zero()
      for i = 1,V do
         lt.weight[i][i] = 1
      end
      return nn.Sequential():add(lt):add(nn.SplitTable(2))
   end
   local function out()
      --[[
      local out = nn.Linear(worddim, V)
      out.weight:fill(0)
      out.bias:fill(0)
      for i = 1,math.min(worddim,V) do
         out.weight[i]:zero()
         out.weight[i][i] = 1
      end
      return nn.Sequential():add(out):add(nn.LogSoftMax())
      --]]
      return nn.Identity()
   end

   mlp:add(oxnn.SequenceOfWords{
      lookuptable = lookup_table(),
      recurrent = simple_rec(),
      output = out(),
      loss = 'nllloss',
   })
   if not double then
      mlp:add(nn.SelectTable(2))
   end
   mlp = oxnn.cu:MaybeType(mlp)

   local lens = {}
   local maxlen = 0
   for _,v in ipairs(seqs) do
      maxlen = math.max(maxlen, #v)
      lens[#lens+1] = #v
   end
   local tseqs = torch.LongTensor(#seqs, maxlen):fill(1)
   for i,v in ipairs(seqs) do
      for j,w in ipairs(v) do
         tseqs[i][j] = w
      end
   end

   local input = {
                   oxnn.cu:MaybeType(torch.zeros(#seqs,worddim)),
                   oxnn.cu:MaybeType(tseqs),
                   lens,
                   --batch_size = 1,
                 }
   return mlp, input
end

function oxnn.tests.oxnn_SequenceOfWords_counting_simple1()
   local worddim = 5
   local V = 5
   local seqs = {
                 {2,3,4},
                }
   local mlp, input = counting_net(worddim, V, seqs)

   mlp:forward(input)
   mlp:backward(input, mlp.output)
   local lt_gradWeight =
      mlp.modules[1].modules[1].modules[1].modules[1].gradWeight
   local lt_gradWeightExpected = oxnn.cu:MaybeType(torch.Tensor({
                                               { 0, 0, 0, 0, 0 },
                                               { 0, 0, -1, -1, 0 },
                                               { 0, 0, 0, -1, 0 },
                                               { 0, 0, 0, 0, 0 },
                                               { 0, 0, 0, 0, 0 },
                                              }))
   t:assertTensorEq(lt_gradWeight, lt_gradWeightExpected, precision)
end

function oxnn.tests.oxnn_SequenceOfWords_counting_simple2()
   local worddim = 5
   local V = 5
   local seqs = {
                 {2,3,4,5},
                 {2,3,4,5}
                }
   local mlp, input = counting_net(worddim, V, seqs)

   mlp:forward(input)
   mlp:backward(input, mlp.output)
   local lt_gradWeight =
      mlp.modules[1].modules[1].modules[1].modules[1].gradWeight
   local lt_gradWeightExpected = oxnn.cu:MaybeType(torch.Tensor({
                                               { 0, 0, 0, 0, 0 },
                                               { 0, 0, -1, -1, -1 },
                                               { 0, 0, 0, -1, -1 },
                                               { 0, 0, 0, 0, -1 },
                                               { 0, 0, 0, 0, 0 },
                                              })) * 2
   t:assertTensorEq(lt_gradWeight, lt_gradWeightExpected, precision)
end

function oxnn.tests.oxnn_SequenceOfWords_lens_different()
   local worddim = 5
   local V = 5
   local seqs = {
                 {2,3,4,5},
                 {2,3,4}
                }
   local mlp, input = counting_net(worddim, V, seqs)

   mlp:forward(input)
   mlp:backward(input, mlp.output)
   local lt_gradWeight =
      mlp.modules[1].modules[1].modules[1].modules[1].gradWeight
   local lt_gradWeightExpected = oxnn.cu:MaybeType(torch.Tensor({
                                               { 0, 0, 0, 0, 0 },
                                               { 0, 0, -1, -1, -0.5 },
                                               { 0, 0, 0, -1, -0.5 },
                                               { 0, 0, 0, 0, -0.5 },
                                               { 0, 0, 0, 0, 0 },
                                              })) * 2
   t:assertTensorEq(lt_gradWeight, lt_gradWeightExpected, precision)
end

function oxnn.tests.oxnn_SequenceOfWords_lens_same()
   local worddim = 5
   local V = 5
   local seqs = {
                 {2,3,4,5},
                 {2,3,4,1}
                }
   local mlp, input = counting_net(worddim, V, seqs)

   mlp:forward(input)
   mlp:backward(input, mlp.output)
   local lt_gradWeight =
      mlp.modules[1].modules[1].modules[1].modules[1].gradWeight
   local lt_gradWeightExpected = oxnn.cu:MaybeType(torch.Tensor({
                                               { 0, 0, 0, 0, 0 },
                                               { -0.5, 0, -1, -1, -0.5 },
                                               { -0.5, 0, 0, -1, -0.5 },
                                               { -0.5, 0, 0, 0, -0.5 },
                                               { 0, 0, 0, 0, 0 },
                                              })) * 2
   t:assertTensorEq(lt_gradWeight, lt_gradWeightExpected, precision)
end

function oxnn.tests.oxnn_SequenceOfWords_lens_different4()
   local worddim = 5
   local V = 5
   local seqs = {
                 {2,3,4,5,2},
                 {2,3,4},
                 {2,3,4,5},
                 {2,3},
                }
   local mlp, input = counting_net(worddim, V, seqs)

   mlp:forward(input)
   mlp:backward(input, mlp.output)
   local lt_gradWeight =
      mlp.modules[1].modules[1].modules[1].modules[1].gradWeight
   local lt_gradWeightExpected = oxnn.cu:MaybeType(torch.Tensor({
                                               { 0, 0, 0, 0, 0 },
                                               { 0, -0.25, -1, -0.75, -0.5 },
                                               { 0, -0.25, 0, -0.75, -0.5 },
                                               { 0, -0.25, 0, 0, -0.5 },
                                               { 0, -0.25, 0, 0, 0 },
                                              })) * 4
   t:assertTensorEq(lt_gradWeight, lt_gradWeightExpected, precision)
end

function oxnn.tests.oxnn_SequenceOfWords_lens_different4b()
   local worddim = 5
   local V = 5
   local seqs = {
                 {2,2,5,4,3},
                 {2,3,4},
                 {5,3,4,3},
                 {2,3},
                }
   local mlp, input = counting_net(worddim, V, seqs)

   mlp:forward(input)
   mlp:backward(input, mlp.output)
   local lt_gradWeight =
      mlp.modules[1].modules[1].modules[1].modules[1].gradWeight
   local lt_gradWeightExpected = oxnn.cu:MaybeType(torch.Tensor({
                                               { 0, 0, 0, 0, 0 },
                                               { 0, -0.25, -1, -0.75, -0.5 },
                                               { 0, 0, -0.25, -0.5, 0 },
                                               { 0, 0, -0.5, 0, 0 },
                                               { 0, 0, -0.75, -0.5, 0 },
                                              })) * 4
   t:assertTensorEq(lt_gradWeight, lt_gradWeightExpected, precision)
end


function oxnn.tests.oxnn_SequenceOfWords_lens_different4b_with_output()
   local worddim = 5
   local V = 5
   local seqs = {
                 {2,2,5,4,3},
                 {2,3,4},
                 {5,3,4,3},
                 {2,3},
                }
   local mlp, input = counting_net(worddim, V, seqs, true)

   mlp:forward(input)
   mlp:backward(input, { oxnn.cu:MaybeType(torch.ones(4,5)), 0 })
   local lt_gradWeight =
      mlp.modules[1].modules[1].modules[1].modules[1].gradWeight
   local lt_gradWeightExpected = oxnn.cu:MaybeType(torch.Tensor({
                                               { 0, 0, 0, 0, 0 },
                                               { 0, -0.25, -3.5, -2, -0.75 },
                                               { 0, 0, -0.5, -0.5, 0 },
                                               { 0, 0, -0.5, 0, 0 },
                                               { 0, 0, -1.75, -0.75, 0 },
                                              }) * 4
                                + torch.Tensor({
                                               { 0, 0, 0, 0, 0 },
                                               { 30, 30, 30, 30, 30 },
                                               { 9, 9, 9, 9, 9 },
                                               { 5, 5, 5, 5, 5 },
                                               { 12, 12, 12, 12, 12 },
                                              }))
   t:assertTensorEq(lt_gradWeight, lt_gradWeightExpected, precision)
end

function oxnn.tests.oxnn_SequenceOfWords_additive_loss()
   local worddim = 5
   local V = 5
   local seqs = {
                 {2,2,5,4,3},
                 {2,3,4},
                 {5,3,4,3},
                 {2,3},
                }
   local mlp, input = counting_net(worddim, V, seqs, true)
   local _, input1 = counting_net(worddim, V, {seqs[1]}, true)
   local _, input2 = counting_net(worddim, V, {seqs[2]}, true)
   local _, input3 = counting_net(worddim, V, {seqs[3]}, true)
   local _, input4 = counting_net(worddim, V, {seqs[4]}, true)

   local output = mlp:forward(input)[2]
   local output1 = mlp:forward(input1)[2]
   local output2 = mlp:forward(input2)[2]
   local output3 = mlp:forward(input3)[2]
   local output4 = mlp:forward(input4)[2]
   local outputs = output1*4/10 + output2*2/10 + output3*3/10 + output4*1/10
   t:assert(math.abs(output - outputs) < precision)
end

function oxnn.tests.oxnn_SequenceOfWords_additive_loss_randominit()
   local worddim = 5
   local V = 5
   local seqs = {
                 {2,2,5,4,3},
                 {2,3,4},
                 {5,3,4,3},
                 {2,3},
                }
   local mlp, input = counting_net(worddim, V, seqs, true)
   local _, input1 = counting_net(worddim, V, {seqs[1]}, true)
   local _, input2 = counting_net(worddim, V, {seqs[2]}, true)
   local _, input3 = counting_net(worddim, V, {seqs[3]}, true)
   local _, input4 = counting_net(worddim, V, {seqs[4]}, true)
   mlp:reset(0.1)

   local output = mlp:forward(input)[2]
   local output1 = mlp:forward(input1)[2]
   local output2 = mlp:forward(input2)[2]
   local output3 = mlp:forward(input3)[2]
   local output4 = mlp:forward(input4)[2]
   local outputs = output1*4/10 + output2*2/10 + output3*3/10 + output4*1/10
   t:assert(math.abs(output - outputs) < precision)
end

function oxnn.tests.oxnn_SequenceOfWords_additive_gradient()
   local worddim = 5
   local V = 5
   local seqs = {
                 {2,2,5,4,3},
                 {2,3,4},
                 {5,3,4,3},
                 {2,3},
                }
   local mlp, input = counting_net(worddim, V, seqs, true)
   local _, input1 = counting_net(worddim, V, {seqs[1]}, true)
   local _, input2 = counting_net(worddim, V, {seqs[2]}, true)
   local _, input3 = counting_net(worddim, V, {seqs[3]}, true)
   local _, input4 = counting_net(worddim, V, {seqs[4]}, true)

   local w,dw = mlp:getParameters()
   local orig_w = w:clone()
   local orig_dw = dw:clone()
   mlp:zeroGradParameters()

   local output = mlp:forward(input)[2]
   mlp:backward(input, { oxnn.cu:MaybeType(torch.ones(#seqs,5)), 0 })
   local dw_run1 = dw:clone()

   w:copy(orig_w)
   mlp:zeroGradParameters()
   local output1 = mlp:forward(input1)[2]
   mlp:backward(input1, { oxnn.cu:MaybeType(torch.ones(1,5)), 0 })
   local output2 = mlp:forward(input2)[2]
   mlp:backward(input2, { oxnn.cu:MaybeType(torch.ones(1,5)), 0 })
   local output3 = mlp:forward(input3)[2]
   mlp:backward(input3, { oxnn.cu:MaybeType(torch.ones(1,5)), 0 })
   local output4 = mlp:forward(input4)[2]
   mlp:backward(input4, { oxnn.cu:MaybeType(torch.ones(1,5)), 0 })
   local outputs = output1*4/10 + output2*2/10 + output3*3/10 + output4*1/10
   t:assert(math.abs(output - outputs) < precision)
   t:assertTensorEq(dw_run1, dw, precision)
end

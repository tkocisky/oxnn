-- Authors: Tomas Kocisky
--
local ModelUtil = torch.class('oxnn.ModelUtil')

-- Original implementation.
function ModelUtil.LSTMCell1(i2h, h2h, include_out)
   include_out = include_out or false

   local prev_state = nn.Identity()()
   local i = nn.Identity()()
   local prev_h, prev_c = prev_state:split(2)

   local next_h, next_c = ModelUtil.LSTMCell1Graph(prev_h, prev_c, i, i2h, h2h)

   local next_state = nn.Identity()({next_h, next_c})
   return nn.gModule({prev_state, i}, {next_state,
                                       include_out and next_h or nil})
end
function ModelUtil.LSTMCell1Graph(prev_h, prev_c, i, i2h, h2h)
   local function new_input_sum()
      local i2h = i2h:clone()
      local h2h = h2h:clone()
      return nn.CAddTable()({i2h(i), h2h(prev_h)})
   end

   local in_gate = nn.Sigmoid()(new_input_sum())
   local forget_gate = nn.Sigmoid()(new_input_sum())
   local out_gate = nn.Sigmoid()(new_input_sum())

   local input = nn.Tanh()(new_input_sum())
   local next_c = nn.CAddTable()({nn.CMulTable()({forget_gate, prev_c}),
                                  nn.CMulTable()({in_gate, input})})

   local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

   return next_h, next_c
end

-- More time and memory efficient implementation of basic LSTM.
--
-- Input/hidden transformation into 4 vectors at the same time. Some
-- nonlinearities are inplace. More efficient CAddTable and CMulTable layers.
--
function ModelUtil.LSTMCell11(i2h4, h2h4, h42h4, dim, include_out)
   include_out = include_out or false

   local prev_state = nn.Identity()()
   local i = nn.Identity()()
   local prev_h, prev_c = prev_state:split(2)

   local next_h, next_c = ModelUtil.LSTMCell11Graph(prev_h, prev_c, i, i2h4, h2h4, h42h4, dim)

   local next_state = nn.Identity()({next_h, next_c})
   return nn.gModule({prev_state, i}, {next_state,
                                       include_out and next_h or nil})
end
function ModelUtil.LSTMCell11Graph(prev_h, prev_c, i, i2h4, h2h4, h42h4, dim)
   local h4_1 = i2h4(i)
   local h4_2 = h2h4(prev_h)
   local h4 = oxnn.CAddTableNoCopy()({h4_1, h4_2})
   local raw_gates = h42h4(h4)

   -- inplace nonlinearities from cudnn
   local function Sigmoid() return oxnn.cu.use_cuda and cudnn.Sigmoid(true)
                                                    or nn.Sigmoid() end
   local function Tanh() return oxnn.cu.use_cuda and cudnn.Tanh(true)
                                                    or nn.Tanh() end

   local raw_gates_split = oxnn.NarrowTable(
                              2, {
                                  {0 * dim + 1, 1 * dim},
                                  {1 * dim + 1, 1 * dim},
                                  {2 * dim + 1, 1 * dim},
                                  {3 * dim + 1, 1 * dim},
                                 },
                             --[[copy=]]true, --[[zero/add=]]false)(raw_gates)
   local raw_in_gate, raw_forget_gate, raw_out_gate, raw_input
         = raw_gates_split:split(4)
   local forget_gate = Sigmoid()(raw_forget_gate)
   local out_gate =    Sigmoid()(raw_out_gate)
   local in_gate =     Sigmoid()(raw_in_gate)
   local input =          Tanh()(raw_input)

   local next_c = oxnn.CAddTableNoCopy()({
                                  oxnn.CMulTable2()({forget_gate, prev_c}),
                                  oxnn.CMulTable2()({in_gate, input})})

   local next_h = oxnn.CMulTable2()({out_gate, nn.Tanh()(next_c)})

   return next_h, next_c
end

-- Like LSTMCell11 but second part, where everything is component-wise, is fused
-- into one kernel invocation on CUDA.
function ModelUtil.LSTMCell12(i2h4, h2h4, h42h4, include_out)
   include_out = include_out or false

   local prev_state = nn.Identity()()
   local i = nn.Identity()()
   local prev_h, prev_c = prev_state:split(2)

   local next_h, next_c =
      ModelUtil.LSTMCell12Graph(prev_h, prev_c, i, i2h4, h2h4, h42h4)

   local next_state = nn.Identity()({next_h, next_c})
   return nn.gModule({prev_state, i}, {next_state,
                                       include_out and next_h or nil})
end

function ModelUtil.LSTMCell12Graph(prev_h, prev_c, i, i2h4, h2h4, h42h4)
   local h4_1 = i2h4(i)
   local h4_2 = h2h4(prev_h)
   local h4 = oxnn.CAddTableNoCopyInplace()({h4_1, h4_2})
   local raw_gates = h42h4(h4)

   local LSTM12Part2 = oxnn.LSTM12Part2()
   local next_c, next_h = LSTM12Part2({prev_c, raw_gates}):split(2)

   return next_h, next_c
end

-- Like LSTMCell12 but assumes:
--    i can be modified in the forward pass and is reused in backward pass,
--       also, there's no transformation applied to i
function ModelUtil.LSTMCell12c(hdim, include_out)
   include_out = include_out or false

   local prev_state = nn.Identity()()
   local i = nn.Identity()()
   local prev_h, prev_c = prev_state:split(2)

   local next_h, next_c = ModelUtil.LSTMCell12cGraph(hdim, prev_h, prev_c, i)
   local next_state = nn.Identity()({next_h, next_c})

   return nn.gModule({prev_state, i}, {next_state,
                                       include_out and next_h or nil})
end
function ModelUtil.LSTMCell12cGraph(hdim, prev_h, prev_c, i)
   local raw_gates = oxnn.LinearCAddInplace(hdim, 4*hdim)({ i, prev_h })

   local LSTM12Part2 = oxnn.LSTM12Part2()
   LSTM12Part2.gradInput_inplace_of_input = true

   local next_c, next_h = LSTM12Part2({prev_c, raw_gates}):split(2)
   return next_h, next_c
end

-- Like LSTMCell12c but includes transformation of i
function ModelUtil.LSTMCell12cl(hdim, include_out)
   include_out = include_out or false

   local prev_state = nn.Identity()()
   local i = nn.Identity()()
   local prev_h, prev_c = prev_state:split(2)

   local next_h, next_c = ModelUtil.LSTMCell12clGraph(hdim, prev_h, prev_c, i)
   local next_state = nn.Identity()({next_h, next_c})

   return nn.gModule({prev_state, i}, {next_state,
                                       include_out and next_h or nil})
end
function ModelUtil.LSTMCell12clGraph(hdim, prev_h, prev_c, i)
   local i = nn.Linear(hdim, 4*hdim)(i)
   return ModelUtil.LSTMCell12cGraph(hdim, prev_h, prev_c, i)
end


-- Block diagonal versions of LSTM12c and 12cl
function ModelUtil.LSTMCell12cb(hdim, blocks, include_out)
   include_out = include_out or false

   local prev_state = nn.Identity()()
   local i = nn.Identity()()
   local prev_h, prev_c = prev_state:split(2)

   local next_h, next_c = ModelUtil.LSTMCell12cbGraph(hdim, blocks, prev_h,
                                                      prev_c, i)
   local next_state = nn.Identity()({next_h, next_c})

   return nn.gModule({prev_state, i}, {next_state,
                                       include_out and next_h or nil})
end
function ModelUtil.LSTMCell12cbGraph(hdim, blocks, prev_h, prev_c, i)
   local raw_gates = oxnn.LinearCAddInplace(hdim, 4*hdim, blocks)({ i, prev_h })

   local LSTM12Part2 = oxnn.LSTM12Part2()
   LSTM12Part2.gradInput_inplace_of_input = true

   local next_c, next_h = LSTM12Part2({prev_c, raw_gates}):split(2)
   return next_h, next_c
end

-- Like LSTMCell12c but includes transformation of i
function ModelUtil.LSTMCell12cbl(hdim, blocks, include_out)
   include_out = include_out or false

   local prev_state = nn.Identity()()
   local i = nn.Identity()()
   local prev_h, prev_c = prev_state:split(2)

   local next_h, next_c = ModelUtil.LSTMCell12cblGraph(hdim, blocks, prev_h,
                                                       prev_c, i)
   local next_state = nn.Identity()({next_h, next_c})

   return nn.gModule({prev_state, i}, {next_state,
                                       include_out and next_h or nil})
end
function ModelUtil.LSTMCell12cblGraph(hdim, blocks, prev_h, prev_c, i)
   local i = oxnn.LinearBlockDiagonal(hdim, 4*hdim, blocks)(i)
   return ModelUtil.LSTMCell12cbGraph(hdim, blocks, prev_h, prev_c, i)
end






-- LSTMCell(hdim, include_out) ->
--    Module s.t. {{prev_h,prev_c},i} -> {{next_h,next_c},include_out?next_h:nil}
ModelUtil.LSTMCell = ModelUtil.LSTMCell12cl
-- LSTMCellGraph(hdim, prev_h, prev_c, i) -> next_h, next_c
ModelUtil.LSTMCellGraph = ModelUtil.LSTMCell12clGraph

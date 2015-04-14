-- Authors: Tomas Kocisky
--
-- Component-wise "part 2" of LSTM fused into one kernel for CUDA.
--
local LSTM12Part2, parent = torch.class('oxnn.LSTM12Part2', 'nn.Container')

function LSTM12Part2:__init()
   parent.__init(self)
   self.output = {}
   self.gradInput = {}
   self.gradInput_inplace_of_input = false
end

function LSTM12Part2:updateOutput(input)
   self.output[1] = self.output[1] or input[1].new()
   self.output[2] = self.output[2] or input[1].new()
   self.output[1]:resizeAs(input[1])
   self.output[2]:resizeAs(input[1])

   if torch.type(input[1]) == 'torch.CudaTensor' then
      local next_c = self.output[1]
      local next_h = self.output[2]
      input[1].oxnn.LSTM12Part2_updateOutput(input[1], input[2], next_c, next_h)
   else
      if not self._cpu then
         self.dim = input[1]:size(2)
         self:cpuVersion()
      end
      self.output = self._cpu:updateOutput(input)
   end
   return self.output
end

function LSTM12Part2:updateGradInput(input, gradOutput)
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[2] = self.gradInput[2] or input[1].new()
   if self.gradInput_inplace_of_input then
      --self.gradInput[1]:set(input[1])
      self.gradInput[2]:set(input[2])
   else
      --self.gradInput[1]:resizeAs(input[1]):copy(input[1])
      self.gradInput[2]:resizeAs(input[2]):copy(input[2])
   end
   self.gradInput[1]:resizeAs(input[1]):copy(input[1])

   if torch.type(input[1]) == 'torch.CudaTensor' then
      input[1].oxnn.LSTM12Part2_updateGradInput(
         self.gradInput[1], self.gradInput[2], gradOutput[1], gradOutput[2])
   else
      self.gradInput = self._cpu:updateGradInput(input, gradOutput)
   end
   return self.gradInput
end

function LSTM12Part2:cpuVersion()
   local prev_c = nn.Identity()()
   local raw_gates = nn.Identity()()

   local dim = assert(self.dim, 'LSTM12Part2 requires self.dim to be set.')
   local aa = nn.View(4*dim):setNumInputDims(2)(
                  nn.Transpose({2,3})(
                     nn.View(dim,4):setNumInputDims(1)(raw_gates)))
   local raw_gates_split = oxnn.NarrowTable(
                              2, {
                                  {1 * dim + 1, 1 * dim},
                                  {0 * dim + 1, 1 * dim},
                                  {3 * dim + 1, 1 * dim},
                                  {2 * dim + 1, 1 * dim},
                                 },
                             --[[copy=]]true, --[[zero/add=]]false)(aa)
   local raw_in_gate, raw_forget_gate, raw_out_gate, raw_input
         = raw_gates_split:split(4)
   local forget_gate = nn.Sigmoid()(raw_forget_gate)
   local out_gate =    nn.Sigmoid()(raw_out_gate)
   local in_gate =     nn.Sigmoid()(raw_in_gate)
   local input =          nn.Tanh()(raw_input)

   local next_c = oxnn.CAddTableNoCopy()({
                                  oxnn.CMulTable2()({forget_gate, prev_c}),
                                  oxnn.CMulTable2()({in_gate, input})})

   local next_h = oxnn.CMulTable2()({out_gate, nn.Tanh()(next_c)})

   local next_state = nn.Identity()({next_c, next_h})

   self._cpu = nn.gModule({prev_c, raw_gates}, {next_state})
   self.modules = { self._cpu }
end

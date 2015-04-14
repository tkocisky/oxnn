local LSTM12Part2, parent = torch.class('oxnn.LSTM12Part2', 'nn.Module')

local test = false
local gradInput_inplace = false

function LSTM12Part2:__init()
   parent.__init(self)
   self.output = {}
   self.gradInput = {}

   --assert(torch.CudaTensor, 'Cuda is required for oxnn.LSTM12Part2 layer.')
end

function LSTM12Part2:cpuVersion()
   local prev_c = nn.Identity()()
   local raw_gates = nn.Identity()()

   local function Sigmoid() return oxnn.cu.use_cuda and cudnn.Sigmoid(true)
                                                    or nn.Sigmoid() end
   local function Tanh() return oxnn.cu.use_cuda and cudnn.Tanh(true)
                                                    or nn.Tanh() end
   local dim = params.worddim
   local aa = nn.View(100,4*dim)(nn.Transpose({2,3})(nn.View(100,dim,4)(raw_gates)))
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
   local forget_gate = Sigmoid()(raw_forget_gate)
   local out_gate =    Sigmoid()(raw_out_gate)
   local in_gate =     Sigmoid()(raw_in_gate)
   local input =          Tanh()(raw_input)

   local next_c = oxnn.CAddTableNoCopy()({
                                  oxnn.CMulTable2()({forget_gate, prev_c}),
                                  oxnn.CMulTable2()({in_gate, input})})

   local next_h = oxnn.CMulTable2()({out_gate, nn.Tanh()(next_c)})

   local next_state = nn.Identity()({next_c, next_h})

   self._cpu = nn.gModule({prev_c, raw_gates}, {next_state}):cuda()
   self.models = { self._cpu }
end

function LSTM12Part2:updateOutput(input)
   self.output[1] = self.output[1] or input[1].new()
   self.output[2] = self.output[2] or input[1].new()
   self.output[1]:resizeAs(input[1])
   self.output[2]:resizeAs(input[1])
   local next_c = self.output[1]
   local next_h = self.output[2]
   input[1].oxnn.LSTM12Part2_updateOutput(input[1], input[2], next_c, next_h)
if test then
   local out = self.aa:updateOutput(input)
   print(out[1]:sum(), self.output[1]:sum())
   print(out[2]:sum(), self.output[2]:sum())
   print(math.abs((out[1] - self.output[1]):sum()))
   print(math.abs((out[2] - self.output[2]):sum()))
   assert(math.abs((out[1] - self.output[1]):sum()) < 2e-4)
   assert(math.abs((out[2] - self.output[2]):sum()) < 2e-4)
   --assert(out[1]:sum() == self.output[1]:sum())
   --assert(out[2]:sum() == self.output[2]:sum())
end
   return self.output
end

function LSTM12Part2:updateGradInput(input, gradOutput)
if test then
   print("go sum", gradOutput[1]:sum(), gradOutput[2]:sum())
end
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[2] = self.gradInput[2] or input[1].new()
   if gradInput_inplace then
      self.gradInput[1]:set(input[1])
      self.gradInput[2]:set(input[2])
   else
      self.gradInput[1]:resizeAs(input[1]):copy(input[1])
      self.gradInput[2]:resizeAs(input[2]):copy(input[2])
   end
   input[1].oxnn.LSTM12Part2_updateGradInput(
   self.gradInput[1], self.gradInput[2],
      gradOutput[1], gradOutput[2]--,
      --self.gradInput[1], self.gradInput[2]
      )
if test then
   --print(self.gradInput[1]:sum(), self.gradInput[1]:sum()/self.gradInput[1]:nElement())
   local gi = self.aa:updateGradInput(input, gradOutput)
   print("1", gi[1]:sum(), self.gradInput[1]:sum())
   print(gi[2]:sum(), self.gradInput[2]:sum())
   print(math.abs((gi[1] - self.gradInput[1]):sum()))
   print(math.abs((gi[2] - self.gradInput[2]):sum()))
   assert(math.abs((gi[1] - self.gradInput[1]):sum()) < 1e-5)
   assert(math.abs((gi[2] - self.gradInput[2]):sum()) < 1e-5)
end
   return self.gradInput
end

-- Authors: Tomas Kocisky
-- (modified copy of nn.Linear)
--
-- Linear layer with block diagonal weight matrix. Blocks are of the same size.
--
local LinearBlockDiagonal, parent = torch.class('oxnn.LinearBlockDiagonal', 'nn.Module')

function LinearBlockDiagonal:__init(inputSize, outputSize, blocks)
   parent.__init(self)

   assert(type(blocks) == 'number')
   assert(inputSize % blocks == 0)
   assert(outputSize % blocks == 0)

   self.weight = torch.Tensor(blocks, outputSize/blocks, inputSize/blocks)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(blocks, outputSize/blocks, inputSize/blocks)
   self.gradBias = torch.Tensor(outputSize)

   self.zeroOutput = true

   self:reset()
end

function LinearBlockDiagonal:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(1)*self.weight:size(3))
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function LinearBlockDiagonal:updateOutput(input)
   assert('input must be a matrix')

   local nframe = input:size(1)
   local nunit = self.bias:size(1)
   local blocks = self.weight:size(1)

   self.output:resize(nframe, nunit)

   if not self.addBuffer or self.addBuffer:size(1) ~= nframe then
      self.addBuffer = input.new(nframe):fill(1)
   end
   if nunit == 1 then
      assert(blocks == 1)
      -- Special case to fix output size of 1 bug:
      self.output:copy(self.bias:view(1,nunit):expand(#self.output))
      self.output:select(2,1):addmv(1, input,
                                    self.weight:view(1,-1):select(1,1))
   else
      if self.zeroOutput then self.output:zero() end

      self.output:addr(1, self.addBuffer, self.bias)

      local input_ = input:view(nframe, blocks, -1):transpose(1,2)
      local output_ = self.output:view(nframe, blocks, -1):transpose(1,2)
      output_:baddbmm(1, input_, self.weight:transpose(2,3))
   end

   return self.output
end

function LinearBlockDiagonal:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end

      local nframe = input:size(1)
      local nunit = self.bias:size(1)
      local blocks = self.weight:size(1)
      local gradOutput_ = gradOutput:view(nframe, blocks, -1):transpose(1,2)
      local gradInput_ = self.gradInput:view(nframe, blocks, -1):transpose(1,2)
      gradInput_:baddbmm(0, 1, gradOutput_, self.weight)

      return self.gradInput
   end
end

function LinearBlockDiagonal:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   local nframe = input:size(1)
   local nunit = self.bias:size(1)
   local blocks = self.weight:size(1)

   if nunit == 1 then
      -- Special case to fix output size of 1 bug:
      self.gradWeight:view(1,-1):select(1,1)
                     :addmv(scale, input:t(), gradOutput:select(2,1))
      self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
   else
      local input_ = input:view(nframe, blocks, -1):transpose(1,2)
      local gradOutput_ = gradOutput:view(nframe, blocks, -1):transpose(1,2)
      self.gradWeight:baddbmm(scale, gradOutput_:transpose(2,3), input_)
      self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
   end
end

-- we do not need to accumulate parameters when sharing
LinearBlockDiagonal.sharedAccUpdateGradParameters = LinearBlockDiagonal.accUpdateGradParameters


function LinearBlockDiagonal:__tostring__()
  return torch.type(self) ..
      string.format('(%dx %d -> %d)', self.weight:size(1), self.weight:size(3), self.weight:size(2))
end

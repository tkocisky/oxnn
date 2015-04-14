-- Authors: Tomas Kocisky
--
-- Given 2 inputs, applies a linear transformation to the second one, and adds
-- the result inplace to the first one.
--
local LinearCAddInplace, parent = torch.class('oxnn.LinearCAddInplace', 'nn.Container')

function LinearCAddInplace:__init(inputSize, outputSize, blocks)
   parent.__init(self)

   if not blocks then
      self.modules = { oxnn.LinearNoOutputZero(inputSize, outputSize) }
   else
      self.modules = { oxnn.LinearBlockDiagonal(inputSize, outputSize, blocks) }
      self.modules[1].zeroOutput = false
   end
   self.linear = self.modules[1]
   self.output = torch.Tensor()
   self.gradInput = {}
end

function LinearCAddInplace:updateOutput(input)
   -- input = { i, prev_h }
   assert(input[1]:dim() == 2, 'Expecting input[1] to be a 2-dim tensor.')
   assert(input[1]:size(1) == input[2]:size(1))
   assert(input[1]:size(2) == self.linear.bias:size(1))
   self.linear.output:set(input[1])
   self.linear:updateOutput(input[2])
   self.output:set(self.linear.output)
   return self.output
end

function LinearCAddInplace:updateGradInput(input, gradOutput)
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[1]:set(gradOutput)
   self.gradInput[2] = self.gradInput[2] or input[2].new()
   self.linear:updateGradInput(input[2], gradOutput)
   self.gradInput[2]:set(self.linear.gradInput)
   return self.gradInput
end

function LinearCAddInplace:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.linear:accGradParameters(input[2], gradOutput, scale)
end

function LinearCAddInplace:accUpdateGradParameters(input, gradOutput, lr)
   assert(false, 'Unimplemented.')
end

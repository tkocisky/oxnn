-- Authors: Tomas Kocisky
--
-- Customizes the nn.LogSoftMax to be computed inplace
--
local LogSoftMaxInplace, parent = torch.class('oxnn.LogSoftMaxInplace', 'nn.LogSoftMax')

function LogSoftMaxInplace:__init(outputInplace, gradInputInplace)
   parent.__init(self)
   self.outputInplace = outputInplace
   self.gradInputInplace = gradInputInplace
end

function LogSoftMaxInplace:updateOutput(input)
   if self.outputInplace then
      assert(input:isContiguous())
      self.output = input
   end
   return input.nn.LogSoftMax_updateOutput(self, input)
end

function LogSoftMaxInplace:updateGradInput(input, gradOutput)
   if self.gradInputInplace then
      assert(gradOutput:isContiguous())
      self.gradInput = gradOutput
   end
   return input.nn.LogSoftMax_updateGradInput(self, input, gradOutput)
end

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
   input.THNN.LogSoftMax_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
   return self.output
end

function LogSoftMaxInplace:updateGradInput(input, gradOutput)
   if self.gradInputInplace then
      assert(gradOutput:isContiguous())
      self.gradInput = gradOutput
   end
   input.THNN.LogSoftMax_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   return self.gradInput
end

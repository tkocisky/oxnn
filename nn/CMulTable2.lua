-- Input is table of two tensors only.

local CMulTable2, parent = torch.class('oxnn.CMulTable2', 'nn.Module')

function CMulTable2:__init()
   parent.__init(self)
   self.gradInput = {}
end

function CMulTable2:updateOutput(input)
   self.output:cmul(input[1], input[2])
   return self.output
end

function CMulTable2:updateGradInput(input, gradOutput)
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[1]:cmul(gradOutput, input[2])
   self.gradInput[2] = self.gradInput[2] or input[1].new()
   self.gradInput[2]:cmul(gradOutput, input[1])
   return self.gradInput
end


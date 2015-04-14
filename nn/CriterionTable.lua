-- Authors: Tomas Kocisky
--
-- Modified nn.CriterionTable to send back gradient to both parts of input
-- table.
--
local CriterionTable, parent = torch.class('oxnn.CriterionTable', 'nn.Module')

function CriterionTable:__init(criterion)
   parent.__init(self)
   self.criterion = criterion
   self.gradInput = {}
end

function CriterionTable:updateOutput(input)
   self.output = self.criterion:updateOutput(unpack(input))
   return self.output
end

function CriterionTable:updateGradInput(input, gradOutput)
   self.gradInput[1] = self.criterion:updateGradInput(unpack(input))
   if input[2] then
      self.gradInput[2] = self.gradInput[2] or torch.Tensor()
      local zl = oxnn.ZeroLoss()
      zl.gradInput = self.gradInput[2]
      zl:updateGradInput(input[2], nil)
      self.gradInput[2] = zl.gradInput
   else
      self.gradInput[2] = nil
   end
   return self.gradInput
end

function CriterionTable:type(type)
   self.criterion:type(type)
   if torch.typename(self.gradInput)
      and torch.typename(self.gradInput):find('torch%..+Tensor') then
      self.gradInput:type(type)
   end
   return self
end

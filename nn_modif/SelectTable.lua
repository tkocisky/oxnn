-- Authors: Tomas Kocisky
--
-- Modifies the nn.SelectTable so that the input can contain a number in the
-- table.
--
local SelectTable, parent = nn.SelectTable, nn.Module

function SelectTable:updateGradInput(input, gradOutput)
   self.gradInput[self.index] = gradOutput
   for i = 1,#input do
      if i ~= self.index then
         oxnn.RecursiveResizeZero(self.gradInput, i, input[i])
      end
   end
   return self.gradInput
end

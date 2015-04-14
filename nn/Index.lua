-- Authors: Tomas Kocisky
--
local Index, parent = torch.class('oxnn.Index', 'nn.Module')

function Index:updateOutput(input)
   local t = input[1]
   local dim = input[2]
   local index = input[3]
   self.output:index(t, dim, index)
   return self.output
end

function Index:updateGradInput(input, gradOutput)
   local t = input[1]
   local dim = input[2]
   local index = input[3]
   self.gradInput:resizeAs(t):zero()
   self.gradInput:indexCopy(dim, index, gradOutput)
   return self.gradInput
end

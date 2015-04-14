-- Authors: Tomas Kocisky
--
-- Modification of nn.JoinTable to handle changing #input across minibatches.
--
local JoinTable, parent = torch.class('oxnn.JoinTable', 'nn.JoinTable')

function JoinTable:updateGradInput(input, gradOutput)
   self._gradInput = self._gradInput or self.gradInput
   self.gradInput = {}

   local dimension = self.dimension
   if self.nInputDims and input[1]:dim()==(self.nInputDims+1) then
       dimension = dimension + 1
   end

   for i=1,#input do
      if self._gradInput[i] == nil then
         self._gradInput[i] = input[i].new()
      end
      self._gradInput[i]:resizeAs(input[i])
      self.gradInput[i] = self._gradInput[i]
   end

   local offset = 1
   for i=1,#input do
      local currentOutput = input[i]
      local currentGradInput = gradOutput:narrow(dimension, offset,
                      currentOutput:size(dimension))
      self.gradInput[i]:copy(currentGradInput)
      offset = offset + currentOutput:size(dimension)
   end
   return self.gradInput
end

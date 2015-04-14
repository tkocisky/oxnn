-- Authors: Tomas Kocisky, Lei Yu
--
-- CAddTable with more memory efficient updateGradInput. We also support when
-- the number of tensors to add changes across minibatches.
--
local CAddTableNoCopy, parent = torch.class('oxnn.CAddTableNoCopy', 'nn.CAddTable')

function CAddTableNoCopy:updateGradInput(input, gradOutput)
   self.gradInput = {}
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or input[1].new()
      self.gradInput[i]:set(gradOutput)
   end
   return self.gradInput
end

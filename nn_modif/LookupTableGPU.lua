-- Modifies some methods of nn.LookupTableGPU from fbcunn.

if nn.LookupTableGPU then  -- from fbcunn

local LookupTableGPU, parent = nn.LookupTableGPU, nn.Module

function LookupTableGPU:zeroGradParameters()
   if not self.accUpdate then
      self.gradWeight:zero()
   end
   self.inputs = {}
   self.nBackward = 0
end

function LookupTableGPU:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or input.new()
   self.gradInput = self.gradInput:typeAs(input)
   self.gradInput:resize(input:size()):zero()
   return self.gradInput
end

function LookupTableGPU:type(type)
   self._indices = nil
   self._inputView = nil
   return parent.type(self, type)
end

-- from nn.LookupTable
-- since self.weight:normal(stdv) in fbcunn seems to be not the same
-- as self.weight:normal(0, stdv) in nn.
function LookupTableGPU:reset(stdv)
   stdv = stdv or 1
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.normal(0, stdv)
      end)
   else
      self.weight:normal(0, stdv)
   end
end

end

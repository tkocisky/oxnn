local LookupTable, parent = nn.LookupTable, nn.Module

function LookupTable:zeroGradParameters()
   if not self.accUpdate then
      self.gradWeight:zero()
   end
   self.inputs = {}
   self.nBackward = 0
end

function LookupTable:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or input.new()
   self.gradInput = self.gradInput:typeAs(input)
   self.gradInput:resize(input:size()):zero()
   return self.gradInput
end

function LookupTable:type(type)
   self._indices = nil
   self._inputView = nil
   return parent.type(self, type)
end

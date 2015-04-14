-- Modification for getParamsByDevice branch of nn
local MM, parent = torch.getmetatable('nn.MM'), torch.getmetatable('nn.Module')

function MM:type(type)
   self.gradInput[1] = self.gradInput[1]:type(type)
   self.gradInput[2] = self.gradInput[2]:type(type)
   parent.type(self, type)
   return self
end

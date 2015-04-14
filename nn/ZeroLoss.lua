-- Authors: Tomas Kocisky
--
local ZeroLoss, parent = torch.class('oxnn.ZeroLoss', 'nn.Criterion')

function ZeroLoss:__init()
   parent.__init(self)
end

function ZeroLoss:updateOutput(input)
   self.output = 0
   return self.output
end

function ZeroLoss:updateGradInput(input, target)
   if torch.typename(input)
         and torch.typename(input):find('torch%..+Tensor') then
      self.gradInput = self.gradInput:typeAs(input)
      self.gradInput:resizeAs(input):zero()
   elseif torch.type(input) == 'table' then
      if torch.type(self.gradInput) ~= 'table' then
         self.gradInput = {}
      end
      for i = 1,#input do
         oxnn.RecursiveResizeZero(self.gradInput, i, input[i])
      end
   elseif torch.type(input) == 'number' then
      self.gradInput = 0
   else
      error('this shouldn\'t happen')
   end
   return self.gradInput
end

function ZeroLoss:type(type)
   self.gradInput = self.gradInput:type(type)
   return self
end

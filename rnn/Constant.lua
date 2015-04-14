local Constant, parent = torch.class('oxnn.Constant', 'nn.Module')

function Constant:__init(c)
   parent.__init(self)
   self.output = c
   self.gradInput = {}
end

-- Authors: Tomas Kocisky
--
-- Disables the parameter update of a module.
--
local NoAccGradParameters, parent = torch.class('oxnn.NoAccGradParameters', 'nn.Sequential')

function NoAccGradParameters:__init(module)
   parent.__init(self)
   self:add(module)
   self.noAccGradParameters = true
end

function NoAccGradParameters:accGradParameters(input, gradOutput, scale)
   if not self.noAccGradParameters then
      parent.accGradParameters(self, input, gradOutput, scale)
   end
end

function NoAccGradParameters:accUpdateGradParameters(input, gradOutput, lr)
   if not self.noAccGradParameters then
      parent.accUpdateGradParameters(self, input, gradOutput, lr)
   end
end

function NoAccGradParameters:sharedAccUpdateGradParameters(input, gradOutput, lr)
   if not self.noAccGradParameters then
      parent.sharedAccUpdateGradParameters(self, input, gradOutput, lr)
   end
end

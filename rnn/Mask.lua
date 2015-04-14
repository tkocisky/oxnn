-- Authors: Tomas Kocisky
--
-- Multiplies the input by a 0/1 mask along the examples in the minibatch
-- (zeroing out some of the examples).
--
local Mask, parent = torch.class('oxnn.Mask', 'nn.Module')

function Mask:__init(mask)
   parent.__init(self)
   self.mask = mask
end

function Mask.mask(t, mask)  -- does not modify t
   if type(t) == 'table' then
      local res = {}
      for k,v in pairs(t) do
         res[k] = Mask.mask(v, mask)
      end
      return res
   end
   assert(t:size(1) == mask:size(1))
   local t_ = t:view(t:size(1),-1)
   local mask_ = mask:view(mask:size(1),-1):expandAs(t_)
   local res = torch.cmul(t_, mask:view(mask:size(1),-1):expandAs(t_))
   return res:viewAs(t)
end
function Mask.add_mask(res, t, mask)  -- does not modify t
   if type(t) == 'table' then
      for k,v in pairs(t) do
         Mask.add_mask(res[k], v, mask)
      end
      return
   end
   assert(t:size(1) == mask:size(1))
   local t_ = t:view(t:size(1),-1)
   local mask_ = mask:view(mask:size(1),-1):expandAs(t_)
   res:viewAs(t_):addcmul(t_, mask:view(mask:size(1),-1):expandAs(t_))
   return
end

function Mask:updateOutput(input)
   self.output = oxnn.recursiveClone(input,
                                     function(t)
                                        return Mask.mask(t, self.mask)
                                     end)
   return self.output
end

function Mask:updateGradInput(input, gradOutput)
   self.gradInput = oxnn.recursiveClone(gradOutput,
                                        function(t)
                                           return Mask.mask(t, self.mask)
                                        end)
   return self.gradInput
end

function Mask:type(type)
   self.output = self.output:type(type)
   self.gradInput = self.gradInput:type(type)
   return self
end

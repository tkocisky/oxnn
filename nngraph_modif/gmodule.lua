-- Authors: Tomas Kocisky

-- fix for nn.Module.reset propagation
function nn.gModule:reset(stdv)
   self:apply(function(module) module:reset(stdv) end)
   return self
end

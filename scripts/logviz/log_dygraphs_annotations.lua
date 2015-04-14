
--require 'env'
require 'gnuplot'
local file = require('pl.file')
local stringx = require('pl.stringx')
local tablex = require('pl.tablex')

local log_file = assert(arg[1], 'Expecting log file as the first argument.')

local out = {}

local last_it = nil
for line in io.lines(log_file) do
   if line:match('It:%d+ P ') then
   elseif line:match('It:%d+ F ') then
      local it = line:match('It:(%d+)')
      if it ~= last_it then
         local a = '{ series: \'train avg\', attachAtBottom: true, '
         local l = stringx.split(line)
         a = a .. 'x: \'' .. l[1]..' '..l[2] ..'\','
         a = a .. 'shortText: \'' .. it ..'\','
         a = a .. 'text: \'' .. line .. '\','
         a = a .. ' }'
         table.insert(out, a)
         last_it = it
      end
   elseif line:match('EV:[^ ]+ F ') then
   end
end

for i,v in ipairs(out) do
   print(v)
   if i ~= #out then print(',') end
end


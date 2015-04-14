
--require 'env'
require 'gnuplot'
local file = require('pl.file')
local stringx = require('pl.stringx')
local tablex = require('pl.tablex')

local log_file = assert(arg[1], 'Expecting log file as the first argument.')

local s = {}

local function parse_time(t)
   if t:match('^%d+ms$') then
      t = t:match('%d+')/1000
   elseif t:match('^%d+s%d+ms$') then
      local a,b = t:match('(%d+).(%d+).')
      t = a + b / 1000
   elseif t:match('^%d+s$') then
      local a = t:match('(%d+).')
      t = a
   elseif t:match('^%d+m%d+s$') then
      local a,b = t:match('(%d+).(%d+).')
      t = a * 60 + b
   elseif t:match('^%d+m$') then
      local a = t:match('(%d+).')
      t = a * 60
   elseif t:match('^%d+h%d+m$') then
      local a,b = t:match('(%d+).(%d+).')
      t = a * 60 * 60 + b * 60
   else
      error('Unimplemented case in parse_time.')
   end
   return t
end

local function parse1(name, line, time)
   s[name] = s[name] or {}
   s[name].x = s[name].x or {}
   s[name].y = s[name].y or {}
   local l = stringx.split(line)
   local time_ = time or l[1]..' '..l[2]
   local t = stringx.split(l[14], '/')[1]
   t = parse_time(t)
   table.insert(s[name].x, time_)
   table.insert(s[name].y, t)
end

for line in io.lines(log_file) do
   if line:match('It:%d+ P ') then
      local name = 's/minibatch'
      parse1(name, line)
   elseif line:match('It:%d+ F ') then
   elseif line:match('EV:[^ ]+ F ') then
   end
end

local function mypairs(t)
   local k = {}
   for k_ in pairs(t) do
      if not tablex.find(k, k_) then
         table.insert(k, k_)
      end
   end
   local idx = 1
   return function(_s, _var)
      if k[idx] then
         idx = idx + 1
         return k[idx-1], t[k[idx-1]]
      else
         return nil
      end
   end
end

local out = {}

local first = true
local keys = {'Time'}
for k,v in mypairs(s) do
   table.insert(keys, k)
end
table.insert(out, table.concat(keys, ','))

local function out_map_def()
   local res = {}
   for i=1,#keys-1 do res[i] = '' end
   return res
end
local out_map = {}
idx = 1
for k,v in mypairs(s) do
   for i=1,#v.x do
      out_map[v.x[i]] = out_map[v.x[i]] or out_map_def()
      out_map[v.x[i]][idx] = string.format("%g", v.y[i])
   end
   idx = idx + 1
end
for k,v in pairs(out_map) do
   table.insert(out, k..','..table.concat(v, ','))
end

for i,v in ipairs(out) do
   print(v)
end


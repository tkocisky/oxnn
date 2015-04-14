
--require 'env'
require 'gnuplot'
local file = require('pl.file')
local stringx = require('pl.stringx')
local tablex = require('pl.tablex')

local log_file = assert(arg[1], 'Expecting log file as the first argument.')

local s = {}

local function parse1(name, line, time)
   s[name] = s[name] or {}
   s[name].x = s[name].x or {}
   s[name].y = s[name].y or {}
   local l = stringx.split(line)
   local time_ = time or l[1]..' '..l[2]
   local perp = l[7]:match('%d+.?%d+')
   table.insert(s[name].x, time_)
   table.insert(s[name].y, perp)
end
local function parse2(name, line)
   s[name] = s[name] or {}
   s[name].x = s[name].x or {}
   s[name].y = s[name].y or {}
   local l = stringx.split(line)
   local time = l[1]..' '..l[2]
   local perp = l[8]:match('%d+.?%d+')
   table.insert(s[name].x, time)
   table.insert(s[name].y, perp)
end
local function parse3(name, line)
   s[name] = s[name] or {}
   s[name].x = s[name].x or {}
   s[name].y = s[name].y or {}
   local l = stringx.split(line)
   local time = l[1]..' '..l[2]
   local perp = l[10]:match('%d+.?%d+')
   table.insert(s[name].x, time)
   table.insert(s[name].y, perp)
end

local sync_time = nil
for line in io.lines(log_file) do
   if line:match('It:%d+ P ') then
      local name = 'train'
      parse2(name, line)
      local name = 'train avg partial'
      parse3(name, line)
   elseif line:match('It:%d+ F ') then
      local name = 'train avg'
      local l = stringx.split(line)
      sync_time = l[1]..' '..l[2]
      parse1(name, line, sync_time)
   elseif line:match('EV:[^ ]+ F ') then
      local name = stringx.split(line)[4]:match('EV:([^ ]+)')
      parse1(name, line, sync_time)
   end
end

local function mypairs(t)
   local k = { 'train', 'train avg partial' }
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


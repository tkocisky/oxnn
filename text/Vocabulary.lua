-- Authors: Tomas Kocisky
--
-- Vocabulary to map words to ids.
--
local Vocabulary = torch.class('oxnn.Vocabulary')

function Vocabulary:__init(words)
   self.id2word = {}
   self.word2id = {}
   if words then for _,w in ipairs(words) do self:Insert(w) end end
end

function Vocabulary:WordToId(word)
   return self.word2id[word]
end

function Vocabulary:IdToWord(id)
   return self.id2word[id]
end

function Vocabulary:size()
   return #self.id2word
end

function Vocabulary:Insert(word)
   assert(type(word) == 'string',
          'Expecting a string word to insert but given \'' .. word .. '\'.')
   if self.word2id[word] then
      error('Word \'' .. word .. '\' is already in vocabulary.')
   end
   self.id2word[#self.id2word + 1] = word
   self.word2id[word] = #self.id2word
   return self.word2id[word]
end

function Vocabulary:__tostring()
   local buf = {}
   buf[#buf + 1] = '{\n'
   for id,word in ipairs(self.id2word) do
      buf[#buf + 1] = '   '
      buf[#buf + 1] = tostring(id)
      buf[#buf + 1] = ' : '
      buf[#buf + 1] = word
      buf[#buf + 1] = '\n'
   end
   buf[#buf + 1] = '}\n'
   return table.concat(buf, '')
end


---- Static functions ----

-- Converts a sequence of (sequences of) strings to ids.
--
-- If the a word is not in the vocabulary, calls unk_callback(voc, v).
-- If given id2freq creates a frequency table.
function Vocabulary.WordsToIds(words, voc, unk_callback, id2freq)
   if not unk_callback then unk_callback = Vocabulary.unk_callback_error end
   local inc_freq = function (id)
      if id2freq then
         if not id2freq[id] then id2freq[id] = 1
         else id2freq[id] = id2freq[id] + 1 end
       end
   end

   local result = {}
   for i,v in ipairs(words) do
      if type(v) == 'string' then
         result[i] = voc:WordToId(v) or unk_callback(voc, v)
         inc_freq(result[i])
      elseif type(v) == 'table' then
         result[i] = Vocabulary.WordsToIds(v, voc, unk_callback, id2freq)
      else
         error('Expecting a sequence of strings or a sequence of sequences ' ..
               'of strings.')
      end
   end
   return result
end

function Vocabulary.unk_callback_insert(vocabulary, word)
   return vocabulary:Insert(word)
end

function Vocabulary.unk_callback_error(vocabulary, word)
   error('Word \'' .. word .. '\' is not in vocabulary.')
end

function Vocabulary.IdsToWords(voc, ids)
   local result = {}
   for i,v in ipairs(ids) do
      result[i] = assert(voc:IdToWord(v))
   end
   return result
end

-- Returns a new vocabulary containing only words with freq >= min_freq.
function Vocabulary.CutoffVocabulary(voc, id2freq, min_freq)
   local keep = {}
   for id,freq in pairs(id2freq) do
      if freq >= min_freq then
         keep[#keep + 1] = id
      end
   end
   print('Words to keep after cut off', #keep, 'cut off', min_freq)

   local new_voc = oxnn.Vocabulary()
   for _,id in pairs(keep) do
      new_voc:Insert(voc:IdToWord(id))
   end
   return new_voc
end

-- Returns a new vocabulary containing only max_size top frequency words.
--
-- The choice between words of equal freqency around the max_size limit is
-- arbitrary.
function Vocabulary.LimitVocabulary(voc, id2freq, max_size)
   local freq2id = {}
   for id,freq in pairs(id2freq) do
      freq2id[#freq2id + 1] = { freq=freq, id=id }
   end
   table.sort(freq2id, function (a,b) return a.freq > b.freq end)

   local new_voc = oxnn.Vocabulary()
   for i,pair in ipairs(freq2id) do
      new_voc:Insert(voc:IdToWord(pair.id))
      if i >= max_size then
         print('Frequency of last included word is', pair.freq, 'max size', max_size)
         break
      end
   end
   return new_voc
end

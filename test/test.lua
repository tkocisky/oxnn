-- run with:
--    th -loxnn -e "oxnn.test()"   # runs all tests
--    th -loxnn -e "oxnn.test{'oxnn_SequenceOfWords'}"  # run tests with given
--                                                      # prefix
--    th -loxnn -e "oxnn.InitCuda() oxnn.test()" 2>&1  # run all tests on
--                                                     # gpu

oxnn.tests = {}
oxnn.mytester = torch.Tester()

torch.include('oxnn', 'test/test_ModelUtil.lua')
torch.include('oxnn', 'test/test_LinearBlockDiagonal.lua')
torch.include('oxnn', 'test/test_SequenceOfWords.lua')

oxnn.mytester:add(oxnn.tests)

function oxnn.test(tests)  -- tests is a table of name _prefixes_
   math.randomseed(os.time())
   if tests then  -- expand prefixes
      local new_tests = {}
      for _,pref in ipairs(tests) do
         for _,v in ipairs(oxnn.mytester.testnames) do
            local first,_ = string.find(v, pref)
            if first and first == 1 then
               new_tests[#new_tests+1] = v
            end
         end
      end
      tests = new_tests
   end
   oxnn.mytester:run(tests)
   return oxnn.mytester
end

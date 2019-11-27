import Test.Tasty (defaultMain, testGroup)
import Test.Tasty.HUnit (assertEqual, testCase, (@?=))
import qualified Perceptron as P

main = defaultMain unitTests

unitTests =
  testGroup
    "Unit tests"
    [dotProductTest, dotProductTestNegative, orPerceptron, checkPredictionTest,
     determineWeights]

dotProductTest =
  let vector = [1, 2, 3]
      product = P.dotProduct vector vector
  in testCase "Dot product of (1, 2, 3)' * (1 2 3) = 14" $ assertEqual [] 14 product 

dotProductTestNegative =
  let vector1 = [1, 2, 3]
      vector2 = [1, -2, 3]
      product = P.dotProduct vector1 vector2
  in testCase "Dot product of (1, 2, 3)' * (1 -2 3) = 6" $ assertEqual [] 6 product 

orPerceptron =
  let td = [P.TD [0.0,0.0] 0, P.TD [1.0, 0.0] 1, P.TD [0.0, 1.0] 1, P.TD [1.0, 1.0] 1]
      weights = [0.0, 0.0, 0.0]
      (finalWeights, result) = P.trainPerceptron weights 1 10 td
  in testCase "train OR perceptron" $ do finalWeights @?= [-1.0, 1.0, 1.0]
                                         result @?= True
                                         P.checkPrediction finalWeights td @?= True

sampleData = [P.TD [2.7810836,2.550537003] 0,
              P.TD [1.465489372,2.362125076] 0,
              P.TD [3.396561688,4.400293529] 0,
              P.TD [1.38807019,1.850220317] 0,
              P.TD [3.06407232,3.005305973] 0,
              P.TD [7.627531214,2.759262235] 1,
              P.TD [5.332441248,2.088626775] 1,
              P.TD [6.922596716,1.77106367] 1,
              P.TD [8.675418651,-0.242068655] 1,
              P.TD [7.673756466,3.508563011] 1]

checkPredictionTest = let weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
                      in testCase "checkPrediction" $ P.checkPrediction weights sampleData @?= True
                          

determineWeights = let weights = [0.0, 0.0, 0.0]
                       (finalWeights, result) = P.trainPerceptron weights 0.01 10 sampleData 
                   in testCase "determine weights test" $ 
                     do finalWeights @?= [-0.01, 0.020653640140000002, -0.023418117710000003]
                        result @?= True
                        P.checkPrediction finalWeights sampleData @?= True

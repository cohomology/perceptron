import Test.Tasty (defaultMain, testGroup)
import Test.Tasty.HUnit (assertEqual, testCase)
import qualified Perceptron as P

main = defaultMain unitTests

unitTests =
  testGroup
    "Unit tests"
    [dotProductTest, dotProductTestNegative]

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
      (finalWeights, result) = trainPerceptron weights 0.01 10 td
  in testCase "OR perceptron" $ assertEqual [] [1.0, 1.0, 1.0] finalWeights 

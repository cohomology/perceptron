module Perceptron (
  dotProduct, 
  predictPerceptron
) where    

import Data.List (foldl')

data Num a => TrainingData a = TD {
    trainingValues :: [a],
    trainingActivation :: Int 
  } deriving (Show, Eq)

-- |Computes the dot product v1 * v2 for vectors v1, v2.
dotProduct :: Num a =>    [a]  -- ^ First vector 
                       -> [a]  -- ^ Second vector
                       -> a    -- ^ v1 . v2 
dotProduct v1 v2 = sum (zipWith (*) v1 v2)

-- |Basic predictor function for single perceptron
predictPerceptron :: (Num a, Ord a) =>    [a]  -- ^ Weight vector 
                                       -> [a]  -- ^ Value vector 
                                       -> a    -- ^ Result (0 or 1)
predictPerceptron weights values 
  | dotProduct weights values >= 0 = 1
  | otherwise                      = 0

-- |Check if a single training datum satisfies the prediction
checkSinglePrediction :: (Num a, Ord a) =>    [a]             -- ^ Weights
                                           -> TrainingData a  -- ^ Training data
                                           -> Bool           
checkSinglePrediction weights (TD v a) = 
  predictPerceptron weights v == fromIntegral a

-- |Check if the prediction is correct for the complete training data set 
checkPrediction :: (Num a, Ord a) =>    [a]              -- ^ Weights
                                     -> [TrainingData a] -- ^ Training data set
                                     -> Bool
checkPrediction weights = all (checkSinglePrediction weights)

-- |Update weights (single step in perceptron algorithm)
updateWeights :: (Num a, Ord a) =>    a               -- ^ learning rate
                                   -> [a]             -- ^ weights
                                   -> TrainingData a  -- ^ training data 
                                   -> [a]             -- ^ updated weights 
updateWeights trainingRate weights input = updatedWeights
  where values = trainingValues input 
        activation = fromIntegral $ trainingActivation input 
        prediction  = predictPerceptron weights (1 : values) 
        correctionFactor = trainingRate * (activation - prediction)
        correctionVector = correctionFactor : map (correctionFactor *) values
        updatedWeights = zipWith (+) weights correctionVector

-- |Train perceptron until the training data is correctly recognized.
trainPerceptron :: (Num a, Ord a) =>   [a]                     -- ^ initial weights
                                     -> a                       -- ^ learning rate
                                     -> Int                     -- ^ max recursion
                                     -> [TrainingData a]        -- ^ training set
                                     -> ([a], Bool)             -- ^ result weights
trainPerceptron weights    _ 0     input   = (weights, checkPrediction weights input)
trainPerceptron weights rate epoch input  = 
  let updatedWeights = foldl' (updateWeights rate) weights input
  in if checkPrediction updatedWeights input 
        then (updatedWeights, True)
        else trainPerceptron updatedWeights rate (epoch - 1) input

package br.puc.drm.ranker;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class RankEvaluation {

	private List<List<Integer>> resultSet;
	private List<List<List<Integer>>> resultSetForCrossValidation;
	private Double totalScore;
	private Double maxScoreAvg;
	private long trainElapsedTime;
	private long testElapsedTime;
	private long trainElapsedTimeAvg;
	private long testElapsedTimeAvg;
	private Integer[] kAccuracy;
	private Integer[] kAccuracyAvg;

	//Generates all relevant statistics about the result and stores internally
	private void generatePerformanceStatistics() {
		
		Double totalScore = 0.0;
		
		for (List<Integer> list : resultSet) {
			
			Integer actualClass = list.get(0);
			Double position = list.subList(1, list.size()-1).indexOf(actualClass) + 1.0;
			
			if (list.subList(1, list.size()-1).indexOf(actualClass) == -1) {
				position = (double) list.size();
			}
			
			Double score = (list.size() - position) / (list.size()-1);
			totalScore += score;
		}
		
		this.totalScore = totalScore;
		this.maxScoreAvg = (double) resultSet.size();
	}

	//Generates all relevant statistics about the result for cross validation and stores internally
	private void generateCrossValidationPerformanceStatistics(Integer k) {

		Double maxScore = 0.0;
		kAccuracy = new Integer[k];
		Integer numFolds = resultSetForCrossValidation.size();
		
		for (List<List<Integer>> oneFoldResultSet : resultSetForCrossValidation) {
			for (List<Integer> list : oneFoldResultSet) {
				
				//The actual class is the first element of the list
				Integer actualClass = list.get(0);
				
				//Position considering the first equal to 1
				Integer position = list.subList(1, list.size()).indexOf(actualClass) + 1;
				
				//If the actual class is not on the list
				if (position == 0) {
					
					position = Integer.MAX_VALUE;
				}
				
				//Generates k Accuracy
				for (int i = 0; i < kAccuracy.length; i++) {
					
					if (position <= i+1) {
						
						if (kAccuracy[i] == null) {
							
							kAccuracy[i] = 1;
									
						} else {
							
							kAccuracy[i]++;
						}
					}
				}
			}
			
			maxScore += oneFoldResultSet.size();
		}
		
		//Calculates the final statistics
		
		kAccuracyAvg = new Integer[k];
		for (int i = 0; i < kAccuracy.length; i++) {
			
			kAccuracyAvg[i] = kAccuracy[i] / numFolds;
			
		}
		
		maxScoreAvg = maxScore / numFolds;
		trainElapsedTimeAvg = trainElapsedTime / numFolds;
		testElapsedTimeAvg = testElapsedTime / numFolds;
	}
	
	/**
	 * Evaluates a model of the MetaRanker class.
	 * 
	 * @param mr A instance of MetaRanker
	 * @param data The test dataset
	 * @return A summary string of the result's performance
	 */
	public void evaluateRankModel(MetaRanker mr, Instances data) {

		if (mr == null) {
			
			throw new IllegalArgumentException("Invalid classifier.");
			
		}
		
		if (data == null) {
			
			throw new IllegalArgumentException("Invalid data.");
		}
		
		//List where the raw result is stored
		this.resultSet = new ArrayList<List<Integer>>();
		
		//Iterates over the data to classify the instances 
		for (Instance instance : data) {
			
			List<Integer> result = new ArrayList<Integer>();
			
			//Stores the actual class value on the first position (zero) and the ranked list on the next positions
			result.add((int) instance.classValue() + 1);
			result.addAll(mr.classifyInstance(instance));
			
			this.resultSet.add(result);
		}
		
		this.generatePerformanceStatistics();
	}

	/**
	 * Evaluates a model of a Classifier class.
	 * 
	 * @param cls A instance of a Classifier
	 * @param data The test dataset
	 * @param rankSize Size of the rank (must be lass or equal to the number of class values)
	 * @return A summary string of the result's performance
	 */
	public void evaluateRankModel(Classifier cls, Instances data, Integer rankSize) {

		if (cls == null) {
			
			throw new IllegalArgumentException("Invalid classifier.");
			
		}
		
		if (data == null) {
			
			throw new IllegalArgumentException("Invalid data.");
		}
		
		//List where the raw result is stored
		this.resultSet = new ArrayList<List<Integer>>();
		
		try {
			
			//Iterates over the data to classify the instances 
			for (Instance instance : data) {
				
				List<Integer> result = new ArrayList<Integer>();
				
				//Stores the actual class value on the first position (zero)
				result.add((int) instance.classValue() + 1);
				
				//Gets the class distribution and prepares to rank the class indexes
				double[] probDist = cls.distributionForInstance(instance);
				double[] sortedProbDist = probDist.clone();		
				Arrays.sort(sortedProbDist);
				
				//Stores the ranked list on the next positions
				for (int i = sortedProbDist.length-1; i > -1 ; i--) {					
					for (int j = 0; j < probDist.length; j++) {
						if (probDist[j] == sortedProbDist[i]) {
							result.add(j+1);
							probDist[j] = -1;
						}
					}
				}				

				//Verify if the rank size is set, uses the number of class values instead
				Integer finalRankSize = rankSize;
				if (finalRankSize == null) {
					finalRankSize = probDist.length;
				}

				List<Integer> trimmedResult = new ArrayList<Integer>();
				for (int i = 0; i <= finalRankSize; i++) {
					trimmedResult.add(result.get(i));
				}
				
				this.resultSet.add(trimmedResult);			
			}

			this.generatePerformanceStatistics();
			
		} catch (Exception e) {
			
			throw new IllegalStateException("Unable to evaluate model.");
		}		
	}

	/**
	 * Evaluates with cross validation a model of the MetaRanker class.
	 * 
	 * @param mr A instance of MetaRanker
	 * @param cls A instance of a Classifier to be used by the MetaRanker
	 * @param data The complete dataset
	 * @param numFolds Number of folds for cross validation
	 * @param random A randon number generator
	 * @return
	 */
	public String crossValidateRankModel(MetaRanker mr, Classifier cls, String classifierOptions, Instances data, Integer numFolds, Random random) {
	
		//Argument validation
		
		if (mr == null) {
			
			throw new IllegalArgumentException("MetaRanker is null.");
			
		}
		
		if (cls == null) {
			
			throw new IllegalArgumentException("Classifier is null.");
		}
		
		if (data == null) {
			
			throw new IllegalArgumentException("Data is null.");
			
		}
		
		if (data.classIndex() < 0) {
			
			throw new IllegalArgumentException("The class is not set.");
			
		}
		
		if (!data.classAttribute().isNominal()) {
			
			throw new IllegalArgumentException("The class is not nominal.");
			
		}
		
		if (numFolds == null) {
			
			throw new IllegalArgumentException("Number of folds is null.");
			
		}
		
		if (numFolds < 2) {
			
			throw new IllegalArgumentException("Number of folds can't be less than 2.");
			
		}
		
		if (numFolds > data.numInstances()) {
			
			throw new IllegalArgumentException("Number of folds can't be greater then the number of instances.");
			
		}
		
		if (random == null) {
			
			throw new IllegalArgumentException("Random number generator is null.");
		}

		Instances randData = new Instances(data);   // create a copy of the original data
		randData.randomize(random);                 // randomize data with number generator		 
		randData.stratify(numFolds);				// stratify the data to enable cross validation
		
		this.resultSetForCrossValidation = new ArrayList<List<List<Integer>>>();
		this.testElapsedTime = 0;
		this.trainElapsedTime = 0;
		
		//Perform the cross validation
		for (int n = 0; n < numFolds; n++) {
			
			//Generate train and test sets
			Instances train = randData.trainCV(numFolds, n);
			Instances test = randData.testCV(numFolds, n);
			
			long startTime = System.nanoTime();
			mr.buildClassifier(cls, train, classifierOptions);
			this.trainElapsedTime += System.nanoTime() - startTime;
			
			startTime = System.nanoTime();
			this.evaluateRankModel(mr, test);
			this.testElapsedTime += System.nanoTime() - startTime;
			
			List<List<Integer>> tmpResultSet = new ArrayList<List<Integer>>();
			tmpResultSet.addAll(this.resultSet);
			this.resultSetForCrossValidation.add(tmpResultSet);
			  
		}
		
		this.generateCrossValidationPerformanceStatistics(5);
		
		return this.toSummaryString();

	}

	public String crossValidateRankModel(Classifier classifier, Instances data, Integer numFolds, Random random, Integer rankSize) {
		
		
		//Argument validation
		
		if (classifier == null) {
			
			throw new IllegalArgumentException("MetaRanker is null.");
			
		}
		
		if (data == null) {
			
			throw new IllegalArgumentException("Data is null.");
			
		}
		
		if (data.classIndex() < 0) {
			
			throw new IllegalArgumentException("The class is not set.");
			
		}
		
		if (!data.classAttribute().isNominal()) {
			
			throw new IllegalArgumentException("The class is not nominal.");
			
		}
		
		if (numFolds == null) {
			
			throw new IllegalArgumentException("Number of folds is null.");
			
		}
		
		if (numFolds < 2) {
			
			throw new IllegalArgumentException("Number of folds can't be less than 2.");
			
		}
		
		if (numFolds > data.numInstances()) {
			
			throw new IllegalArgumentException("Number of folds can't be greater then the number of instances.");
			
		}
		
		if (random == null) {
			
			throw new IllegalArgumentException("Random number generator is null.");
		}

		Instances randData = new Instances(data);   // create a copy of the original data
		randData.randomize(random);                 // randomize data with number generator		 
		randData.stratify(numFolds);				// stratify the data to enable cross validation
		
		this.resultSetForCrossValidation = new ArrayList<List<List<Integer>>>();
		this.testElapsedTime = 0;
		this.trainElapsedTime = 0;
		
		//Perform the cross validation
		for (int n = 0; n < numFolds; n++) {
			
			//Generate train and test sets
			Instances train = randData.trainCV(numFolds, n);
			Instances test = randData.testCV(numFolds, n);
			
			try {
				
				long startTime = System.nanoTime();
				classifier.buildClassifier(train);
				this.trainElapsedTime += System.nanoTime() - startTime;
				
				startTime = System.nanoTime();				
				this.evaluateRankModel(classifier, test, rankSize);
				this.testElapsedTime += System.nanoTime() - startTime;
				
				List<List<Integer>> tmpResultSet = new ArrayList<List<Integer>>();
				tmpResultSet.addAll(this.resultSet);
				this.resultSetForCrossValidation.add(tmpResultSet);
				
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}			  
		}
		
		this.generateCrossValidationPerformanceStatistics(5);
		
		return this.toSummaryString();
		
	}
	
	/**
	 * Outputs the performance statistics in summary form
	 * 
	 * @return A summary string of the result's performance
	 */
	public String toSummaryString() {
	
		String ret = "Average k-accuracies: ";
		
		if (kAccuracyAvg != null && maxScoreAvg != null) {			
		
			for (int i = 0; i < kAccuracyAvg.length; i++) {
				
				ret = ret + kAccuracyAvg[i] + " (" + (kAccuracyAvg[i]/maxScoreAvg)*100.0 + " %) " ;
				
			}
			
			ret = ret + "of maximum " + maxScoreAvg +  " - Train and test elapsed times (miliseconds): " + trainElapsedTimeAvg / 1000000.0 + " | " + testElapsedTimeAvg / 1000000.0;
			
		} else {
			
			ret = null;
		}
		
		return ret;
	}
	
	public String toCSVLine() {
		
		String ret = "";
		
		for (int i = 0; i < kAccuracyAvg.length; i++) {
			
			ret = ret + kAccuracyAvg[i] + ", " + (kAccuracyAvg[i]/maxScoreAvg)*100.0 + ", " ;
			
		}
		
		ret = ret + maxScoreAvg + ", " + trainElapsedTimeAvg / 1000000.0 + ", " + testElapsedTimeAvg / 1000000.0;

		
		return ret;
	}

}

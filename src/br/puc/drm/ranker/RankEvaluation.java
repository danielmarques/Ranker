package br.puc.drm.ranker;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
	private Double[] kAccuracy;
	private Double[] kAccuracyAvg;
	private List<List<Double>> resultSetProdDist;
	private List<List<List<Double>>> resultSetProbDistForCrossValidation;
	private long maxExperimentTime = -1;
	private Map<Integer, Map<Integer, Map<String, Double>>> allLabelsKMetrics;
	private Double[] kPrecisionMicro;
	private Double[] kPrecisionAvg;
	private Double[] kPrecisionPon;
	private Double[] kRecallMicro;
	private Double[] kRecallAvg;
	private Double[] kRecallPon;

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

	//Generates all relevant statistics about the result of cross validation for metaranker and stores internally
	private void generateCrossValidationPerformanceStatisticsForMetaranker(Integer k) {

		Double maxScore = 0.0;
		kAccuracy = new Double[k];
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
							
							kAccuracy[i] = 1.0;
									
						} else {
							
							kAccuracy[i]++;
						}
					}
				}
			}
			
			maxScore += oneFoldResultSet.size();
		}
		
		//Calculates the final statistics
		
		kAccuracyAvg = new Double[k];
		for (int i = 0; i < kAccuracy.length; i++) {
			
			kAccuracyAvg[i] = kAccuracy[i] / numFolds;
			
		}
		
		maxScoreAvg = maxScore / numFolds;
		trainElapsedTimeAvg = trainElapsedTime / numFolds;
		testElapsedTimeAvg = testElapsedTime / numFolds;
	}

	//Generates all relevant statistics about the result of cross validation for classifier and stores internally
	private void generateCrossValidationPerformanceStatisticsForClassifier(Integer k) {

		Double maxScore = 0.0;
		kAccuracy = new Double[k];
		Integer numFolds = resultSetForCrossValidation.size();
		
		for (int i = 0; i < resultSetForCrossValidation.size(); i++) {
			
			maxScore += resultSetForCrossValidation.get(i).size();
			
			for (int j = 0; j < resultSetForCrossValidation.get(i).size(); j++) {				
				
				List<Integer> retList = resultSetForCrossValidation.get(i).get(j);							
				
				//The actual class is the first element of the list
				Integer actualClass = retList.get(0);
				
				//Position considering the first equal to 0
				Integer actualClassPosition = retList.subList(1, retList.size()).indexOf(actualClass);
				
				if (actualClassPosition > -1) {
					//Probabilities of classes
					List<Double> probDist = resultSetProbDistForCrossValidation.get(i).get(j);
					
					double actualClassProb = probDist.get(actualClassPosition);
					
					int classesBefore = 0;
					int classesEqual = 0;
					for (int l = 0; l < probDist.size(); l++) {
						
						if (probDist.get(l) > actualClassProb) {
							classesBefore++;						
						}
						
						if (probDist.get(l) == actualClassProb) {
							classesEqual++;					
						}
					}
					
					//Generates k Accuracy
					for (int l = classesBefore; l < kAccuracy.length; l++) {
						
						if (l <= classesBefore + classesEqual - 1) {
							
							if (kAccuracy[l] == null) {
								
								kAccuracy[l] = (l - classesBefore + 1.0) / classesEqual;
										
							} else {
								
								kAccuracy[l] += (l - classesBefore + 1.0) / classesEqual;
							}
							
						} else {
							
							if (kAccuracy[l] == null) {
								
								kAccuracy[l] = 1.0;
										
							} else {
								
								kAccuracy[l]++;
							}
						}						
					}					
				}
			}		
		}
		
		//Calculates the final statistics
		
		kAccuracyAvg = new Double[k];
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

	public void evaluateRankModelDynamic(MetaRanker mr, Instances train, Instances test, Classifier classifier, String classifierOptions ) {

		if (mr == null) {
			
			throw new IllegalArgumentException("Invalid classifier.");
			
		}
		
		if (train == null || test == null) {
			
			throw new IllegalArgumentException("Invalid data.");
		}
		
		//List where the raw result is stored
		this.resultSet = new ArrayList<List<Integer>>();
		
		MetaRanker tempMr = new MetaRanker();
		tempMr.setRankSize(mr.getRankSize());
		
		//Iterates over the data to classify the instances 
		for (Instance instance : test) {
			
			List<Integer> result = new ArrayList<Integer>();
			
			//Stores the actual class value on the first position (zero) and the ranked list on the next positions
			result.add((int) instance.classValue() + 1);
			result.addAll(tempMr.classifyInstanceDynamic(instance, train, classifier, classifierOptions));
			
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
		resultSet = new ArrayList<List<Integer>>();
		resultSetProdDist = new ArrayList<List<Double>>();
		
		try {
			
			//Iterates over the data to classify the instances 
			for (Instance instance : data) {
				
				List<Integer> result = new ArrayList<Integer>();
				
				//Stores the actual class value on the first position (zero)
				result.add((int) instance.classValue() + 1);
				
				//Gets the class distribution and prepares to rank the class indexes
				double[] probDist = cls.distributionForInstance(instance);
				
				List<Double> sortedProbDist = new ArrayList<Double>();
				for (int i = 0; i < probDist.length; i++) {
					sortedProbDist.add(probDist[i]);					
				}

				Collections.sort(sortedProbDist, Collections.reverseOrder());
				
				resultSetProdDist.add(sortedProbDist);
				
				//Stores the ranked list on the next positions
				for (int i = 0; i < sortedProbDist.size() ; i++) {					
					for (int j = 0; j < probDist.length; j++) {
						if (probDist[j] == (double) sortedProbDist.get(i)) {
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
	 * @return
	 */
	public String crossValidateRankModel(MetaRanker mr, Classifier cls, String classifierOptions, Instances data, Integer numFolds) {
	
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

		Instances randData = new Instances(data);   // create a copy of the original data		
		randData.randomize(new Random((long) 1.0)); // randomize data with number generator		 
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
		
		this.generateCrossValidationPerformanceStatisticsForMetaranker(5);
		
		return this.toSummaryString();

	}

	public String crossValidateRankModelDynamic(MetaRanker mr, Classifier cls, String classifierOptions, Instances data, Integer numFolds) {
		
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

		Instances randData = new Instances(data);   // create a copy of the original data		
		randData.randomize(new Random((long) 1.0)); // randomize data with number generator		 
		randData.stratify(numFolds);				// stratify the data to enable cross validation
		
		resultSetForCrossValidation = new ArrayList<List<List<Integer>>>();
		testElapsedTime = 0;
		trainElapsedTime = 0;
		
		//Perform the cross validation
		boolean outOfTime = false;
		for (int n = 0; n < numFolds; n++) {
			if (trainElapsedTime <= getMaxExperimentTime()) {
				
				//Generate train and test sets
				Instances train = randData.trainCV(numFolds, n);
				Instances test = randData.testCV(numFolds, n);
				
				long startTime = System.nanoTime();
				this.evaluateRankModelDynamic(mr, train, test, cls, classifierOptions);
				
				trainElapsedTime += System.nanoTime() - startTime;
				
				List<List<Integer>> tmpResultSet = new ArrayList<List<Integer>>();
				tmpResultSet.addAll(resultSet);
				resultSetForCrossValidation.add(tmpResultSet);
			} else {
				
				outOfTime = true;
				
			}
		}
		
		if (outOfTime) {
			trainElapsedTime = 0;
		}

		generateCrossValidationPerformanceStatisticsForMetaranker(5);
		
		return toSummaryString();

	}
	
	public String crossValidateRankModel(Classifier classifier, Instances data, Integer numFolds, Integer rankSize) {
		
		
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

		Instances randData = new Instances(data);   // create a copy of the original data
		randData.randomize(new Random((long) 1.0)); // randomize data with number generator		 
		randData.stratify(numFolds);				// stratify the data to enable cross validation
		
		resultSetForCrossValidation = new ArrayList<List<List<Integer>>>();
		resultSetProbDistForCrossValidation = new ArrayList<List<List<Double>>>();
		testElapsedTime = 0;
		trainElapsedTime = 0;
		
		//Perform the cross validation
		for (int n = 0; n < numFolds; n++) {
			
			//Generate train and test sets
			Instances train = randData.trainCV(numFolds, n);
			Instances test = randData.testCV(numFolds, n);
			
			try {
				
				long startTime = System.nanoTime();
				classifier.buildClassifier(train);
				trainElapsedTime += System.nanoTime() - startTime;
				
				startTime = System.nanoTime();				
				evaluateRankModel(classifier, test, rankSize);
				testElapsedTime += System.nanoTime() - startTime;
				
				resultSetForCrossValidation.add(resultSet);
				resultSetProbDistForCrossValidation.add(resultSetProdDist);
				
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}			  
		}
		
		generateCrossValidationPerformanceStatisticsForClassifier(5);
		
		return toSummaryString();
		
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

	public long getMaxExperimentTime() {
		
		long retMaxTime = Long.MAX_VALUE;
		
		if (maxExperimentTime > 0) {
			
			retMaxTime = maxExperimentTime;
			
		}
		
		return retMaxTime;
	}

	public void setMaxExperimentTime(long maxExperimentTime) {
		this.maxExperimentTime = maxExperimentTime;
	}
	
	private void initiateAllLabelKMetrics(Integer numMetrics, Integer numInstances) {
		
		if (numInstances!=null && numMetrics!=null) {

			allLabelsKMetrics = new HashMap<Integer, Map<Integer, Map<String, Double>>>();
			
			for (int i = 1; i <= numMetrics; i++) {
				
				allLabelsKMetrics.put(i, new HashMap<Integer, Map<String, Double>>());
				
				for (int j = 1; j <= numInstances; j++) {
					
					HashMap<String, Double> oneLabelMetrics = new HashMap<String, Double>();					
					Map<Integer, Map<String, Double>> tempLabelsMetrics = allLabelsKMetrics.get(i);
					tempLabelsMetrics.put(j, oneLabelMetrics);
					allLabelsKMetrics.put(i, tempLabelsMetrics);
				}
			}
		}
	}
	 
	private void incrementLabelKMetricsField(Integer kMetricIndex, Integer classLabel, String metricName, Double increment) {
		
		Double checkedIncrement = 1.0;
		if (increment != null) {
			checkedIncrement = increment;
		}
		
		if (metricName != null && !metricName.isEmpty() && classLabel != null && kMetricIndex != null) {
			
			Map<Integer, Map<String, Double>> tempLabelsMetrics = allLabelsKMetrics.get(kMetricIndex);
			Map<String, Double> tempOneLabelMetrics = tempLabelsMetrics.get(classLabel);
			
			//Updates the value for a key
			if (tempOneLabelMetrics.containsKey(metricName)) {
				
				tempOneLabelMetrics.put(metricName, tempOneLabelMetrics.get(metricName) + checkedIncrement);				
				
			} else {
				
				tempOneLabelMetrics.put(metricName, checkedIncrement);
				
			}
			
			tempLabelsMetrics.put(classLabel, tempOneLabelMetrics);
			allLabelsKMetrics.put(kMetricIndex, tempLabelsMetrics);			
			
		}	
	}
	
	private void calculateKMetricsForMetaranker(Integer k, Integer numLabels) {		
		
		//Clacular as 3 estatisticas a seguir para cada classe para cada nível		
		//tp : na lista da classe, todos da posição da classe verdadeira em diante
		//fn : na lista da classe, todos anteriores à posição da classe verdadeira 
		//fp : nas listas das outras classes, todos da posição da classe (falsa) em diante contam como fp para classe falsa		
		//total : total de instancias de uma classe = tp + fn (guardar a principio so para conferencia)
		int numMetrics = k;
		kPrecisionMicro = new Double[k];
		kPrecisionAvg = new Double[k];
		kPrecisionPon = new Double[k];
		kRecallMicro = new Double[k];
		kRecallAvg = new Double[k];
		kRecallPon = new Double[k];
		
		//Calculates the total number of instances so that allLabelsKMetrics map can be created
		Integer numInstances = 0;	
		for (List<List<Integer>> oneFoldResultSet : resultSetForCrossValidation) {			
			numInstances += oneFoldResultSet.size();
		}
		
		initiateAllLabelKMetrics(numMetrics, numLabels);
		
		//Calculate the metrics iterating over the result set		
		for (List<List<Integer>> oneFoldResultSet : resultSetForCrossValidation) {
			for (List<Integer> list : oneFoldResultSet) {
				
				//The actual class is the first element of the list
				Integer actualClass = list.get(0);				

				//The number of metrics can't be greater than the ranked list size
				numMetrics = k;
				if ((list.size()-1) < numMetrics) {
					numMetrics = list.size()-1;
				}
				
				//Counts tp, fp and fn
				boolean actualClassFound = false;
				for (int i = 1; i <= numMetrics; i++) {
					
					incrementLabelKMetricsField(i, actualClass, "total", null);
					
					if (list.get(i)==actualClass || actualClassFound) {
						//Check if it is a tp
						
						actualClassFound = true;
						
						incrementLabelKMetricsField(i, actualClass, "tp", null);
						
					} else {
						//If not it is a fn to the actual class and a fp to the other label
						
						incrementLabelKMetricsField(i, actualClass, "fn", null);
						incrementLabelKMetricsField(i, list.get(i), "fp", null);
						
					}					
				}
			}
		}
		
		generateKMetrics();
		
	}
	
	private void calculateKMetricsForClassifier(Integer k, Integer numLabels) {		
		
		//Cacular as 3 estatisticas a seguir para cada classe para cada nível		
		//tp : na lista da classe, todos da posição da classe verdadeira em diante
		//fn : na lista da classe, todos anteriores à posição da classe verdadeira 
		//fp : nas listas das outras classes, todos da posição da classe (falsa) em diante contam como fp para classe falsa		
		//total : total de instancias de uma classe = tp + fn (guardar a principio so para conferencia)
		int numMetrics = k;
		kPrecisionMicro = new Double[k];
		kPrecisionAvg = new Double[k];
		kPrecisionPon = new Double[k];
		kRecallMicro = new Double[k];
		kRecallAvg = new Double[k];
		kRecallPon = new Double[k];
		
		initiateAllLabelKMetrics(numMetrics, numLabels);
		
		//Calculate the metrics iterating over the result set		
		for (int i = 0; i < resultSetForCrossValidation.size(); i++) {			
			for (int j = 0; j < resultSetForCrossValidation.get(i).size(); j++) {				
				
				//Gets the result list (ranking) for that instance
				List<Integer> list = resultSetForCrossValidation.get(i).get(j);
				
				//Probabilities of classes
				List<Double> probDist = resultSetProbDistForCrossValidation.get(i).get(j);
				
				//The actual class is the first element of the list
				Integer actualClass = list.get(0);

				//Position considering the first equal to 0
				Integer actualClassPosition = list.subList(1, list.size()).indexOf(actualClass);
				
				Double actualClassProb = probDist.get(actualClassPosition);
				
				//Calculates the increment since there may be classes with probability equal to the actual class					
				Integer numClassesEqualProb = 0;
				for (Double probability : probDist) {
					if (probability == actualClassProb) {
						numClassesEqualProb++;
					}
				}
				Double increment = (double) (1.0/numClassesEqualProb);

				//The number of metrics can't be greater than the ranked list size
				numMetrics = k;
				if ((list.size()-1) < numMetrics) {
					numMetrics = list.size()-1;
				}

				//Counts tp, fp and fn
				Integer numPasClassEqualProb = 1;
				for (int z = 1; z <= numMetrics; z++) {
					
					incrementLabelKMetricsField(z, actualClass, "total", null);
					
					if (probDist.get(z-1)==actualClassProb) {
						//Check if it is a tp
						//Case where there may be several classes with probability equal to the actual class
						
						incrementLabelKMetricsField(z, actualClass, "tp", numPasClassEqualProb*increment);
						
						if (probDist.get(z-1) == actualClassProb) {
							numPasClassEqualProb++;
						}
						
					} else if (probDist.get(z-1)<actualClassProb) {
						//Case where the actual class is already on a previous position on the list
						
						incrementLabelKMetricsField(z, actualClass, "tp", null);
						
					} else {
						//If not it is a fn to the actual class and a fp to the other label
						
						incrementLabelKMetricsField(z, actualClass, "fn", null);
						incrementLabelKMetricsField(z, list.get(z), "fp", null);
						
					}					
				}
			}
		}
		
		generateKMetrics();
		
	}
	
	private void generateKMetrics() {
		
		//Generates kPrecision and kRecall
		for (int i = 1; i <= allLabelsKMetrics.size(); i++) {
			
			Double truePositives = 0.0;
			Double falseNegatives = 0.0;
			Double falsePositives = 0.0;
			kPrecisionAvg[i-1] = 0.0;
			kPrecisionPon[i-1] = 0.0;
			kRecallAvg[i-1] = 0.0;
			kRecallPon[i-1] = 0.0;
			
			Map<Integer, Map<String, Double>> labelMetrics = allLabelsKMetrics.get(i);
			
			for (int j = 1; j <= labelMetrics.size(); j++) {
				
				Map<String, Double> oneLabelMetrics = labelMetrics.get(j);
				
				//Calculates iPrecision and iRecall per label j
				
				//If it does not conten the key the then value is zero
				Double tpI =0.0, fpI =0.0, fnI =0.0, totalI = 0.0;
				if (oneLabelMetrics.containsKey("tp")) {				
					tpI = oneLabelMetrics.get("tp");
				}
				
				if (oneLabelMetrics.containsKey("fp")) {				
					fpI = oneLabelMetrics.get("fp");
				}
				
				if (oneLabelMetrics.containsKey("fn")) {				
					fnI = oneLabelMetrics.get("fn");
				}
				
				if (oneLabelMetrics.containsKey("total")) {				
					totalI = oneLabelMetrics.get("total");
				}
				
				//Sums the totals
				
				if (tpI+fpI > 0) {
					
					kPrecisionAvg[i-1] += (double) tpI/(tpI+fpI);
					kPrecisionPon[i-1] += (double) (tpI*totalI)/(tpI+fpI);
					
				}
				
				if (tpI+fnI > 0) {
				
					kRecallAvg[i-1] += (double) tpI/(tpI+fnI);
					kRecallPon[i-1] += (double) (tpI*totalI)/(tpI+fnI);
					
				}
				
				truePositives += tpI;
				falsePositives += fpI;
				falseNegatives += fnI;				
			}
			/*
			System.out.println("##### Metric " + i + " #####");
			System.out.println("True Positives: " + truePositives);
			System.out.println("False Positives: " + falsePositives);
			System.out.println("False Negatives: " + falseNegatives);
			System.out.println();
			*/
			//Calculates the total number of instances so that allLabelsKMetrics map can be created
			Integer numInstances = 0;	
			for (List<List<Integer>> oneFoldResultSet : resultSetForCrossValidation) {			
				numInstances += oneFoldResultSet.size();
			}
			
			//Calculate the metrics
			kPrecisionMicro[i-1] = (double) (truePositives/(truePositives + falsePositives));
			kPrecisionAvg[i-1] = kPrecisionAvg[i-1] / (double) labelMetrics.size();
			kPrecisionPon[i-1] = kPrecisionPon[i-1] / (double) numInstances;
			
			kRecallMicro[i-1] = (double) (truePositives/(truePositives + falseNegatives));
			kRecallAvg[i-1] = kRecallAvg[i-1] / (double) labelMetrics.size();
			kRecallPon[i-1] = kRecallPon[i-1] / (double) numInstances;
		}		
	}
}

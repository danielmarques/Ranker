package br.puc.drm.ranker;

import java.util.ArrayList;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;

public class RankEvaluation {

	private List<List<Integer>> resultSet;
	private Double totalScore;

	//Generates all relevant statistics about the result and stores internally
	private void generatePerformanceStatistics() {
		
		Double totalScore = 0.0;
		for (List<Integer> list : resultSet) {
			
			Integer actualClass = list.get(0);
			Double position = list.subList(1, list.size()-1).indexOf(actualClass) + 1.0;
			
			Double score = (list.size() - position) / (list.size()-1);
			totalScore += score;
		}
		
		this.totalScore = totalScore;	
	}
	
	/**
	 * Evaluates a model of the MetaRanker class.
	 * 
	 * @param mr A instance of MetaRanker
	 * @param data The test dataset
	 * @return A summary string of the result's performance
	 */
	public String evaluateModel(MetaRanker mr, Instances data) {

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
		
		return this.toSummaryString();
	}

	/**
	 * Outputs the performance statistics in summary form
	 * 
	 * @return A summary string of the result's performance
	 */
	public String toSummaryString() {
	
		String ret = "Precision: " + totalScore + " of " + resultSet.size() + " - " + (totalScore/resultSet.size())*100 + " %";

		return ret;
	}

}

package br.puc.drm.ranker.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Collection;
import java.util.List;

import org.junit.Test;

import br.puc.drm.ranker.MetaRanker;
import br.puc.drm.ranker.RankEvaluation;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class RankEvaluationTest {

	public Instances loadTestFile(String fileName) {
		
		Instances data = null;
		
		ArffLoader loader = new ArffLoader();
	    try {
	    	
			loader.setFile(new File(fileName));
			data = loader.getDataSet();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    
	    return data;
	}
	
	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByEvaluateModel() {
		
		Instances data = loadTestFile("iris.arff");
		MetaRanker mr = new MetaRanker();
		mr.buildClassifier(new J48(), data);
		RankEvaluation eval = new RankEvaluation();
		eval.evaluateRankModel(mr, null);
		
		
	}

	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByEvaluateModel2() {
		
		Instances data = loadTestFile("iris.arff");
		RankEvaluation eval = new RankEvaluation();
		MetaRanker mr = null;
		eval.evaluateRankModel(mr, data);
		
		
	}

	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByEvaluateModel3() {
		
		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes());
		Classifier cls = new J48();
		try {
			cls.buildClassifier(data);
			RankEvaluation eval = new RankEvaluation();
			eval.evaluateRankModel(cls, null);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByEvaluateModel4() {
		
		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		Classifier cls = null;
		eval.evaluateRankModel(cls, data);
		
		
	}
	
	@Test
	public void modelShouldBeEvaluated() {
		
		Instances data = loadTestFile("iris.arff");
		MetaRanker mr = new MetaRanker();
		mr.buildClassifier(new J48(), data);
		RankEvaluation eval = new RankEvaluation();
		String ret = eval.evaluateRankModel(mr, data);
		
		assertFalse(ret == null);
		assertFalse(ret.isEmpty());
		
		try {
			Field field = eval.getClass().getDeclaredField("resultSet");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			List<List<Integer>> retResultSet = (List<List<Integer>>) field.get(eval);			
			
			assertFalse(retResultSet == null);
			assertFalse(retResultSet.isEmpty());
			assertTrue(retResultSet.size() == 150);
			
			for (List<Integer> list : retResultSet) {
				assertTrue(list.size()==4);
				for (Integer i : list) {
					assertTrue(i>0 && i<4);
				}
			}
			
			field = eval.getClass().getDeclaredField("totalScore");
			field.setAccessible(true);
			Double retTotalScore = (Double) field.get(eval);
			
			assertFalse(retTotalScore == null);
			assertTrue(0 < retTotalScore && retTotalScore < retResultSet.size());
			
		} catch (NoSuchFieldException | SecurityException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalArgumentException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	@Test
	public void modelShouldBeEvaluated2() {
		
		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		Classifier cls = new J48();
		
		try {
			
			cls.buildClassifier(data);
			
			RankEvaluation eval = new RankEvaluation();
			String ret = eval.evaluateRankModel(cls, data);
			
			assertFalse(ret == null);
			assertFalse(ret.isEmpty());
			
			Field field = eval.getClass().getDeclaredField("resultSet");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			List<List<Integer>> retResultSet = (List<List<Integer>>) field.get(eval);			
			
			assertFalse(retResultSet == null);
			assertFalse(retResultSet.isEmpty());
			assertTrue(retResultSet.size() == 150);
			
			for (List<Integer> list : retResultSet) {
				assertTrue(list.size()==4);				
				for (Integer i : list) {
					assertTrue(i>0 && i<4);
				}
			}
			
			field = eval.getClass().getDeclaredField("totalScore");
			field.setAccessible(true);
			Double retTotalScore = (Double) field.get(eval);
			
			assertFalse(retTotalScore == null);
			assertTrue(0 < retTotalScore && retTotalScore < retResultSet.size());
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();;			
		}
		
	}

	@Test
	public void summaryStringShoudBeReturned () {
		
		Instances data = loadTestFile("iris.arff");
		MetaRanker mr = new MetaRanker();
		mr.buildClassifier(new J48(), data);
		RankEvaluation eval = new RankEvaluation();
		eval.evaluateRankModel(mr, data);
		
		String srt = eval.toSummaryString();
		
		assertFalse(srt == null);
		assertFalse(srt.isEmpty());
	}

}

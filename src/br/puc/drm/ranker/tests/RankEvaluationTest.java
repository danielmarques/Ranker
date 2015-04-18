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
		eval.evaluateModel(mr, null);
		
		
	}

	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByEvaluateModel2() {
		
		Instances data = loadTestFile("iris.arff");
		RankEvaluation eval = new RankEvaluation();
		eval.evaluateModel(null, data);
		
		
	}
	
	@Test
	public void modelShouldBeEvaluated() {
		
		Instances data = loadTestFile("iris.arff");
		MetaRanker mr = new MetaRanker();
		mr.buildClassifier(new J48(), data);
		RankEvaluation eval = new RankEvaluation();
		String ret = eval.evaluateModel(mr, data);
		
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
	public void summaryStringShoudBeReturned () {
		
		Instances data = loadTestFile("iris.arff");
		MetaRanker mr = new MetaRanker();
		mr.buildClassifier(new J48(), data);
		RankEvaluation eval = new RankEvaluation();
		eval.evaluateModel(mr, data);
		
		String srt = eval.toSummaryString();
		
		assertFalse(srt == null);
		assertFalse(srt.isEmpty());
	}

}

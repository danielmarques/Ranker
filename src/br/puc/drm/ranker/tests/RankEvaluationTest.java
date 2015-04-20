package br.puc.drm.ranker.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Collection;
import java.util.List;
import java.util.Random;

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
	
	//EvaluateRankModel tests for MetaRanker
	
	//If data is null
	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByEvaluateRankModelForMetaRanker1() {
		
		Instances data = loadTestFile("iris.arff");
		MetaRanker mr = new MetaRanker();
		mr.buildClassifier(new J48(), data);
		RankEvaluation eval = new RankEvaluation();
		eval.evaluateRankModel(mr, null);		
		
	}

	//If MetaRanker is null
	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByEvaluateRankModelForMetaRanker2() {
		
		Instances data = loadTestFile("iris.arff");
		RankEvaluation eval = new RankEvaluation();
		MetaRanker mr = null;
		eval.evaluateRankModel(mr, data);
		
		
	}

	@Test
	public void modelShouldBeEvaluatedForMetaRanker() {
		
		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
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

	//EvaluateRankModel tests for classifiers
	
	//If the data is null
	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByEvaluateRankModelForClassifiers1() {
		
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

	//If the Classifier is null
	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByEvaluateRankModelForClassifiers2() {
		
		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		Classifier cls = null;
		eval.evaluateRankModel(cls, data);
		
		
	}

	//If the Classifier is not trained
	@Test (expected = IllegalStateException.class)
	public void illegalStateExceptionShouldBeReturnedByEvaluateRankModelForClassifiers() {
		
		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		eval.evaluateRankModel(new J48(), data);		
		
	}
	
	@Test
	public void modelShouldBeEvaluatedForAClassifier() {
		
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

	//CrossValidateRankModel (for MetaRanker) tests
	
	//If the MetaRanker is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel1() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		MetaRanker mr = null;
		eval.crossValidateRankModel(mr, new J48(), data, 3, new Random(1));
	}

	//If the data is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel2() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(mr, new J48(), null, 3, new Random(1));
	}

	//If the number of folds is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel3() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		Integer n = null;
		eval.crossValidateRankModel(mr, new J48(), data, n , new Random(1));
	}
	
	//If the random is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel4() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		Random r = null;
		eval.crossValidateRankModel(mr, new J48(), data, 3, r);
	}
	
	//If the number of folds is less than 2
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel5() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(mr, new J48(), data, 1, new Random(1));
	}
	
	//If the number of folds is greater than the number of instances.
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel6() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(mr, new J48(), data, data.numInstances()+1, new Random(1));
	}
	
	//If the class is not nominal
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel8() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-2);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(mr, new J48(), data, 3, new Random(1));
	}
	
	@Test
	public void modelShouldBeCrossValidatedForMetaRanker() {
		
		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		String ret = eval.crossValidateRankModel(mr, new J48(), data, 3, new Random(1));
		
		assertFalse(ret == null);
		assertFalse(ret.isEmpty());
		
		try {
			Field field = eval.getClass().getDeclaredField("resultSetForCrossValidation");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			List<List<List<Integer>>> retResultSet = (List<List<List<Integer>>>) field.get(eval);			

			assertFalse(retResultSet == null);
			assertFalse(retResultSet.isEmpty());
			assertTrue(retResultSet.size() == 3);
			assertTrue(retResultSet.get(0).size() == 50);
			
			for (List<List<Integer>> oneFoldResultSet : retResultSet) {
				for (List<Integer> list : oneFoldResultSet) {
					assertTrue(list.size()==4);
					for (Integer i : list) {
						assertTrue(i>0 && i<4);
					}
				}
			}
			
			field = eval.getClass().getDeclaredField("totalScore");
			field.setAccessible(true);
			Double retTotalScore = (Double) field.get(eval);
			
			assertFalse(retTotalScore == null);
			assertTrue(0 < retTotalScore && retTotalScore < retResultSet.get(0).size());
			
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

	//CrossValidateRankModel (for Classifier) tests
	
	//If the Classifier is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS1() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		Classifier cls = null;
		eval.crossValidateRankModel(cls, data, 3, new Random(1));
	}

	//If the data is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS2() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(new J48(), null, 3, new Random(1));
	}

	//If the number of folds is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS3() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		Integer n = null;
		eval.crossValidateRankModel(new J48(), data, n , new Random(1));
	}
	
	//If the random is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS4() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		Random r = null;
		eval.crossValidateRankModel(new J48(), data, 3, r);
	}
	
	//If the number of folds is less than 2
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS5() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(new J48(), data, 1, new Random(1));
	}
	
	//If the number of folds is greater than the number of instances.
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS6() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(new J48(), data, data.numInstances()+1, new Random(1));
	}
	
	//If the class is not nominal
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS8() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-2);
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(new J48(), data, 3, new Random(1));
	}
	
	@Test
	public void modelShouldBeCrossValidatedForClassifier() {
		
		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		String ret = eval.crossValidateRankModel(new J48(), data, 3, new Random(1));
		
		assertFalse(ret == null);
		assertFalse(ret.isEmpty());
		
		try {
			Field field = eval.getClass().getDeclaredField("resultSetForCrossValidation");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			List<List<List<Integer>>> retResultSet = (List<List<List<Integer>>>) field.get(eval);			

			assertFalse(retResultSet == null);
			assertFalse(retResultSet.isEmpty());
			assertTrue(retResultSet.size() == 3);
			assertTrue(retResultSet.get(0).size() == 50);
			
			for (List<List<Integer>> oneFoldResultSet : retResultSet) {
				for (List<Integer> list : oneFoldResultSet) {
					assertTrue(list.size()==4);
					for (Integer i : list) {
						assertTrue(i>0 && i<4);
					}
				}
			}
			
			field = eval.getClass().getDeclaredField("totalScore");
			field.setAccessible(true);
			Double retTotalScore = (Double) field.get(eval);
			
			assertFalse(retTotalScore == null);
			assertTrue(0 < retTotalScore && retTotalScore < retResultSet.get(0).size());
			
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

	// toString tests
	
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

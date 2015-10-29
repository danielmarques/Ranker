package br.puc.drm.ranker.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

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
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		MetaRanker mr = new MetaRanker();
		mr.buildClassifier(new J48(), data, null);
		RankEvaluation eval = new RankEvaluation();
		eval.evaluateRankModel(mr, null);		
		
	}

	//If MetaRanker is null
	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByEvaluateRankModelForMetaRanker2() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		RankEvaluation eval = new RankEvaluation();
		MetaRanker mr = null;
		eval.evaluateRankModel(mr, data);		
		
	}

	@Test
	public void modelShouldBeEvaluatedForMetaRanker() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		mr.buildClassifier(new J48(), data, null);
		RankEvaluation eval = new RankEvaluation();
		eval.evaluateRankModel(mr, data);
		
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
				List<Integer> resultList = list.subList(1, list.size());				
				for (int i = 0; i < resultList.size(); i++) {
					Integer element = resultList.get(i);
					resultList.remove(i);
					assertFalse(resultList.contains(element));					
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
	public void modelShouldBeEvaluatedForMetaRanker2() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/glass.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		mr.setRankSize(3);
		mr.buildClassifier(new J48(), data, null);
		RankEvaluation eval = new RankEvaluation();
		eval.evaluateRankModel(mr, data);
		
		try {
			Field field = eval.getClass().getDeclaredField("resultSet");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			List<List<Integer>> retResultSet = (List<List<Integer>>) field.get(eval);			
			
			assertFalse(retResultSet == null);
			assertFalse(retResultSet.isEmpty());
			assertTrue(retResultSet.size() == 214);
			
			for (List<Integer> list : retResultSet) {
				assertTrue(list.size()==4);
				for (Integer i : list) {
					assertTrue(i>0 && i<8);
				}
				List<Integer> resultList = list.subList(1, list.size());				
				for (int i = 0; i < resultList.size(); i++) {
					Integer element = resultList.get(i);
					resultList.remove(i);
					assertFalse(resultList.contains(element));					
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
	public void modelShouldBeEvaluatedForMetaRankerDynamic() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		eval.evaluateRankModelDynamic(mr, data, data, new J48(), null);
		
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
				List<Integer> resultList = list.subList(1, list.size());				
				for (int i = 0; i < resultList.size(); i++) {
					Integer element = resultList.get(i);
					resultList.remove(i);
					assertFalse(resultList.contains(element));					
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
	public void modelShouldBeEvaluatedForMetaRankerDynamic2() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/glass.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		mr.setRankSize(3);
		RankEvaluation eval = new RankEvaluation();
		eval.evaluateRankModelDynamic(mr, data, data, new J48(), null);
		
		try {
			Field field = eval.getClass().getDeclaredField("resultSet");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			List<List<Integer>> retResultSet = (List<List<Integer>>) field.get(eval);			
			
			assertFalse(retResultSet == null);
			assertFalse(retResultSet.isEmpty());
			assertTrue(retResultSet.size() == 214);
			
			for (List<Integer> list : retResultSet) {
				assertTrue(list.size()==4);
				for (Integer i : list) {
					assertTrue(i>0 && i<8);
				}
				List<Integer> resultList = list.subList(1, list.size());				
				for (int i = 0; i < resultList.size(); i++) {
					Integer element = resultList.get(i);
					resultList.remove(i);
					assertFalse(resultList.contains(element));					
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
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes());
		Classifier cls = new J48();
		try {
			cls.buildClassifier(data);
			RankEvaluation eval = new RankEvaluation();
			eval.evaluateRankModel(cls, null, 3);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	//If the Classifier is null
	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByEvaluateRankModelForClassifiers2() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		Classifier cls = null;
		eval.evaluateRankModel(cls, data, 3);
		
		
	}

	//If the Classifier is not trained
	@Test (expected = IllegalStateException.class)
	public void illegalStateExceptionShouldBeReturnedByEvaluateRankModelForClassifiers() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		eval.evaluateRankModel(new J48(), data, 3);		
		
	}
	
	@Test
	public void modelShouldBeEvaluatedForAClassifier() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		Classifier cls = new J48();
		
		try {
			
			cls.buildClassifier(data);
			
			RankEvaluation eval = new RankEvaluation();
			eval.evaluateRankModel(cls, data, 3);
			
			Field field = eval.getClass().getDeclaredField("resultSet");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			List<List<Integer>> retResultSet = (List<List<Integer>>) field.get(eval);			
			
			assertFalse(retResultSet == null);
			assertFalse(retResultSet.isEmpty());
			assertTrue(retResultSet.size() == 150);
			
			for (List<Integer> list : retResultSet) {
				assertTrue(list.size()==4);
				assertTrue(list.subList(1, list.size()).contains(list.get(0)));				
				
				for (Integer i : list) {
					assertTrue(i>0 && i<4);
				}
				
				List<Integer> resultList = list.subList(1, list.size());				
				for (int i = 0; i < resultList.size(); i++) {
					Integer element = resultList.get(i);
					resultList.remove(i);
					assertFalse(resultList.contains(element));					
				}
			}
			
			field = eval.getClass().getDeclaredField("resultSetProdDist");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			List<List<Double>> retResultSetProbDist = (List<List<Double>>) field.get(eval);			
			
			assertFalse(retResultSetProbDist == null);
			assertFalse(retResultSetProbDist.isEmpty());
			assertTrue(retResultSetProbDist.size() == 150);
			
			for (List<Double> probDist : retResultSetProbDist) {
				assertTrue(probDist.size() == 3);
				Double probSum = 0.0;
				for (Double d : probDist) {
					assertTrue(d >= 0 && d <=1);
					probSum += d;					
				}
				assertTrue(probSum==1.0);
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
	public void modelShouldBeEvaluatedForAClassifier2() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/glass.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		Classifier cls = new J48();
		
		try {
			
			cls.buildClassifier(data);
			
			RankEvaluation eval = new RankEvaluation();
			eval.evaluateRankModel(cls, data, 3);
			
			Field field = eval.getClass().getDeclaredField("resultSet");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			List<List<Integer>> retResultSet = (List<List<Integer>>) field.get(eval);			
			
			assertFalse(retResultSet == null);
			assertFalse(retResultSet.isEmpty());
			assertTrue(retResultSet.size() == 214);
			
			for (List<Integer> list : retResultSet) {
				assertTrue(list.size()==4);				
				for (Integer i : list) {
					assertTrue(i>0 && i<8);
				}
				
				List<Integer> resultList = list.subList(1, list.size());				
				for (int i = 0; i < resultList.size(); i++) {
					Integer element = resultList.get(i);
					resultList.remove(i);
					assertFalse(resultList.contains(element));					
				}
			}
			
			field = eval.getClass().getDeclaredField("resultSetProdDist");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			List<List<Double>> retResultSetProbDist = (List<List<Double>>) field.get(eval);			
			
			assertFalse(retResultSetProbDist == null);
			assertFalse(retResultSetProbDist.isEmpty());
			assertTrue(retResultSetProbDist.size() == 214);
			
			for (List<Double> probDist : retResultSetProbDist) {
				assertTrue(probDist.size() == 7);
				Double probSum = 0.0;
				for (Double d : probDist) {
					assertTrue(d >= 0 && d <=1);
					probSum += d;					
				}
				assertTrue(probSum==1.0);				
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

		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		MetaRanker mr = null;
		eval.crossValidateRankModel(mr, new J48(), null, data, 3);
	}

	//If the data is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel2() {

		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(mr, new J48(), null, null, 3);
	}

	//If the number of folds is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel3() {

		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		Integer n = null;
		eval.crossValidateRankModel(mr, new J48(), null, data, n);
	}
	
	//If the number of folds is less than 2
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel5() {

		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(mr, new J48(), null, data, 1);
	}
	
	//If the number of folds is greater than the number of instances.
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel6() {

		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(mr, new J48(), null, data, data.numInstances()+1);
	}
	
	//If the class is not nominal
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel8() {

		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-2);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(mr, new J48(), null, data, 3);
	}

	@Test
	public void modelShouldBeCrossValidatedForMetaRanker1() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		String ret = eval.crossValidateRankModel(mr, new J48(), null, data, 3);
		
		assertFalse(ret == null);
		assertFalse(ret.isEmpty());
		
		try {
			
			//Verify the brute result set
			
			Field field = eval.getClass().getDeclaredField("resultSetForCrossValidation");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			List<List<List<Integer>>> retResultSetForCrossValidation = (List<List<List<Integer>>>) field.get(eval);			

			assertFalse(retResultSetForCrossValidation == null);
			assertFalse(retResultSetForCrossValidation.isEmpty());
			assertTrue(retResultSetForCrossValidation.size() == 3);
			assertTrue(retResultSetForCrossValidation.get(0).size() == 50);
			assertTrue(retResultSetForCrossValidation.get(1).size() == 50);
			assertTrue(retResultSetForCrossValidation.get(2).size() == 50);
			
			for (List<List<Integer>> oneFoldResultSet : retResultSetForCrossValidation) {
				for (List<Integer> list : oneFoldResultSet) {
					assertTrue(list.size()==4);
					for (Integer i : list) {
						assertTrue(i>0 && i<4);
					}
					List<Integer> resultList = list.subList(1, list.size());				
					for (int i = 0; i < resultList.size(); i++) {
						Integer element = resultList.get(i);
						resultList.remove(i);
						assertFalse(resultList.contains(element));					
					}
				}
			}
			
			//Verify result statistics fields
			
			field = eval.getClass().getDeclaredField("maxScoreAvg");
			field.setAccessible(true);
			Double retMaxScoreAvg = (Double) field.get(eval);			
			assertFalse(retMaxScoreAvg == null);
			assertTrue(retMaxScoreAvg == 50);
			
			field = eval.getClass().getDeclaredField("kAccuracy");
			field.setAccessible(true);
			Double[] kAccuracy = (Double[]) field.get(eval);
			
			assertFalse(kAccuracy == null);
			assertTrue(kAccuracy.length == 5);
			assertTrue(kAccuracy[4] == 150);
			assertTrue(kAccuracy[3] == 150);
			assertTrue(kAccuracy[2] == 150);
			assertTrue(kAccuracy[1] <= 150);
			assertTrue(kAccuracy[0] <= 150);
			
			
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
	public void modelShouldBeCrossValidatedForMetaRanker2() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/glass.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		mr.setRankSize(3);
		RankEvaluation eval = new RankEvaluation();
		String ret = eval.crossValidateRankModel(mr, new J48(), null, data, 3);
		
		assertFalse(ret == null);
		assertFalse(ret.isEmpty());
		
		try {
			
			//Verify the brute result set
			
			Field field = eval.getClass().getDeclaredField("resultSetForCrossValidation");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			List<List<List<Integer>>> retResultSetForCrossValidation = (List<List<List<Integer>>>) field.get(eval);			

			assertFalse(retResultSetForCrossValidation == null);
			assertFalse(retResultSetForCrossValidation.isEmpty());
			assertTrue(retResultSetForCrossValidation.size() == 3);
			assertTrue(retResultSetForCrossValidation.get(0).size() <= 72);
			assertTrue(retResultSetForCrossValidation.get(1).size() <= 72);
			assertTrue(retResultSetForCrossValidation.get(2).size() <= 72);
			
			for (List<List<Integer>> oneFoldResultSet : retResultSetForCrossValidation) {
				for (List<Integer> list : oneFoldResultSet) {
					assertTrue(list.size()==4);
					for (Integer i : list) {
						assertTrue(i>0 && i<8);
					}
					
					List<Integer> resultList = list.subList(1, list.size());				
					for (int i = 0; i < resultList.size(); i++) {
						Integer element = resultList.get(i);
						resultList.remove(i);
						assertFalse(resultList.contains(element));					
					}
				}
			}
			
			//Verify result statistics fields
			
			field = eval.getClass().getDeclaredField("maxScoreAvg");
			field.setAccessible(true);
			Double retMaxScoreAvg = (Double) field.get(eval);			
			assertFalse(retMaxScoreAvg == null);
			assertTrue(retMaxScoreAvg == 214.0/3.0);
			
			field = eval.getClass().getDeclaredField("kAccuracy");
			field.setAccessible(true);
			Double[] kAccuracy = (Double[]) field.get(eval);
			
			assertFalse(kAccuracy == null);
			assertTrue(kAccuracy.length == 5);
			assertTrue(kAccuracy[4] <= 214);
			assertTrue(kAccuracy[3] <= 214);
			assertTrue(kAccuracy[2] <= 214);
			assertTrue(kAccuracy[1] <= 214);
			assertTrue(kAccuracy[0] <= 214);
			
			
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
	public void modelShouldBeCrossValidatedForMetaRankerDynamic1() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		String ret = eval.crossValidateRankModelDynamic(mr, new J48(), null, data, 3);
		
		assertFalse(ret == null);
		assertFalse(ret.isEmpty());
		
		try {
			
			//Verify the brute result set
			
			Field field = eval.getClass().getDeclaredField("resultSetForCrossValidation");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			List<List<List<Integer>>> retResultSetForCrossValidation = (List<List<List<Integer>>>) field.get(eval);			

			assertFalse(retResultSetForCrossValidation == null);
			assertFalse(retResultSetForCrossValidation.isEmpty());
			assertTrue(retResultSetForCrossValidation.size() == 3);
			assertTrue(retResultSetForCrossValidation.get(0).size() == 50);
			assertTrue(retResultSetForCrossValidation.get(1).size() == 50);
			assertTrue(retResultSetForCrossValidation.get(2).size() == 50);
			
			for (List<List<Integer>> oneFoldResultSet : retResultSetForCrossValidation) {
				for (List<Integer> list : oneFoldResultSet) {
					assertTrue(list.size()==4);
					for (Integer i : list) {
						assertTrue(i>0 && i<4);
					}
					List<Integer> resultList = list.subList(1, list.size());				
					for (int i = 0; i < resultList.size(); i++) {
						Integer element = resultList.get(i);
						resultList.remove(i);
						assertFalse(resultList.contains(element));					
					}
				}
			}
			
			//Verify result statistics fields
			
			field = eval.getClass().getDeclaredField("maxScoreAvg");
			field.setAccessible(true);
			Double retMaxScoreAvg = (Double) field.get(eval);			
			assertFalse(retMaxScoreAvg == null);
			assertTrue(retMaxScoreAvg == 50);
			
			field = eval.getClass().getDeclaredField("kAccuracy");
			field.setAccessible(true);
			Double[] kAccuracy = (Double[]) field.get(eval);
			
			assertFalse(kAccuracy == null);
			assertTrue(kAccuracy.length == 5);
			assertTrue(kAccuracy[4] == 150);
			assertTrue(kAccuracy[3] == 150);
			assertTrue(kAccuracy[2] == 150);
			assertTrue(kAccuracy[1] <= 150);
			assertTrue(kAccuracy[0] <= 150);
			
			
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
	public void modelShouldBeCrossValidatedForMetaRankerDynamic2() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/glass.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		mr.setRankSize(3);
		RankEvaluation eval = new RankEvaluation();
		String ret = eval.crossValidateRankModelDynamic(mr, new J48(), null, data, 3);
		
		assertFalse(ret == null);
		assertFalse(ret.isEmpty());
		
		try {
			
			//Verify the brute result set
			
			Field field = eval.getClass().getDeclaredField("resultSetForCrossValidation");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			List<List<List<Integer>>> retResultSetForCrossValidation = (List<List<List<Integer>>>) field.get(eval);			

			assertFalse(retResultSetForCrossValidation == null);
			assertFalse(retResultSetForCrossValidation.isEmpty());
			assertTrue(retResultSetForCrossValidation.size() == 3);
			assertTrue(retResultSetForCrossValidation.get(0).size() <= 72);
			assertTrue(retResultSetForCrossValidation.get(1).size() <= 72);
			assertTrue(retResultSetForCrossValidation.get(2).size() <= 72);
			
			for (List<List<Integer>> oneFoldResultSet : retResultSetForCrossValidation) {
				for (List<Integer> list : oneFoldResultSet) {
					assertTrue(list.size()==4);
					for (Integer i : list) {
						assertTrue(i>0 && i<8);
					}
					
					List<Integer> resultList = list.subList(1, list.size());				
					for (int i = 0; i < resultList.size(); i++) {
						Integer element = resultList.get(i);
						resultList.remove(i);
						assertFalse(resultList.contains(element));					
					}
				}
			}
			
			//Verify result statistics fields
			
			field = eval.getClass().getDeclaredField("maxScoreAvg");
			field.setAccessible(true);
			Double retMaxScoreAvg = (Double) field.get(eval);			
			assertFalse(retMaxScoreAvg == null);
			assertTrue(retMaxScoreAvg == 214.0/3.0);
			
			field = eval.getClass().getDeclaredField("kAccuracy");
			field.setAccessible(true);
			Double[] kAccuracy = (Double[]) field.get(eval);
			
			assertFalse(kAccuracy == null);
			assertTrue(kAccuracy.length == 5);
			assertTrue(kAccuracy[4] <= 214);
			assertTrue(kAccuracy[3] <= 214);
			assertTrue(kAccuracy[2] <= 214);
			assertTrue(kAccuracy[1] <= 214);
			assertTrue(kAccuracy[0] <= 214);
			
			
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
	public void modelShouldNotBeCrossValidatedForMetaRankerDynamic() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/glass.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		mr.setRankSize(3);
		RankEvaluation eval = new RankEvaluation();
		eval.setMaxExperimentTime((long) 1);
		String ret = eval.crossValidateRankModelDynamic(mr, new J48(), null, data, 3);
		
		assertFalse(ret == null);
		assertFalse(ret.isEmpty());
		
		try {
			
			//Verify the trainElapsedTime
			
			Field field = eval.getClass().getDeclaredField("trainElapsedTime");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			long trainElapsedTime = (long) field.get(eval);
			
			assertTrue(trainElapsedTime == 0);	
			
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

		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		Classifier cls = null;
		eval.crossValidateRankModel(cls, data, 3, null);
	}

	//If the data is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS2() {

		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(new J48(), null, 3, null);
	}

	//If the number of folds is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS3() {

		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		Integer n = null;
		eval.crossValidateRankModel(new J48(), data, n , null);
	}
	
	//If the number of folds is less than 2
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS5() {

		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(new J48(), data, 1, null);
	}
	
	//If the number of folds is greater than the number of instances.
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS6() {

		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(new J48(), data, data.numInstances()+1, null);
	}
	
	//If the class is not nominal
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS8() {

		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-2);
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(new J48(), data, 3, null);
	}
	
	@Test
	public void modelShouldBeCrossValidatedForClassifier1() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		String ret = eval.crossValidateRankModel(new J48(), data, 3, null);
		
		assertFalse(ret == null);
		assertFalse(ret.isEmpty());
		
		try {
			Field field = eval.getClass().getDeclaredField("resultSetForCrossValidation");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			List<List<List<Integer>>> retResultSetForCrossValidation = (List<List<List<Integer>>>) field.get(eval);			

			assertFalse(retResultSetForCrossValidation == null);
			assertFalse(retResultSetForCrossValidation.isEmpty());
			assertTrue(retResultSetForCrossValidation.size() == 3);
			assertTrue(retResultSetForCrossValidation.get(0).size() == 50);
			assertTrue(retResultSetForCrossValidation.get(1).size() == 50);
			assertTrue(retResultSetForCrossValidation.get(2).size() == 50);
			
			for (List<List<Integer>> oneFoldResultSet : retResultSetForCrossValidation) {
				for (List<Integer> list : oneFoldResultSet) {
					assertTrue(list.size()==4);
					for (Integer i : list) {
						assertTrue(i>0 && i<4);
					}
					List<Integer> resultList = list.subList(1, list.size());				
					for (int i = 0; i < resultList.size(); i++) {
						Integer element = resultList.get(i);
						resultList.remove(i);
						assertFalse(resultList.contains(element));					
					}
				}
			}
			
			//Verify result statistics fields
			
			field = eval.getClass().getDeclaredField("maxScoreAvg");
			field.setAccessible(true);
			Double retMaxScoreAvg = (Double) field.get(eval);			
			assertFalse(retMaxScoreAvg == null);
			assertTrue(retMaxScoreAvg == 50);
			
			field = eval.getClass().getDeclaredField("kAccuracy");
			field.setAccessible(true);
			Double[] kAccuracy = (Double[]) field.get(eval);

			assertFalse(kAccuracy == null);
			assertTrue(kAccuracy.length == 5);
			assertTrue(kAccuracy[4] == 150);
			assertTrue(kAccuracy[3] == 150);
			assertTrue(kAccuracy[2] == 150);
			assertTrue(kAccuracy[1] <= 150);
			assertTrue(kAccuracy[0] <= 150);
			
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
	public void modelShouldBeCrossValidatedForClassifier2() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/glass.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		String ret = eval.crossValidateRankModel(new J48(), data, 3, 3);
		
		assertFalse(ret == null);
		assertFalse(ret.isEmpty());
		
		try {
			Field field = eval.getClass().getDeclaredField("resultSetForCrossValidation");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			List<List<List<Integer>>> retResultSetForCrossValidation = (List<List<List<Integer>>>) field.get(eval);			

			assertFalse(retResultSetForCrossValidation == null);
			assertFalse(retResultSetForCrossValidation.isEmpty());
			assertTrue(retResultSetForCrossValidation.size() == 3);
			assertTrue(retResultSetForCrossValidation.get(0).size() <= 72);
			assertTrue(retResultSetForCrossValidation.get(1).size() <= 72);
			assertTrue(retResultSetForCrossValidation.get(2).size() <= 72);
			
			for (List<List<Integer>> oneFoldResultSet : retResultSetForCrossValidation) {
				for (List<Integer> list : oneFoldResultSet) {
					assertTrue(list.size()==4);
					for (Integer i : list) {
						assertTrue(i>0 && i<8);
					}
					
					List<Integer> resultList = list.subList(1, list.size());				
					for (int i = 0; i < resultList.size(); i++) {
						Integer element = resultList.get(i);
						resultList.remove(i);
						assertFalse(resultList.contains(element));					
					}
				}
			}
			
			//Verify result statistics fields
			
			field = eval.getClass().getDeclaredField("maxScoreAvg");
			field.setAccessible(true);
			Double retMaxScoreAvg = (Double) field.get(eval);			
			assertFalse(retMaxScoreAvg == null);
			assertTrue(retMaxScoreAvg == 214.0/3.0);
			
			field = eval.getClass().getDeclaredField("kAccuracy");
			field.setAccessible(true);
			Double[] kAccuracy = (Double[]) field.get(eval);
			
			assertFalse(kAccuracy == null);
			assertTrue(kAccuracy.length == 5);
			assertTrue(kAccuracy[4] <= 214);
			assertTrue(kAccuracy[3] <= 214);
			assertTrue(kAccuracy[2] <= 214);
			assertTrue(kAccuracy[1] <= 214);
			assertTrue(kAccuracy[0] <= 214);
			
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
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(mr, new J48(), null, data, 3);
		
		String srt = eval.toSummaryString();
		assertFalse(srt == null);
		assertFalse(srt.isEmpty());
	}
	
	@Test
	public void csvLineShoudBeReturned() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();

		eval.crossValidateRankModel(new J48(), data, 3, null);
		
		String srt = eval.toCSVLine();
		assertFalse(srt == null);
		assertFalse(srt.isEmpty());
	}
	
	@Test
	public void allLabelKMetricsShouldBeInitiated() {
		
		RankEvaluation eval = new RankEvaluation();
		Class[] cArg = new Class[2];
		cArg[0] = Integer.class;
		cArg[1] = Integer.class;
		
		try {
		
			Method method = eval.getClass().getDeclaredMethod("initiateAllLabelKMetrics", cArg);	
			method.setAccessible(true);
			method.invoke(eval, 5, 10);
			
			Field field = eval.getClass().getDeclaredField("allLabelsKMetrics");
			field.setAccessible(true);
			Map<Integer, Map<Integer, Map<String, Integer>>> allLabelsKMetrics = (Map<Integer, Map<Integer, Map<String, Integer>>>) field.get(eval);

			assertFalse(allLabelsKMetrics == null);
			assertFalse(allLabelsKMetrics.isEmpty());
			assertTrue(allLabelsKMetrics.size() == 5);
			
			for (int i = 1; i < allLabelsKMetrics.size(); i++) {
				
				assertFalse(allLabelsKMetrics.get(i) == null);
				assertTrue(allLabelsKMetrics.get(i).size() == 10);
				
			}
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	@Test
	public void allLabelMetricsShouldBeIncremented() {
		
		RankEvaluation eval = new RankEvaluation();
		Class[] cArg = new Class[2];
		cArg[0] = Integer.class;
		cArg[1] = Integer.class;
		
		try {
		
			Method method = eval.getClass().getDeclaredMethod("initiateAllLabelKMetrics", cArg);	
			method.setAccessible(true);
			method.invoke(eval, 5, 5);
			
			Field field = eval.getClass().getDeclaredField("allLabelsKMetrics");
			field.setAccessible(true);			

			cArg = new Class[4];
			cArg[0] = Integer.class;
			cArg[1] = Integer.class;
			cArg[2] = String.class;
			cArg[3] = Double.class;
			
			//Integer kMetricIndex, Integer classLabel, String metricName
			method = eval.getClass().getDeclaredMethod("incrementLabelKMetricsField", cArg);
			method.setAccessible(true);
			
			method.invoke(eval, 1, 1, "key1", null);
			method.invoke(eval, 2, 2, "key1", null);
			method.invoke(eval, 3, 3, "key1", null);
			method.invoke(eval, 4, 4, "key1", null);
			method.invoke(eval, 5, 5, "key1", null);			
			method.invoke(eval, 1, 1, "key1", null);
			method.invoke(eval, 1, 5, "key1", null);
			method.invoke(eval, 5, 1, "key1", null);
			method.invoke(eval, 5, 5, "key1", null);
			
			method.invoke(eval, 1, 1, "key2", null);
			method.invoke(eval, 5, 5, "key2", null);
			
			method.invoke(eval, 1, 1, "key2", 0.1);
			method.invoke(eval, 5, 5, "key2", 0.1);
			
			Map<Integer, Map<Integer, Map<String, Double>>> allLabelsKMetrics = (Map<Integer, Map<Integer, Map<String, Double>>>) field.get(eval);
			
			assertTrue(allLabelsKMetrics.get(1).get(1).get("key1") == 2.0);
			assertTrue(allLabelsKMetrics.get(2).get(2).get("key1") == 1.0);
			assertTrue(allLabelsKMetrics.get(3).get(3).get("key1") == 1.0);
			assertTrue(allLabelsKMetrics.get(4).get(4).get("key1") == 1.0);
			assertTrue(allLabelsKMetrics.get(5).get(5).get("key1") == 2.0);
			assertTrue(allLabelsKMetrics.get(1).get(5).get("key1") == 1.0);
			assertTrue(allLabelsKMetrics.get(5).get(1).get("key1") == 1.0);
			assertTrue(allLabelsKMetrics.get(5).get(5).get("key2") == 1.1);
			assertTrue(allLabelsKMetrics.get(1).get(1).get("key2") == 1.1);
			assertTrue(allLabelsKMetrics.get(1).get(2).get("key1") == null);
			assertTrue(allLabelsKMetrics.get(1).get(1).get("key3") == null);
			assertTrue(allLabelsKMetrics.get(2).get(2).get("key2") == null);
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	
	@Test
	public void kMetricsForMetarankerShouldBeCalculated() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		String ret = eval.crossValidateRankModel(mr, new J48(), null, data, 3);
		Integer numLabels = data.numClasses();
		
		try {			
			//Verify the brute result set
			
			Class[] cArg = new Class[2];
			cArg[0] = Integer.class;
			cArg[1] = Integer.class;
			
			Method method = eval.getClass().getDeclaredMethod("calculateKMetricsForMetaranker", cArg);	
			method.setAccessible(true);
			method.invoke(eval, 3, numLabels);			
			
			Field field = eval.getClass().getDeclaredField("allLabelsKMetrics");
			field.setAccessible(true);
			Map<Integer, Map<Integer, Map<String, Double>>> allLabelsKMetrics = (Map<Integer, Map<Integer, Map<String, Double>>>) field.get(eval);

			assertFalse(allLabelsKMetrics == null);
			assertFalse(allLabelsKMetrics.isEmpty());
			/*
			System.out.println(allLabelsKMetrics.get(1));
			System.out.println(allLabelsKMetrics.get(2));
			System.out.println(allLabelsKMetrics.get(3));
			System.out.println();
			*/
			//Verify result statistics fields
			
			field = eval.getClass().getDeclaredField("kPrecisionMicro");
			field.setAccessible(true);
			Double[] kPrecisionMicro = (Double[]) field.get(eval);
			
			assertFalse(kPrecisionMicro == null);
			assertTrue(kPrecisionMicro.length == 3);
			assertTrue(kPrecisionMicro[2] >= kPrecisionMicro[1]);
			assertTrue(kPrecisionMicro[1] >= kPrecisionMicro[0]);
			assertTrue(kPrecisionMicro[0] >= 0);
			/*
			System.out.println("kpMi 0: " + kPrecisionMicro[0]);
			System.out.println("kpMi 1: " + kPrecisionMicro[1]);
			System.out.println("kpMi 2: " + kPrecisionMicro[2]);
			*/
			
			field = eval.getClass().getDeclaredField("kPrecisionAvg");
			field.setAccessible(true);
			Double[] kPrecisionAvg = (Double[]) field.get(eval);
			
			assertFalse(kPrecisionAvg == null);
			assertTrue(kPrecisionAvg.length == 3);
			assertTrue(kPrecisionAvg[2] >= kPrecisionAvg[1]);
			assertTrue(kPrecisionAvg[1] >= kPrecisionAvg[0]);
			assertTrue(kPrecisionAvg[0] >= 0);			
			/*
			System.out.println("kpAv 0: " + kPrecisionAvg[0]);
			System.out.println("kpAv 1: " + kPrecisionAvg[1]);
			System.out.println("kpAv 2: " + kPrecisionAvg[2]);
			*/
			
			field = eval.getClass().getDeclaredField("kPrecisionPon");
			field.setAccessible(true);
			Double[] kPrecisionPon = (Double[]) field.get(eval);
			
			assertFalse(kPrecisionPon == null);
			assertTrue(kPrecisionPon.length == 3);
			assertTrue(kPrecisionPon[2] >= kPrecisionPon[1]);
			assertTrue(kPrecisionPon[1] >= kPrecisionPon[0]);
			assertTrue(kPrecisionPon[0] >= 0);
			/*
			System.out.println("kpPo 0: " + kPrecisionPon[0]);
			System.out.println("kpPo 1: " + kPrecisionPon[1]);
			System.out.println("kpPo 2: " + kPrecisionPon[2]);			
			System.out.println();
			*/
			
			//Recall
			
			field = eval.getClass().getDeclaredField("kRecallMicro");
			field.setAccessible(true);
			Double[] kRecallMicro = (Double[]) field.get(eval);
			
			assertFalse(kRecallMicro == null);
			assertTrue(kRecallMicro.length == 3);
			assertTrue(kRecallMicro[2] >= kRecallMicro[1]);
			assertTrue(kRecallMicro[1] >= kRecallMicro[0]);
			assertTrue(kRecallMicro[0] >= 0);
			/*
			System.out.println("krMi 0: " + kRecallMicro[0]);
			System.out.println("krMi 1: " + kRecallMicro[1]);
			System.out.println("krMi 2: " + kRecallMicro[2]);	
			*/
			
			field = eval.getClass().getDeclaredField("kRecallAvg");
			field.setAccessible(true);
			Double[] kRecallAvg = (Double[]) field.get(eval);
			
			assertFalse(kRecallAvg == null);
			assertTrue(kRecallAvg.length == 3);
			assertTrue(kRecallAvg[2] >= kRecallAvg[1]);
			assertTrue(kRecallAvg[1] >= kRecallAvg[0]);
			assertTrue(kRecallAvg[0] >= 0);
			/*
			System.out.println("krAv 0: " + kRecallAvg[0]);
			System.out.println("krAv 1: " + kRecallAvg[1]);
			System.out.println("krAv 2: " + kRecallAvg[2]);	
			*/
			field = eval.getClass().getDeclaredField("kRecallPon");
			field.setAccessible(true);
			Double[] kRecallPon = (Double[]) field.get(eval);
			
			assertFalse(kRecallPon == null);
			assertTrue(kRecallPon.length == 3);
			assertTrue(kRecallPon[2] >= kRecallPon[1]);
			assertTrue(kRecallPon[1] >= kRecallPon[0]);
			assertTrue(kRecallPon[0] >= 0);
			/*
			System.out.println("krPo 0: " + kRecallPon[0]);
			System.out.println("krPo 1: " + kRecallPon[1]);
			System.out.println("krPo 2: " + kRecallPon[2]);	
			*/
		} catch (Exception e) {
			e.printStackTrace();
		}		
	}
	
	@Test
	public void kMetricsForClassifierShouldBeCalculated() {
		
		Instances data = loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		Integer numLabels = data.numClasses();
		
		RankEvaluation eval = new RankEvaluation();
		String ret = eval.crossValidateRankModel(new J48(), data, 3, null);		
		
		try {			
			//Verify the brute result set
			
			Class[] cArg = new Class[2];
			cArg[0] = Integer.class;
			cArg[1] = Integer.class;
			
			Method method = eval.getClass().getDeclaredMethod("calculateKMetricsForClassifier", cArg);	
			method.setAccessible(true);
			method.invoke(eval, 3, numLabels);
			
			Field field = eval.getClass().getDeclaredField("allLabelsKMetrics");
			field.setAccessible(true);
			Map<Integer, Map<Integer, Map<String, Double>>> allLabelsKMetrics = (Map<Integer, Map<Integer, Map<String, Double>>>) field.get(eval);

			assertFalse(allLabelsKMetrics == null);
			assertFalse(allLabelsKMetrics.isEmpty());
			/*
			System.out.println(allLabelsKMetrics.get(1));
			System.out.println(allLabelsKMetrics.get(2));
			System.out.println(allLabelsKMetrics.get(3));
			System.out.println();
			*/
			//Verify result statistics fields
			
			field = eval.getClass().getDeclaredField("kPrecisionMicro");
			field.setAccessible(true);
			Double[] kPrecisionMicro = (Double[]) field.get(eval);
			
			assertFalse(kPrecisionMicro == null);
			assertTrue(kPrecisionMicro.length == 3);
			assertTrue(kPrecisionMicro[2] >= kPrecisionMicro[1]);
			assertTrue(kPrecisionMicro[1] >= kPrecisionMicro[0]);
			assertTrue(kPrecisionMicro[0] >= 0);
			/*
			System.out.println("kpMi 0: " + kPrecisionMicro[0]);
			System.out.println("kpMi 1: " + kPrecisionMicro[1]);
			System.out.println("kpMi 2: " + kPrecisionMicro[2]);
			*/
			
			field = eval.getClass().getDeclaredField("kPrecisionAvg");
			field.setAccessible(true);
			Double[] kPrecisionAvg = (Double[]) field.get(eval);
			
			assertFalse(kPrecisionAvg == null);
			assertTrue(kPrecisionAvg.length == 3);
			assertTrue(kPrecisionAvg[2] >= kPrecisionAvg[1]);
			assertTrue(kPrecisionAvg[1] >= kPrecisionAvg[0]);
			assertTrue(kPrecisionAvg[0] >= 0);			
			/*
			System.out.println("kpAv 0: " + kPrecisionAvg[0]);
			System.out.println("kpAv 1: " + kPrecisionAvg[1]);
			System.out.println("kpAv 2: " + kPrecisionAvg[2]);
			*/
			
			field = eval.getClass().getDeclaredField("kPrecisionPon");
			field.setAccessible(true);
			Double[] kPrecisionPon = (Double[]) field.get(eval);
			
			assertFalse(kPrecisionPon == null);
			assertTrue(kPrecisionPon.length == 3);
			assertTrue(kPrecisionPon[2] >= kPrecisionPon[1]);
			assertTrue(kPrecisionPon[1] >= kPrecisionPon[0]);
			assertTrue(kPrecisionPon[0] >= 0);
			/*
			System.out.println("kpPo 0: " + kPrecisionPon[0]);
			System.out.println("kpPo 1: " + kPrecisionPon[1]);
			System.out.println("kpPo 2: " + kPrecisionPon[2]);			
			System.out.println();
			*/
			
			//Recall
			
			field = eval.getClass().getDeclaredField("kRecallMicro");
			field.setAccessible(true);
			Double[] kRecallMicro = (Double[]) field.get(eval);
			
			assertFalse(kRecallMicro == null);
			assertTrue(kRecallMicro.length == 3);
			assertTrue(kRecallMicro[2] >= kRecallMicro[1]);
			assertTrue(kRecallMicro[1] >= kRecallMicro[0]);
			assertTrue(kRecallMicro[0] >= 0);
			/*
			System.out.println("krMi 0: " + kRecallMicro[0]);
			System.out.println("krMi 1: " + kRecallMicro[1]);
			System.out.println("krMi 2: " + kRecallMicro[2]);	
			*/
			
			field = eval.getClass().getDeclaredField("kRecallAvg");
			field.setAccessible(true);
			Double[] kRecallAvg = (Double[]) field.get(eval);
			
			assertFalse(kRecallAvg == null);
			assertTrue(kRecallAvg.length == 3);
			assertTrue(kRecallAvg[2] >= kRecallAvg[1]);
			assertTrue(kRecallAvg[1] >= kRecallAvg[0]);
			assertTrue(kRecallAvg[0] >= 0);
			/*
			System.out.println("krAv 0: " + kRecallAvg[0]);
			System.out.println("krAv 1: " + kRecallAvg[1]);
			System.out.println("krAv 2: " + kRecallAvg[2]);	
			*/
			field = eval.getClass().getDeclaredField("kRecallPon");
			field.setAccessible(true);
			Double[] kRecallPon = (Double[]) field.get(eval);
			
			assertFalse(kRecallPon == null);
			assertTrue(kRecallPon.length == 3);
			assertTrue(kRecallPon[2] >= kRecallPon[1]);
			assertTrue(kRecallPon[1] >= kRecallPon[0]);
			assertTrue(kRecallPon[0] >= 0);
			/*
			System.out.println("krPo 0: " + kRecallPon[0]);
			System.out.println("krPo 1: " + kRecallPon[1]);
			System.out.println("krPo 2: " + kRecallPon[2]);	
			*/
		} catch (Exception e) {
			e.printStackTrace();
		}		
	}
}

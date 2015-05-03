package br.puc.drm.ranker.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Arrays;
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
		mr.buildClassifier(new J48(), data, null);
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
			eval.evaluateRankModel(cls, null, 3);
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
		eval.evaluateRankModel(cls, data, 3);
		
		
	}

	//If the Classifier is not trained
	@Test (expected = IllegalStateException.class)
	public void illegalStateExceptionShouldBeReturnedByEvaluateRankModelForClassifiers() {
		
		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		eval.evaluateRankModel(new J48(), data, 3);		
		
	}
	
	@Test
	public void modelShouldBeEvaluatedForAClassifier() {
		
		Instances data = loadTestFile("iris.arff");
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
				for (Integer i : list) {
					assertTrue(i>0 && i<4);
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
				for (Double d : probDist) {
					assertTrue(d >= 0 && d <=1);
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
	public void modelShouldBeEvaluatedForAClassifier2() {
		
		Instances data = loadTestFile("glass.arff");
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
				for (Double d : probDist) {
					assertTrue(d >= 0 && d <=1);
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
		eval.crossValidateRankModel(mr, new J48(), null, data, 3, new Random(1));
	}

	//If the data is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel2() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(mr, new J48(), null, null, 3, new Random(1));
	}

	//If the number of folds is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel3() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		Integer n = null;
		eval.crossValidateRankModel(mr, new J48(), null, data, n , new Random(1));
	}
	
	//If the random is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel4() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		Random r = null;
		eval.crossValidateRankModel(mr, new J48(), null, data, 3, r);
	}
	
	//If the number of folds is less than 2
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel5() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(mr, new J48(), null, data, 1, new Random(1));
	}
	
	//If the number of folds is greater than the number of instances.
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel6() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(mr, new J48(), null, data, data.numInstances()+1, new Random(1));
	}
	
	//If the class is not nominal
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModel8() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-2);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(mr, new J48(), null, data, 3, new Random(1));
	}

	@Test
	public void modelShouldBeCrossValidatedForMetaRanker1() {
		
		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		String ret = eval.crossValidateRankModel(mr, new J48(), null, data, 3, new Random(1));
		
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
		
		Instances data = loadTestFile("glass.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		mr.setRankSize(3);
		RankEvaluation eval = new RankEvaluation();
		String ret = eval.crossValidateRankModel(mr, new J48(), null, data, 3, new Random(1));
		
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
					
					Integer element = list.get(1); 
					for (int i = 2; i < 4; i++) {
						assertTrue(element != list.get(i));
						element = list.get(i);
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
	
	//CrossValidateRankModel (for Classifier) tests
	
	//If the Classifier is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS1() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		Classifier cls = null;
		eval.crossValidateRankModel(cls, data, 3, new Random(1), null);
	}

	//If the data is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS2() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(new J48(), null, 3, new Random(1), null);
	}

	//If the number of folds is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS3() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		Integer n = null;
		eval.crossValidateRankModel(new J48(), data, n , new Random(1), null);
	}
	
	//If the random is null
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS4() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		Random r = null;
		eval.crossValidateRankModel(new J48(), data, 3, r, null);
	}
	
	//If the number of folds is less than 2
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS5() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(new J48(), data, 1, new Random(1), null);
	}
	
	//If the number of folds is greater than the number of instances.
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS6() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(new J48(), data, data.numInstances()+1, new Random(1), null);
	}
	
	//If the class is not nominal
	@Test  (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptioShouldBereturnedByCrossValidateRankModelCLS8() {

		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-2);
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(new J48(), data, 3, new Random(1), null);
	}
	
	@Test
	public void modelShouldBeCrossValidatedForClassifier1() {
		
		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		String ret = eval.crossValidateRankModel(new J48(), data, 3, new Random(1), null);
		
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
		
		Instances data = loadTestFile("glass.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();
		String ret = eval.crossValidateRankModel(new J48(), data, 3, new Random(1), 3);
		
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
					
					Integer element = list.get(1); 
					for (int i = 2; i < 4; i++) {
						assertTrue(element != list.get(i));
						element = list.get(i);
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
		
		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		MetaRanker mr = new MetaRanker();
		RankEvaluation eval = new RankEvaluation();
		eval.crossValidateRankModel(mr, new J48(), null, data, 3, new Random());
		
		String srt = eval.toSummaryString();
		assertFalse(srt == null);
		assertFalse(srt.isEmpty());
	}
	
	@Test
	public void csvLineShoudBeReturned() {
		
		Instances data = loadTestFile("iris.arff");
		data.setClassIndex(data.firstInstance().numAttributes()-1);
		RankEvaluation eval = new RankEvaluation();

		eval.crossValidateRankModel(new J48(), data, 3, new Random(), null);
		
		String srt = eval.toCSVLine();
		assertFalse(srt == null);
		assertFalse(srt.isEmpty());
	}	
}

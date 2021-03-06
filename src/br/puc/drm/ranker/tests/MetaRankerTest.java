package br.puc.drm.ranker.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.supervised.instance.Resample;
import br.puc.drm.ranker.MetaRanker;

public class MetaRankerTest {

	private Instances data;
	
	public void loadTestFile(String fileName) {
	
		ArffLoader loader = new ArffLoader();
	    try {
	    	
			loader.setFile(new File(fileName));
			this.data = loader.getDataSet();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}    
	}
	
	@Test
	public void subSetsShouldBeGenerated() {
	
		Set<Integer> inputSet = new HashSet<Integer>();
		List<Set<Integer>> refTemplateList = new ArrayList<Set<Integer>>();
		Set<Integer> tempSet;
		
		inputSet.add(1);
		inputSet.add(2);
		inputSet.add(3);
		inputSet.add(4);
		
		
		tempSet = new HashSet<Integer>();
		tempSet.add(1);
		refTemplateList.add(tempSet);

		tempSet = new HashSet<Integer>();
		tempSet.add(1);
		tempSet.add(2);
		refTemplateList.add(tempSet);
		
		tempSet = new HashSet<Integer>();
		tempSet.add(2);
		refTemplateList.add(tempSet);
		
		tempSet = new HashSet<Integer>();
		tempSet.add(1);
		tempSet.add(3);
		refTemplateList.add(tempSet);
		
		tempSet = new HashSet<Integer>();
		tempSet.add(1);
		tempSet.add(2);
		tempSet.add(3);
		refTemplateList.add(tempSet);
		
		tempSet = new HashSet<Integer>();
		tempSet.add(2);
		tempSet.add(3);
		refTemplateList.add(tempSet);
		
		tempSet = new HashSet<Integer>();
		tempSet.add(3);
		refTemplateList.add(tempSet);
		
		tempSet = new HashSet<Integer>();
		tempSet.add(1);
		tempSet.add(4);
		refTemplateList.add(tempSet);
		
		tempSet = new HashSet<Integer>();
		tempSet.add(1);
		tempSet.add(2);
		tempSet.add(4);
		refTemplateList.add(tempSet);
		
		tempSet = new HashSet<Integer>();
		tempSet.add(2);
		tempSet.add(4);
		refTemplateList.add(tempSet);
		
		tempSet = new HashSet<Integer>();
		tempSet.add(1);
		tempSet.add(3);
		tempSet.add(4);
		refTemplateList.add(tempSet);
		
		tempSet = new HashSet<Integer>();
		tempSet.add(1);
		tempSet.add(2);
		tempSet.add(3);
		tempSet.add(4);
		refTemplateList.add(tempSet);

		tempSet = new HashSet<Integer>();
		tempSet.add(2);
		tempSet.add(3);
		tempSet.add(4);
		refTemplateList.add(tempSet);
		
		tempSet = new HashSet<Integer>();
		tempSet.add(3);
		tempSet.add(4);
		refTemplateList.add(tempSet);
		
		tempSet = new HashSet<Integer>();
		tempSet.add(4);
		refTemplateList.add(tempSet);
		
		try {
			
			MetaRanker testlMr = new MetaRanker();
			Class[] cArg = new Class[1];
			cArg[0] = Set.class;
			Method method = testlMr.getClass().getDeclaredMethod("generateIntSubSets", cArg);			
			
			method.setAccessible(true);
			
			@SuppressWarnings("unchecked")
			List<Set<Integer>> subSets = ((List<Set<Integer>>) method.invoke(testlMr, inputSet));

			for (int i = 1; i < subSets.size(); i++) {
			    assertTrue(subSets.get(i).equals(refTemplateList.get(i)));
			}
			
		} catch (NoSuchMethodException | SecurityException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalArgumentException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvocationTargetException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByBuildClassifier() {
		
		MetaRanker testMr = new MetaRanker();
		testMr.buildClassifier(new J48(), null, null);
		
	}

	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByBuildClassifier2() {
		
		this.loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		MetaRanker testMr = new MetaRanker();
		testMr.buildClassifier(null, this.data, null);
		
		this.data = null;
		
	}
	
	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByBuildClassifier3() {
		
		this.loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		
		MetaRanker testMr = new MetaRanker();
		
		testMr.setRankSize(1000);
		
		testMr.buildClassifier(new J48(), this.data, null);
		
	}
	
	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByClassityInstance() {
		
		MetaRanker testMr = new MetaRanker();
		testMr.classifyInstance(null);
	}
	
	@Test (expected = IllegalStateException.class)
	public void illegalStateExceptionShouldBeReturnedByClassityInstance() {
		
		MetaRanker testMr = new MetaRanker();
		
		this.loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		
		List<Integer> ret = testMr.classifyInstance(this.data.get(1));		
		
	}
	
	@Test
	public void classifierShoudBeBuilt1() {		
		
		this.loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		
		MetaRanker testMr = new MetaRanker();
		
		testMr.buildClassifier(new J48(), this.data, "-U");
		
		assertTrue(testMr.getClassifierOptions().contains("-U"));
		
		Class<? extends MetaRanker> cls = testMr.getClass();
		
		try {
			Field field = cls.getDeclaredField("classifiers");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			Map<Set<Integer>, Classifier> retClassifiers = (Map<Set<Integer>, Classifier>) field.get(testMr);
			
			assertFalse(retClassifiers == null);
			assertFalse("Returned Empty classifier.", retClassifiers.isEmpty());
			assertTrue("Wrong map key-value number of pairs.", retClassifiers.size()==4);
			
			Classifier oldClassifier = null;
			for (Classifier classifier : retClassifiers.values()) {
				assertFalse(classifier == oldClassifier);
				assertTrue(classifier.getClass() == J48.class);
				oldClassifier = classifier;
			}	

		} catch (NoSuchFieldException | SecurityException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalArgumentException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (NullPointerException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();;
		}
		
		this.data=null;
		
	}
	
	@Test
	public void classifierShoulBeBuilt2() {
		
		this.loadTestFile("/home/daniel/workspace/RankerTestFiles/glass.arff");
		
		MetaRanker testMr = new MetaRanker();
		
		testMr.buildClassifier(new J48(), this.data, null);

		Class<? extends MetaRanker> cls = testMr.getClass();
		
		try {
			Field field = cls.getDeclaredField("classifiers");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			Map<Set<Integer>, Classifier> retClassifiers = (Map<Set<Integer>, Classifier>) field.get(testMr);
			
			assertFalse(retClassifiers == null);
			assertFalse("Returned Empty classifier.", retClassifiers.isEmpty());
			assertTrue("Wrong map key-value number of pairs.", retClassifiers.size()==120);
			
			Classifier oldClassifier = null;
			for (Classifier classifier : retClassifiers.values()) {
				assertFalse(classifier == oldClassifier);
				assertTrue(classifier.getClass() == J48.class);
				oldClassifier = classifier;
			}		

		} catch (NoSuchFieldException | SecurityException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalArgumentException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (NullPointerException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		this.data=null;
	}

	@Test
	public void classifierShoulBeBuilt3() {
		
		this.loadTestFile("/home/daniel/workspace/RankerTestFiles/glass.arff");
		
		MetaRanker testMr = new MetaRanker();
		
		testMr.setRankSize(3);
		testMr.buildClassifier(new J48(), this.data, null);

		Class<? extends MetaRanker> cls = testMr.getClass();
		
		try {
			Field field = cls.getDeclaredField("classifiers");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			Map<Set<Integer>, Classifier> retClassifiers = (Map<Set<Integer>, Classifier>) field.get(testMr);
			
			assertFalse(retClassifiers == null);
			assertFalse("Returned Empty classifier.", retClassifiers.isEmpty());
			assertTrue("Wrong map key-value number of pairs.", retClassifiers.size()==29);
			
			Classifier oldClassifier = null;
			for (Classifier classifier : retClassifiers.values()) {
				assertFalse(classifier == oldClassifier);
				assertTrue(classifier.getClass() == J48.class);
				oldClassifier = classifier;
			}		

		} catch (NoSuchFieldException | SecurityException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalArgumentException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (NullPointerException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		this.data=null;
	}
	
	@Test
	public void instanceShoudBeClassified() {
		
		List<Integer> retList;
		
		this.loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		this.data.setClassIndex(this.data.numAttributes()-1);
		MetaRanker testMr = new MetaRanker();
		testMr.buildClassifier(new J48(), this.data, null);
		
		Resample sampler = new Resample();
		String Fliteroptions="-B 1 -Z 10";
		try {
			
			sampler.setOptions(weka.core.Utils.splitOptions(Fliteroptions));
			sampler.setInputFormat(this.data);
			sampler.setRandomSeed((int)System.currentTimeMillis());
			
			Instances sampleData = Resample.useFilter(this.data, sampler);
			
			for (int i = 1; i < sampleData.size(); i++) {
				
					retList = testMr.classifyInstance(sampleData.get(i));

					assertFalse("Returned list is null.", retList == null);
					assertFalse("Returned list is empty.", retList.isEmpty());
					assertTrue("Wrong list size.", retList.size()==3);		
					
					for (int j = 1; j <= this.data.classAttribute().numValues(); j++) {
						assertTrue("Missing element on ranking list.", retList.contains(j));
					}					
					
					int oldClassIndex = -1;
					for (Integer classIndex : retList) {						
						assertFalse("Redundant class index", classIndex == oldClassIndex);
						oldClassIndex = classIndex;
					}
			}		
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();			
			
		}
	}

	@Test
	public void instanceShoudBeClassified2() {
		
		List<Integer> retList;
		
		this.loadTestFile("/home/daniel/workspace/RankerTestFiles/glass.arff");
		this.data.setClassIndex(this.data.numAttributes()-1);
		MetaRanker testMr = new MetaRanker();
		testMr.buildClassifier(new J48(), this.data, null);
		
		Resample sampler = new Resample();
		String filterOptions="-B 1 -Z 10";
		try {
			
			sampler.setOptions(weka.core.Utils.splitOptions(filterOptions));
			sampler.setInputFormat(this.data);
			sampler.setRandomSeed((int)System.currentTimeMillis());
			
			Instances sampleData = Resample.useFilter(this.data, sampler);
			
			for (int i = 1; i < sampleData.size(); i++) {
				
					retList = testMr.classifyInstance(sampleData.get(i));

					assertFalse("Returned list is null.", retList == null);
					assertFalse("Returned list is empty.", retList.isEmpty());
					assertTrue("Wrong list size.", retList.size()==7);		
					
					for (int j = 1; j <= this.data.classAttribute().numValues(); j++) {
						assertTrue("Missing element on ranking list.", retList.contains(j));
					}					
					
					int oldClassIndex = -1;
					for (Integer classIndex : retList) {						
						assertFalse("Redundant class index", classIndex == oldClassIndex);
						oldClassIndex = classIndex;
					}
			}		
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();			
			
		}
	}
	
	@Test
	public void instanceShoudBeClassified3() {
		
		List<Integer> retList;
		
		this.loadTestFile("/home/daniel/workspace/RankerTestFiles/glass.arff");
		this.data.setClassIndex(this.data.numAttributes()-1);
		MetaRanker testMr = new MetaRanker();
		testMr.setRankSize(4);
		testMr.buildClassifier(new J48(), this.data, null);
		
		Resample sampler = new Resample();
		String filterOptions= "-B 1 -Z 10";
		
		try {
			
			sampler.setOptions(weka.core.Utils.splitOptions(filterOptions));
			sampler.setInputFormat(this.data);
			sampler.setRandomSeed((int)System.currentTimeMillis());
			
			Instances sampleData = Resample.useFilter(this.data, sampler);
			
			for (int i = 1; i < sampleData.size(); i++) {
				
					retList = testMr.classifyInstance(sampleData.get(i));
					assertFalse("Returned list is null.", retList == null);
					assertFalse("Returned list is empty.", retList.isEmpty());
					assertTrue("Wrong list size.", retList.size()==4);
					
					int oldClassIndex = -1;
					for (Integer classIndex : retList) {
						assertTrue(classIndex >=1 && classIndex <= 7);
						assertFalse("Redundant class index", classIndex == oldClassIndex);
						oldClassIndex = classIndex;
					}
			}		
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();			
			
		}
	}

	@Test
	public void instanceShoudBeClassified4() {
		
		List<Integer> retList;
		
		this.loadTestFile("/home/daniel/workspace/RankerTestFiles/segment-challenge.arff");
		this.data.setClassIndex(this.data.numAttributes()-1);
		MetaRanker testMr = new MetaRanker();
		
		long startTime = System.nanoTime();
		testMr.buildClassifier(new J48(), this.data, null);
		
		long elapsedTime = System.nanoTime()-startTime;
		//System.out.println("Elapsed time: " + elapsedTime);
		
		Resample sampler = new Resample();
		String filterOptions= "-B 1 -Z 10";
		
		try {
			
			sampler.setOptions(weka.core.Utils.splitOptions(filterOptions));
			sampler.setInputFormat(this.data);
			sampler.setRandomSeed((int)System.currentTimeMillis());
			
			Instances sampleData = Resample.useFilter(this.data, sampler);
			
			for (int i = 1; i < sampleData.size(); i++) {
				
					retList = testMr.classifyInstance(sampleData.get(i));
					assertFalse("Returned list is null.", retList == null);
					assertFalse("Returned list is empty.", retList.isEmpty());
					assertTrue("Wrong list size.", retList.size()==7);
					
					int oldClassIndex = -1;
					for (Integer classIndex : retList) {
						assertTrue(classIndex >=1 && classIndex <= 7);
						assertFalse("Redundant class index", classIndex == oldClassIndex);
						oldClassIndex = classIndex;
					}
			}		
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();			
			
		}
	}
	
	@Test
	public void instanceShoudBeClassifiedDynamic() {
		
		List<Integer> retList;
		
		this.loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		this.data.setClassIndex(this.data.numAttributes()-1);
		MetaRanker testMr = new MetaRanker();
		
		Resample sampler = new Resample();
		String Fliteroptions="-B 1 -Z 10";
		try {
			
			sampler.setOptions(weka.core.Utils.splitOptions(Fliteroptions));
			sampler.setInputFormat(this.data);
			sampler.setRandomSeed((int)System.currentTimeMillis());
			
			Instances sampleData = Resample.useFilter(this.data, sampler);
			
			for (int i = 1; i < sampleData.size(); i++) {
				
					retList = testMr.classifyInstanceDynamic(sampleData.get(i), data, new J48(), null);

					assertFalse("Returned list is null.", retList == null);
					assertFalse("Returned list is empty.", retList.isEmpty());
					assertTrue("Wrong list size.", retList.size()==3);		
					
					for (int j = 1; j <= this.data.classAttribute().numValues(); j++) {
						assertTrue("Missing element on ranking list.", retList.contains(j));
					}					
					
					int oldClassIndex = -1;
					for (Integer classIndex : retList) {						
						assertFalse("Redundant class index", classIndex == oldClassIndex);
						oldClassIndex = classIndex;
					}
			}		
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();			
			
		}
	}

	@Test
	public void instanceShoudBeClassifiedDynamic2() {
		
		List<Integer> retList;
		
		this.loadTestFile("/home/daniel/workspace/RankerTestFiles/glass.arff");
		this.data.setClassIndex(this.data.numAttributes()-1);
		MetaRanker testMr = new MetaRanker();
		testMr.setRankSize(4);
		
		Resample sampler = new Resample();
		String filterOptions= "-B 1 -Z 10";
		
		try {
			
			sampler.setOptions(weka.core.Utils.splitOptions(filterOptions));
			sampler.setInputFormat(this.data);
			sampler.setRandomSeed((int)System.currentTimeMillis());
			
			Instances sampleData = Resample.useFilter(this.data, sampler);
			
			for (int i = 1; i < sampleData.size(); i++) {
				
					retList = testMr.classifyInstanceDynamic(sampleData.get(i), data, new J48(), null);
					assertFalse("Returned list is null.", retList == null);
					assertFalse("Returned list is empty.", retList.isEmpty());
					assertTrue("Wrong list size.", retList.size()==4);
					
					int oldClassIndex = -1;
					for (Integer classIndex : retList) {
						assertTrue(classIndex >=1 && classIndex <= 7);
						assertFalse("Redundant class index", classIndex == oldClassIndex);
						oldClassIndex = classIndex;
					}
			}		
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();			
			
		}
	}

	@Test
	public void instanceShoudBeClassifiedDynamic3() {
		
		List<Integer> retList;
		
		this.loadTestFile("/home/daniel/workspace/RankerTestFiles/segment-challenge.arff");
		this.data.setClassIndex(this.data.numAttributes()-1);
		MetaRanker testMr = new MetaRanker();
		
		Resample sampler = new Resample();
		String filterOptions= "-B 1 -Z 5";
		
		try {
			
			sampler.setOptions(weka.core.Utils.splitOptions(filterOptions));
			sampler.setInputFormat(this.data);
			sampler.setRandomSeed((int)System.currentTimeMillis());			
			
			Instances sampleData = Resample.useFilter(this.data, sampler);
			
			long startTime = System.nanoTime();
			for (int i = 1; i < sampleData.size(); i++) {
				
					retList = testMr.classifyInstanceDynamic(sampleData.get(i), data, new J48(), null);
					assertFalse("Returned list is null.", retList == null);
					assertFalse("Returned list is empty.", retList.isEmpty());
					assertTrue("Wrong list size.", retList.size()==7);
					
					int oldClassIndex = -1;
					for (Integer classIndex : retList) {
						assertTrue(classIndex >=1 && classIndex <= 7);
						assertFalse("Redundant class index", classIndex == oldClassIndex);
						oldClassIndex = classIndex;
					}
			}
			
			long elapsedTime = System.nanoTime()-startTime;
			//System.out.println("Elapsed time: " + elapsedTime);
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();			
			
		}
	}
	
	@Test
	public void instanceShoudGetTheSameClassification() {
		
		List<Integer> retListDynamic;
		List<Integer> retList;
		
		this.loadTestFile("/home/daniel/workspace/RankerTestFiles/iris.arff");
		this.data.setClassIndex(this.data.numAttributes()-1);
		MetaRanker testMr = new MetaRanker();
		testMr.buildClassifier(new J48(), data, null);
		MetaRanker testMrDynamic = new MetaRanker();
		
		Resample sampler = new Resample();
		String Fliteroptions="-B 1 -Z 10";
		try {
			
			sampler.setOptions(weka.core.Utils.splitOptions(Fliteroptions));
			sampler.setInputFormat(this.data);
			sampler.setRandomSeed((int)System.currentTimeMillis());
			
			Instances sampleData = Resample.useFilter(this.data, sampler);
			
			for (int i = 1; i < sampleData.size(); i++) {
				
					retListDynamic = testMrDynamic.classifyInstanceDynamic(sampleData.get(i), data, new J48(), null);
					retList = testMr.classifyInstance(sampleData.get(i));

					assertTrue("Wrong list size.", retList.size()==retListDynamic.size());		
					
					for (int j = 0; j < retList.size(); j++) {
						assertTrue(retList.get(j)==retListDynamic.get(j));
					}
			}		
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();			
			
		}
	}

	@Test
	public void instanceShoudGetTheSameClassification2() {
		
		List<Integer> retListDynamic;
		List<Integer> retList;
		
		this.loadTestFile("/home/daniel/workspace/RankerTestFiles/glass.arff");
		this.data.setClassIndex(this.data.numAttributes()-1);
		MetaRanker testMr = new MetaRanker();
		testMr.setRankSize(4);
		testMr.buildClassifier(new J48(), data, null);
		MetaRanker testMrDynamic = new MetaRanker();
		testMrDynamic.setRankSize(4);
		
		Resample sampler = new Resample();
		String Fliteroptions="-B 1 -Z 10";
		try {
			
			sampler.setOptions(weka.core.Utils.splitOptions(Fliteroptions));
			sampler.setInputFormat(this.data);
			sampler.setRandomSeed((int)System.currentTimeMillis());
			
			Instances sampleData = Resample.useFilter(this.data, sampler);
			
			for (int i = 1; i < sampleData.size(); i++) {
				
					retListDynamic = testMrDynamic.classifyInstanceDynamic(sampleData.get(i), data, new J48(), null);
					retList = testMr.classifyInstance(sampleData.get(i));

					assertTrue("Wrong list size.", retList.size()==retListDynamic.size());		
					
					for (int j = 0; j < retList.size(); j++) {
						assertTrue(retList.get(j)==retListDynamic.get(j));
					}
			}		
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();			
			
		}
	}
	
	@Test
	public void optionsShoudBeSet() {
		
		MetaRanker mr = new MetaRanker();
		Classifier cls = new J48();
		Class[] cArg = new Class[2];
		cArg[0] = Classifier.class;
		cArg[1] = String.class;
		
		try {
			
			Method method = mr.getClass().getDeclaredMethod("setClassifierOptions", cArg);
			method.setAccessible(true);
			String retString = (String) method.invoke(mr, cls, "-U");
			
			assertFalse(retString.isEmpty());
			
			String[] ret = ((J48) cls).getOptions();
			assertTrue(ret[0] == "-U");			
			
		} catch (NoSuchMethodException | SecurityException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalArgumentException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvocationTargetException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	@Test
	public void classifierUseShouldbeIncremented() {

		MetaRanker mr = new MetaRanker();
		Class[] cArg = new Class[1];
		cArg[0] = Set.class;

		try {
			
			Method method = mr.getClass().getDeclaredMethod("incrementClassifierUse", cArg);
			method.setAccessible(true);
			
			for (int i = 0; i < 10; i++) {
				HashSet<Integer> key = new HashSet<Integer>();
				key.add(1);
				method.invoke(mr, key);
			}
			
			Field field = mr.getClass().getDeclaredField("classifiersUses");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			Map<Set<Integer>, Integer> retClassifierUses = (Map<Set<Integer>, Integer>) field.get(mr);
			
			HashSet<Integer> key = new HashSet<Integer>();
			key.add(1);
			assertTrue(retClassifierUses!=null);
			assertTrue(retClassifierUses.size()==1);
			assertTrue(retClassifierUses.get(key)==10);
			
		} catch (NoSuchMethodException | SecurityException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalArgumentException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvocationTargetException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (NoSuchFieldException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}


	}
}

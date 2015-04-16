package br.puc.drm.ranker.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import br.puc.drm.ranker.MetaRanker;

public class MetaRankerTest {

	private MetaRanker mr = new MetaRanker();
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
		
		this.mr.buildClassifier(new J48(), null);
		
	}

	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByBuildClassifier2() {
		
		this.loadTestFile("iris.arff");
		
		this.mr.buildClassifier(null, this.data);
		
		this.data = null;
		
	}
	
	@Test (expected = IllegalArgumentException.class)
	public void illegalArgumentExceptionShouldBeReturnedByClassityInstance() {
		
		this.mr.classifyInstance(null);
	}
	
	@Test
	public void classifierShoudBeBuilt() {		
		
		this.loadTestFile("iris.arff");
		
		this.mr.buildClassifier(new J48(), this.data);

		Class<? extends MetaRanker> cls = mr.getClass();
		
		try {
			Field field = cls.getDeclaredField("classifiers");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			Map<Set<Integer>, Classifier> retClassifiers = (Map<Set<Integer>, Classifier>) field.get(this.mr);
			
			assertFalse("Returned Empty classifier.", retClassifiers.isEmpty());
			assertTrue("Wrong map key-value number of pairs.", retClassifiers.size()==4);
			
			Set<Integer> keySet = new HashSet<Integer>();
			keySet.add(0);
			assertTrue(retClassifiers.get(keySet).getClass()== J48.class);			

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
		
		this.loadTestFile("glass.arff");
		
		this.mr.buildClassifier(new J48(), this.data);

		Class<? extends MetaRanker> cls = mr.getClass();
		
		try {
			Field field = cls.getDeclaredField("classifiers");
			field.setAccessible(true);
			@SuppressWarnings("unchecked")
			Map<Set<Integer>, Classifier> retClassifiers = (Map<Set<Integer>, Classifier>) field.get(this.mr);
			
			assertFalse("Returned Empty classifier.", retClassifiers.isEmpty());
			assertTrue("Wrong map key-value number of pairs.", retClassifiers.size()==120);
			
			Set<Integer> keySet = new HashSet<Integer>();
			keySet.add(0);
			assertTrue(retClassifiers.get(keySet).getClass()== J48.class);			

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
		
		this.loadTestFile("iris.arff");
		this.data.setClassIndex(data.numAttributes()-1);
		
		for (int i = 1; i < 10; i++) {
			
			try {
				retList = mr.classifyInstance(this.data.get(i));
				assertFalse("Returned list is empty.", retList.isEmpty());
				assertTrue("Wrong list size.", retList.size()==3);				
				
				for (int j = 1; j < this.data.classAttribute().numValues()+1; j++) {
					assertTrue("Missing element on ranking list.", retList.contains(j));
				}
				
			} catch (Exception e) {
				// TODO: handle exception
				e.printStackTrace();
			}
			
			
		}
	}
}

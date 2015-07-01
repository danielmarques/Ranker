package br.puc.drm.ranker.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import org.junit.Test;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import br.puc.drm.ranker.experiments.RankExp;

public class RankExpTest {

	@Test
	public void mainTest1() {
		
		String[] args = new String[1];
		args[0] = "/home/daniel/workspace/RankerExp";
		
		RankExp exp = new RankExp();
		
		exp.main(args);
	}

	@Test
	public void mainTest2() {
		
		String[] args = new String[1];
		args[0] = "/home/daniel/workspace/RankerExp2";
		
		RankExp exp = new RankExp();
		
		exp.main(args);
	}

	@Test
	public void mainTest3() {
		
		String[] args = new String[1];
		args[0] = "/home/daniel/workspace/RankerExp3";
		
		RankExp exp = new RankExp();
		
		exp.main(args);
	}
	
	@Test
	public void histogramShouldBeGenerated() {
		
		RankExp exp = new RankExp();
		Class[] cArgs = new Class[1];
		cArgs[0] = Instances.class;
		
		ArffLoader loader = new ArffLoader();
		
	    try {
	    	
	    	File file = new File("/home/daniel/workspace/RankerTestFiles/iris.arff");
			loader.setFile(file);
			Instances data = loader.getDataSet();
			data.setClassIndex(data.numAttributes()-1);
			
			Method method = exp.getClass().getDeclaredMethod("attributeHistogram", cArgs);
			method.setAccessible(true);
			String ret = (String) method.invoke(exp, data);
			
			assertFalse(ret == null);
			assertFalse(ret.isEmpty());
			assertFalse(ret.contains(", "));
			assertTrue(ret.length() == 8);
			assertTrue(ret.substring(0, 2).equals("50"));
			assertTrue(ret.substring(3, 5).equals("50"));
			assertTrue(ret.substring(6, 8).equals("50"));
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (NoSuchMethodException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (SecurityException e) {
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
	public void classifierShouldBeReturned() {
		
		RankExp exp = new RankExp();
		Class[] cArgs = new Class[2];
		cArgs[0] = String.class;
		cArgs[1] = String.class;
		try {
			
			Method method = exp.getClass().getDeclaredMethod("getClassifier", cArgs);
			method.setAccessible(true);
			
			Classifier cls = (Classifier) method.invoke(exp, "J48", "-U");			
			assertTrue(cls.getClass() == J48.class);

			cls = (Classifier) method.invoke(exp, "J48", null);			
			assertTrue(cls.getClass() == J48.class);

			cls = (Classifier) method.invoke(exp, "NaiveBayes", "-K");			
			assertTrue(cls.getClass() == NaiveBayes.class);
			
			cls = (Classifier) method.invoke(exp, "NaiveBayes", null);			
			assertTrue(cls.getClass() == NaiveBayes.class);

			cls = (Classifier) method.invoke(exp, "SMO", "-N 1 -M");			
			assertTrue(cls.getClass() == SMO.class);
			
			cls = (Classifier) method.invoke(exp, "SMO", null);			
			assertTrue(cls.getClass() == SMO.class);
			
			cls = (Classifier) method.invoke(exp, "IBk", null);			
			assertTrue(cls.getClass() == IBk.class);
			
			
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
	public void testFileShoulBeLoaded() {
		
		RankExp exp = new RankExp();
		Class[] cArgs = new Class[1];
		cArgs[0] = File.class;
		
		Method method;
		try {
			
			method = exp.getClass().getDeclaredMethod("loadTestFile", cArgs);
			method.setAccessible(true);
			
			File file = new File("/home/daniel/workspace/RankerTestFiles/iris.arff");
			Instances data = (Instances) method.invoke(exp, file);			
			assertFalse(data.isEmpty());
			
		} catch (NoSuchMethodException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (SecurityException e) {
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
}

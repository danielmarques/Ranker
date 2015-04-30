package br.puc.drm.ranker.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import org.junit.Test;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Instances;
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
			
			File file = new File("/home/daniel/workspace/RankerExpFiles/iris.arff");
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

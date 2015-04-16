package br.puc.drm.ranker;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class MetaRanker {

	private Map<Set<Integer>, Classifier> classifiers;
	
	private List<Set<Integer>> generateIntSubSets(Set<Integer> inputSet) {
		
		List<Set<Integer>> subSets = new ArrayList<Set<Integer>>();
		
		for(Integer addToSets:inputSet) {
			
		    List<Set<Integer>> newSets = new ArrayList<Set<Integer>>();
		    
		    for(Set<Integer> curSet:subSets) {
		    	
		        Set<Integer> copyPlusNew = new HashSet<Integer>();
		        copyPlusNew.addAll(curSet);
		        copyPlusNew.add(addToSets);
		        newSets.add(copyPlusNew);
		        
		    }
		    
		    Set<Integer> newValSet = new HashSet<Integer>();
		    newValSet.add(addToSets);
		    newSets.add(newValSet);
		    subSets.addAll(newSets);
		}
		
		return subSets;
		
	}
	
	public void buildClassifier(Classifier cls, Instances data) {
		
		if (cls == null) {
			throw new IllegalArgumentException("Invalid classifier.");
		}
		
		if (data == null) {
			throw new IllegalArgumentException("Invalid input data.");
		}
		
		// Variable Declarations and Initializations		
		this.classifiers = new HashMap<Set<Integer>, Classifier>();		

		//Verify if the class index is set, use default as last otherwise
		if (data.classIndex()<0) {
			
			data.setClassIndex(data.numAttributes()-1);			
			
		}

		//Retrieves the class attribute		
		
		try {
			
			Set<Integer> key = new HashSet<Integer>();
			key.add(0);
			
			//Uses reflection to get cls class and generate a new instance
			Classifier tempCls = cls.getClass().newInstance();
			
			//Build and stores the classifier
			tempCls.buildClassifier(data);
			this.classifiers.put(Collections.unmodifiableSet(key), tempCls);
			
			//Generate key subsets for this.classifiers map
			Set<Integer> inputSet = new HashSet<Integer>();
			for (int i = 1; i < data.classAttribute().numValues()+1; i++) {
				inputSet.add(i);
			}
			List<Set<Integer>> clsKeys = generateIntSubSets(inputSet);

			//Create the other classifiers		    
			for (Set<Integer> keySet : clsKeys) {
				
				//Filters the data to remove instances with class values identified by the keySet
				//But should leave the class attribute with at least 2 values
				if (keySet.size() <= data.classAttribute().numValues()-2) {					
					
					//Setting filter options
					String[] options = new String[4];
				    options[0] = "-C";
				    options[1] = Integer.toString(data.classIndex()+1);
				    options[2] = "-L";
				    options[3] = keySet.toString().substring(1, keySet.toString().length()-1);
				    
					//Apply filter
				    RemoveWithValues rwv = new RemoveWithValues();
					rwv.setOptions(options);
					rwv.setInputFormat(data);
					Instances tmpData = Filter.useFilter(data, rwv);
					
					//Uses reflection to get cls class and generate a new instance
					tempCls = cls.getClass().newInstance();
					
					//Build and stores the classifier
					tempCls.buildClassifier(tmpData);
					this.classifiers.put(Collections.unmodifiableSet(keySet), tempCls);
				}
			}			
			
		} catch (InstantiationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	public List<Integer> classifyInstance(Instance instance) {

		if (instance == null) {
			throw new IllegalArgumentException("Invalid instance.");
		}
		
		return new ArrayList<Integer>();
	}

}

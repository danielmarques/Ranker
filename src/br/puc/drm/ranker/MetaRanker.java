package br.puc.drm.ranker;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.KStar;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

/**
 * Metaclassifier that builds a model capable of generate ranked lists. This class may use any classifier as the base classifier.
 * 
 * @author Daniel da Rosa Marques
 *
 */ 
public class MetaRanker {

	private Map<Set<Integer>, Classifier> classifiers;
	private Integer dataNumClassValues;
	private Integer optionRankSize;
	private int internalRankSize;
	private String classifierOptions;

	public String getClassifierOptions() {
		
		return classifierOptions;
		
	}

	//Auxiliary method that generates all subsets (powerset) of a set
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
	
	/**
	 * Generates the classifier
	 * 
	 * @param classifier Instance of the base classifier
	 * @param data Dataset for training
	 */
	public void buildClassifier(Classifier classifier, Instances data, String classifierOptions) {
		
		if (classifier == null) {
			
			throw new IllegalArgumentException("Invalid classifier.");
		}
		
		if (data == null) {
			
			throw new IllegalArgumentException("Invalid input data.");
		}
		
		String checkedClassifierOptions = "";
		if (classifierOptions != null) {
			
			checkedClassifierOptions = classifierOptions;
			
		}
		
		// Variable Declarations and Initializations		
		this.classifiers = new HashMap<Set<Integer>, Classifier>();		

		//Verify if the class index is set, use default as last otherwise
		if (data.classIndex()<0) {
			
			data.setClassIndex(data.numAttributes()-1);			
			
		}
		
		//Stores the rank size for further use
		this.dataNumClassValues = data.classAttribute().numValues();

		int rankSize;
		if (this.optionRankSize == null) {
			
			rankSize = this.dataNumClassValues;
						
			
		} else if (this.optionRankSize <= this.dataNumClassValues) {
			
			rankSize = this.optionRankSize;
			
		} else {
			
			throw new IllegalArgumentException("The dataset has to few class atributte values (" + this.dataNumClassValues + "). The rank size is set to " + this.optionRankSize);
		}
		
		this.internalRankSize = rankSize;
		
		try {
			
			Set<Integer> key = new HashSet<Integer>();
			key.add(0);
			
			//Uses reflection to get cls class and generate a new instance
			Classifier tempCls = classifier.getClass().newInstance();
			this.classifierOptions = setClassifierOptions(tempCls, checkedClassifierOptions);

			//Builds and stores the classifier
			tempCls.buildClassifier(data);
			this.classifiers.put(Collections.unmodifiableSet(key), tempCls);
			
			//Generate key subsets for this.classifiers map
			Set<Integer> inputSet = new HashSet<Integer>();
			for (int i = 1; i < data.classAttribute().numValues()+1; i++) {
				inputSet.add(i);
			}
			List<Set<Integer>> tempClsKeys = generateIntSubSets(inputSet);
			List<Set<Integer>> clsKeys = new ArrayList<Set<Integer>>();
			
			//Format the keys to a list of proper rank size
			for (Set<Integer> keySet : tempClsKeys) {
				
				if (keySet.size() < rankSize) {
					clsKeys.add(keySet);
				}
			}

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
					tempCls = classifier.getClass().newInstance();
					setClassifierOptions(tempCls, checkedClassifierOptions);
					
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

	/**
	 * Classifies an instance
	 * 
	 * @param instance The instance to be classified
	 * @return A ranked list of class indexes for the input instance
	 */
	public List<Integer> classifyInstance(Instance instance) {

		if (instance == null) {
			
			throw new IllegalArgumentException("Invalid instance.");
		}
		
		if (this.classifiers == null) {
			
			throw new IllegalStateException("Invalid state: The classifier should be trained first.");
		}
		
		int rankSize = this.internalRankSize;
		
		//Ranked list to be returned
		List<Integer> retList = new ArrayList<Integer>();
		
		Set<Integer> key = new HashSet<Integer>();
		key.add(0);
		
		try {
			
			//Determines and stores the first class on the rank and updates the key
			double instanceClass = this.classifiers.get(key).classifyInstance(instance);

			key.remove(0);
			key.add((int) instanceClass + 1);
			retList.add((int) instanceClass + 1);
			
			Integer rankGap = 0;
			if (rankSize == this.dataNumClassValues) {
				
				rankGap = 1;
				
			}
			
			//Determines the following elements of the list
			for (int i = 1; i < rankSize-rankGap; i++) {
				
				instanceClass = this.classifiers.get(key).classifyInstance(instance);
				//System.out.println("instanceClass " + i + 1 + ": " + instanceClass + " -> " + (instanceClass + 1.0));
				//System.out.println(this.classifiers.get(key));
				key.add((int) instanceClass + 1);
				retList.add((int) instanceClass + 1);
			}

			//Appends the last element to the list if rankSize = dataNumClassValues -> rankGap = 1
			if (rankGap == 1) {
				for (int i = 1; i < rankSize+1; i++) {					
					if (!retList.contains(i)) {
						retList.add(i);
						break;
					}
				}	
			}
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return retList;
	}

	/**
	 * Sets the rank size to be used on classifier building. The rank size can be greater than the number of classes of the dataset used for training.
	 * Also, setRankSize only affects the model after a call to buildClassifier. 
	 * 
	 * @param rankSize The size of the rank the model should output.
	 */
	public void setRankSize(Integer rankSize) {
		this.optionRankSize = rankSize;
	}
	
	private String setClassifierOptions(Classifier cls, String classifierOptions) {
		
		if (cls == null || classifierOptions == null) {
			
			throw new IllegalArgumentException("The arguments can't be null.");
		}
		
		String[] options;
		String retOptions = null;
		
		try {
			
			options = weka.core.Utils.splitOptions(classifierOptions);
			
			if (cls instanceof J48) {

				((J48)cls).setOptions(options);
				retOptions = Arrays.toString(((J48)cls).getOptions());
				
			} else if (cls instanceof NaiveBayes) {
				
				((NaiveBayes)cls).setOptions(options);
				retOptions = Arrays.toString(((NaiveBayes)cls).getOptions());
				
			} else if (cls instanceof SMO) {
				
				((SMO)cls).setOptions(options);
				retOptions = Arrays.toString(((SMO)cls).getOptions());
				
			} else if (cls instanceof MultilayerPerceptron) {
				
				((MultilayerPerceptron)cls).setOptions(options);
				retOptions = Arrays.toString(((MultilayerPerceptron)cls).getOptions());
				
			} else if (cls instanceof KStar) {
				
				((KStar)cls).setOptions(options);
				retOptions = Arrays.toString(((KStar)cls).getOptions());
				
			} else if (cls instanceof RandomForest) {
				
				((RandomForest)cls).setOptions(options);
				retOptions = Arrays.toString(((RandomForest)cls).getOptions());
				
			}			
			
			return retOptions;
			
		} catch (Exception e) {

			throw new RuntimeException("Unable to set the options");
			
		}
	}
}

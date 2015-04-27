package br.puc.drm.ranker.experiments;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import br.puc.drm.ranker.MetaRanker;
import br.puc.drm.ranker.RankEvaluation;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.KStar;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class RankExp {
	
	public static void main(String[] args) {
		
		//Messages		
		final String USAGE = "Usage:java RankExp <Experiments File Path>";
		final String MSG_TOO_FEW_ARGUMENTS  = "Too few arguments";
		final String MSG_TOO_MANY_ARGUMENTS = "Too many arguments";
		final String MSG_INVALID_ARG = "Invalid argument";
		final String MSG_INVALID_INPUT_FILE = "Invalid input file";
		final String MSG_INVALID_JSON = "Invalid Json";
		final String MSG_SAVE_FILE_ERROR = "Problema while saving file.";
		
		System.out.println(USAGE);
		
		//Command line parsing & testing
	    if (args.length < 1) {
	    	
	         System.out.println(MSG_TOO_FEW_ARGUMENTS);
	         System.exit(0);
	         
	    } else if (args.length > 1) {
	    	
	        System.out.println(MSG_TOO_MANY_ARGUMENTS);
	        System.exit(0);
	        
	    } else if (args[0] == null) {
	    	
	    	System.out.println(MSG_INVALID_ARG);
	    	System.exit(0);
	    }	    
        
		//Auxiliary variables
		List<File> files;
		String classifierOptions;
		Integer numberOfFolds;
		String outputResult = "";		
		String inputFilePath = args[0];		
		JSONObject jsonObj;
		
		try {
			
			jsonObj = new JSONObject(readInputFile(inputFilePath));
			
			if (jsonObj.has("experiments")) {
				
				JSONArray experiments = jsonObj.getJSONArray("experiments");
				
				for (int i = 0; i < experiments.length(); i++) {
					
					JSONObject experiment = new JSONObject(experiments.get(i).toString());
					
					System.out.println();
					System.out.println(" # Experiment " + (i + 1));
					System.out.println();
					System.out.println(" - Json: " + experiment.toString());

					if (experiment.has("directory") && experiment.has("classifier") && experiment.has("validation") ) {
						
						if (experiment.has("classifieroptions")) {
							
							classifierOptions = experiment.getString("classifieroptions");
							
						} else {
							
							classifierOptions = null;
						}					
						
						
						if (experiment.has("numberoffolds")) {
							
							numberOfFolds = experiment.getInt("numberoffolds");
							
						} else {
							
							numberOfFolds = 10;
						}
						
						files = listFilesForFolder(new File(experiment.getString("directory")));
						
						//Types of experiments
						
						//Metaranker cross validation
						if (experiment.has("metaranker") && experiment.getBoolean("metaranker") == true 
								&& experiment.getString("validation").equals("C")) {
							
							System.out.println(" - Metaranker Cross validation");
							
							for (File file : files) {
								
								Instances data = loadTestFile(file);
								System.out.println(" - Dataset: " + file.getName());
								
								outputResult += experimentMetaCrossValidated(getClassifier(experiment.getString("classifier"), classifierOptions), data, numberOfFolds);
							}						
						}
						
						//Classifier cross validation
						if (experiment.has("metaranker") && experiment.getBoolean("metaranker") == false 
								&& experiment.getString("validation").equals("C")) {
							
							System.out.println(" - Classifier Cross Validation");
							
							for (File file : files) {
								
								Instances data = loadTestFile(file);			
								
								System.out.println(" - Dataset: " + file.getName());
								
								outputResult += experimentClassCrossValidated(getClassifier(experiment.getString("classifier"), classifierOptions), data, numberOfFolds);
							}						
						}
						
					} else {
						
						System.out.println(" - The experiment is missing parameters.");
					}
				}
				
				System.out.println();
				
				//Write to output file					
				FileWriter writer = new FileWriter("Experiment_Results_" + System.nanoTime() + ".csv");
				writer.append("Score, Maximum_Score, Percentage, Elapsed_Time_Average\n");
				writer.append(outputResult);
				writer.flush();
				writer.close();
				
			} else { 
				
				System.out.println(" - No experiments to run."); 
			
			}
			
		} catch (FileNotFoundException e1) {

	    	System.out.println(MSG_INVALID_INPUT_FILE);
	    	System.exit(0);
	    	
		} catch (JSONException e2) {

	    	System.out.println(MSG_INVALID_JSON);
	    	System.exit(0);

		} catch (IOException e3) {

	    	System.out.println(MSG_SAVE_FILE_ERROR);
	    	System.exit(0);
			
		}
	}
	
	private static String experimentMetaCrossValidated(Classifier classifier, Instances data, Integer numberOfFolds) {		
		
		MetaRanker mr = new MetaRanker();
		
		RankEvaluation eval = new RankEvaluation();
		
		long startTime = System.nanoTime();
		eval.crossValidateRankModel(mr, classifier, data, numberOfFolds, new Random());
		long elapsedTime = System.nanoTime() - startTime;
		System.out.println(" - " + eval.toSummaryString() + " in " + (elapsedTime/numberOfFolds) + " nanoseconds (average)");
		
		return eval.toCSVLine() + ", " + (elapsedTime/numberOfFolds + "\n");
		
		
	}
	
	private static String experimentClassCrossValidated(Classifier classifier,
			Instances data, Integer numberOfFolds) {
		
		RankEvaluation eval = new RankEvaluation();
		
		long startTime = System.nanoTime();
		eval.crossValidateRankModel(classifier, data, numberOfFolds, new Random());
		long elapsedTime = System.nanoTime() - startTime;
		System.out.println(" - " + eval.toSummaryString() + " in " + (elapsedTime/numberOfFolds) + " nanoseconds (average)");
		
		return eval.toCSVLine() + ", " + (elapsedTime/numberOfFolds + "\n");
		
	}

	private static String readInputFile(String filePath) throws FileNotFoundException {
		
		String content;
	
		Scanner scanner = new Scanner(new File(filePath));
		content = scanner.useDelimiter("\\Z").next();
		scanner.close();
		
		return content;

	}

	//Instantiates a classifier
	private static Classifier getClassifier(String classifierName, String classifierOptions) {
		
		Classifier classifier = null;
		
		try {
		
			switch (classifierName) {
			case "J48":
				classifier = new J48();
				if (classifierOptions != null) ((J48) classifier).setOptions(weka.core.Utils.splitOptions(classifierOptions));
				break;

			case "NaiveBayes":
				classifier = new NaiveBayes();
				if (classifierOptions != null) ((NaiveBayes) classifier).setOptions(weka.core.Utils.splitOptions(classifierOptions));
				break;

			case "SMO":
				classifier = new SMO();
				if (classifierOptions != null) ((SMO) classifier).setOptions(weka.core.Utils.splitOptions(classifierOptions));
				break;
				
			case "MultilayerPerceptron":
				classifier = new MultilayerPerceptron();
				if (classifierOptions != null) ((MultilayerPerceptron) classifier).setOptions(weka.core.Utils.splitOptions(classifierOptions));
				break;
			
			case "KStar":
				classifier = new KStar();
				if (classifierOptions != null) ((KStar) classifier).setOptions(weka.core.Utils.splitOptions(classifierOptions));
				break;
			
			case "RandomForest":
				classifier = new RandomForest();
				if (classifierOptions != null) ((RandomForest) classifier).setOptions(weka.core.Utils.splitOptions(classifierOptions));
				break;
				
			default:
				break;
			}		
						
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return classifier;
		
	}
	
	private static Instances loadTestFile(File file) {
		
		Instances data = null;
		
		ArffLoader loader = new ArffLoader();
		
	    try {
	    	
			loader.setFile(file);
			data = loader.getDataSet();
			data.setClassIndex(data.numAttributes()-1);
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    
	    return data;
	}
	
	static private List<File> listFilesForFolder(final File folder) {
		
		ArrayList<File> files = new ArrayList<File>();
		
	    for (final File fileEntry : folder.listFiles()) {
	    	
	        if (fileEntry.isDirectory()) {
	        	
	            files.addAll(listFilesForFolder(fileEntry));
	            
	        } else {
	        	
	        	files.add(new File(folder.getAbsolutePath()+ "/" + fileEntry.getName()));
	        	
	        }
	    }
	    
		return files;
	}
}

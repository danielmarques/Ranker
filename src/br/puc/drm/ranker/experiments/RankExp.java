package br.puc.drm.ranker.experiments;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import br.puc.drm.ranker.MetaRanker;
import br.puc.drm.ranker.RankEvaluation;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import org.json.JSONArray;
import org.json.JSONObject;

public class RankExp {
	
	public static void main(String[] args) {
		
		//Command line parsing & testing		
		final String USAGE = "Usage:java RankExp <Experiments File Path>";
		final String MSG_TOO_FEW_ARGUMENTS  = "too few arguments";
		final String MSG_TOO_MANY_ARGUMENTS = "too many arguments";
		
	    if (args.length < 1) {
	    	
	         System.out.println(MSG_TOO_FEW_ARGUMENTS);
	         
	    } else if (args.length > 1)
	    	
	        System.out.println(MSG_TOO_MANY_ARGUMENTS);
	    
	    System.out.println(USAGE);
        
		//Auxiliary variables
		List<File> files;
		Classifier classifier;
		String classifierOptions;
		Integer numberOfFolds;
		String outputResult = "";		
		String inputFilePath = args[0];		
		JSONObject jsonObj = new JSONObject(readInputFile(inputFilePath));
		
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
							
							Instances data = loadTestFile("iris.arff");
							System.out.println(" - Dataset: " + file.getName());
							
							outputResult += experimentMetaCrossValidated(getClassifier(experiment.getString("classifier"), classifierOptions), data, numberOfFolds);
						}						
					}
					
					//Classifier cross validation
					if (experiment.has("metaranker") && experiment.getBoolean("metaranker") == false 
							&& experiment.getString("validation").equals("C")) {
						
						System.out.println(" - Classifier Cross Validation");
						
						for (File file : files) {
							
							Instances data = loadTestFile("iris.arff");			
							
							System.out.println(" - Dataset: " + file.getName());
							
							outputResult += experimentClassCrossValidated(getClassifier(experiment.getString("classifier"), classifierOptions), data, numberOfFolds);
						}						
					}
					
				} else {
					
					System.out.println(" - The experiment is missing parameters.");
				}
			}
			
			//Write to output file
			try {
				
				FileWriter writer = new FileWriter("Experiment_Results_" + System.currentTimeMillis() + ".csv");
				writer.append("Score, Maximum_Score, Percentage, Elapsed_Time_Average\n");
				writer.append(outputResult);
				writer.flush();
				writer.close();
				
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
		} else { 
			
			System.out.println(" - No experiments to run."); 
		
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

	private static String readInputFile(String filePath) {
		
		String content;
		
		try {
			
			content = new Scanner(new File(filePath)).useDelimiter("\\Z").next();

			return content;
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return null;
	}
	//Instantiates a classifier
	private static Classifier getClassifier(String classifierName, String classifierOptions) {
		
		J48 classifier = new J48();
		
		try {
			
			if (classifierOptions != null) {
				
				classifier.setOptions(weka.core.Utils.splitOptions(classifierOptions));
			}
						
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return classifier;
		
	}
	
	private static Instances loadTestFile(String fileName) {
		
		Instances data = null;
		
		ArffLoader loader = new ArffLoader();
		
	    try {
	    	
			loader.setFile(new File(fileName));
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
	        	
	        	files.add(new File(folder.getAbsolutePath()+ "\\" + fileEntry.getName()));
	        	
	        }
	    }
	    
		return files;
	}
}

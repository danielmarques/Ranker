package br.puc.drm.ranker.experiments;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
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
import weka.core.Instance;
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
						
						Integer rankSize = null;
						if (experiment.has("ranksize")) {
							
							rankSize = experiment.getInt("ranksize");
							
						}
						
						files = listFilesForFolder(new File(experiment.getString("directory")));
						
						//Types of experiments
						
						String classHistogram = "";
						
						//Metaranker cross validation
						if (experiment.has("metaranker") && experiment.getBoolean("metaranker") == true 
								&& experiment.getString("validation").equals("C")) {
							
							System.out.println(" - Metaranker Cross validation");
							
							for (File file : files) {
								
								Instances data = loadTestFile(file);
								System.out.println(" - Dataset: " + file.getName());
								
								Integer finalRankSize = rankSize;
								if (finalRankSize == null) {
									finalRankSize = data.classAttribute().numValues();
								}
								
								classHistogram = attributeHistogram(data);
								
								String mode = "Train Before";
								if (experiment.has("dynamic") && experiment.getBoolean("dynamic")) {
									
									outputResult = experimentMetaCrossValidatedDynamic(getClassifier(experiment.getString("classifier"), ""), classifierOptions, data, numberOfFolds, rankSize);
									mode = "Dynamic";
									
								} else {
									
									outputResult = experimentMetaCrossValidated(getClassifier(experiment.getString("classifier"), ""), classifierOptions, data, numberOfFolds, rankSize);
									
								}
								
								outputResult += ", " + data.classAttribute().numValues() +
												", " + finalRankSize +
												", " + classHistogram +
												", " + file.getName() + 
												", " + experiment.getBoolean("metaranker") +
												", " + mode +
												", " + experiment.getString("classifier") +
												", " + classifierOptions +
												", " + "Cross Validation" +
												", " + numberOfFolds + "\n";
								
								
								//Write to output file for this experiment			
								FileWriter writer = new FileWriter("Experiment_Results_" + System.nanoTime() + ".csv");
								writer.append(
									"1-Accuracy (Avg), Percentage, 2-Accuracy (Avg), Percentage, 3-Accuracy (Avg), Percentage, 4-Accuracy (Avg), Percentage, 5-Accuracy (Avg), Percentage, "
									+ "Max_Accuracy, Train_Elapsed_Time_Avg (ms), Test_Elapsed_Time_Avg (ms), "
									+ "Number_of_Class_Values, Rank_Size, Class_Histogram, "
									+ "Dataset, Metaranker, Mode, Classifier, Classifier_Options, "
									+ "Validation, Validation_Options\n");
								writer.append(outputResult);
								writer.flush();
								writer.close();
							}						
						}
						
						//Classifier cross validation
						if (experiment.has("metaranker") && experiment.getBoolean("metaranker") == false 
								&& experiment.getString("validation").equals("C")) {
							
							System.out.println(" - Classifier Cross Validation");
							
							for (File file : files) {
								
								Instances data = loadTestFile(file);			
								
								System.out.println(" - Dataset: " + file.getName());
								
								Integer finalRankSize = rankSize;
								if (finalRankSize == null) {
									finalRankSize = data.classAttribute().numValues();
								}
								
								classHistogram = attributeHistogram(data);
								
								outputResult = experimentClassCrossValidated(getClassifier(experiment.getString("classifier"), classifierOptions), data, numberOfFolds, rankSize);
								outputResult += ", " + data.classAttribute().numValues() +
												", " + finalRankSize +
												", " + classHistogram +
												", " + file.getName() + 
												", " + experiment.getBoolean("metaranker") +
												", " + "-" +
												", " + experiment.getString("classifier") +
												", " + classifierOptions +
												", " + "Cross Validation" +
												", " + numberOfFolds + "\n";
								
								
								//Write to output file for this experiment			
								FileWriter writer = new FileWriter("Experiment_Results_" + System.nanoTime() + ".csv");
								writer.append(
									"1-Accuracy (Avg), Percentage, 2-Accuracy (Avg), Percentage, 3-Accuracy (Avg), Percentage, 4-Accuracy (Avg), Percentage, 5-Accuracy (Avg), Percentage, "
									+ "Max_Accuracy, Train_Elapsed_Time_Avg (ms), Test_Elapsed_Time_Avg (ms), "
									+ "Number_of_Class_Values, Rank_Size, Class_Histogram, "
									+ "Dataset, Metaranker, Mode, Classifier, Classifier_Options, "
									+ "Validation, Validation_Options\n");
								writer.append(outputResult);
								writer.flush();
								writer.close();
								
							}						
						}
						
					} else {
						
						System.out.println(" - The experiment is missing parameters.");
					}
					
				}
				
				System.out.println();
				
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
	
	private static String experimentMetaCrossValidated(Classifier classifier, String classifierOptions, Instances data, Integer numberOfFolds, Integer rankSize) {		
		
		MetaRanker mr = new MetaRanker();
		mr.setRankSize(rankSize);
		
		RankEvaluation eval = new RankEvaluation();
		
		long startTime = System.nanoTime();
		eval.crossValidateRankModel(mr, classifier, classifierOptions, data, numberOfFolds);
		long elapsedTime = System.nanoTime() - startTime;
		System.out.println(" - " + eval.toSummaryString());
		
		return eval.toCSVLine();
		
		
	}
	
	private static String experimentMetaCrossValidatedDynamic(Classifier classifier, String classifierOptions, Instances data, Integer numberOfFolds, Integer rankSize) {		
		
		MetaRanker mr = new MetaRanker();
		mr.setRankSize(rankSize);
		
		RankEvaluation eval = new RankEvaluation();
		//eval.setMaxExperimentTime((long) 600000000000.0); //Max time 10 minutes
		
		long startTime = System.nanoTime();
		eval.crossValidateRankModelDynamic(mr, classifier, classifierOptions, data, numberOfFolds);
		long elapsedTime = System.nanoTime() - startTime;
		System.out.println(" - " + eval.toSummaryString());
		
		return eval.toCSVLine();		
		
	}
	
	private static String experimentClassCrossValidated(Classifier classifier, Instances data, Integer numberOfFolds, Integer rankSize) {
		
		RankEvaluation eval = new RankEvaluation();
		
		long startTime = System.nanoTime();
		eval.crossValidateRankModel(classifier, data, numberOfFolds, rankSize);
		long elapsedTime = System.nanoTime() - startTime;
		System.out.println(" - " + eval.toSummaryString());
		
		return eval.toCSVLine();
		
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
	
	static private String attributeHistogram(Instances data) {
		
		Integer[] hist = new Integer[data.classAttribute().numValues()];

		for (int i = 0; i < hist.length; i++) {
			hist[i] = 0;
		}
		
		for (Instance instance : data) {
			
			hist[(int) instance.classValue()]++;			
		}		
		
		Arrays.sort(hist, Collections.reverseOrder());
		String stringHist = Arrays.toString(hist);
		
		return stringHist.replace(", ", " ").replace("[", "").replace("]", "");
	}
}

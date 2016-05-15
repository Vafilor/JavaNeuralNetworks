package com.vafilor.neural;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;


/**
 * Created by Andrey Melnikov on 5/14/2016.
 *
 */
public class Main {

    //TODO Next up -  Add logic to output Net so we can reload it later.
    public static void main(String[] args) throws IOException
    {
        trainNeuralNetwork();
    }

    public static void trainNeuralNetwork()
    {
        Path trainingImagesPath = Paths.get("");
        Path trainingLabelsPath = Paths.get("");
        Path testLabelsPath = Paths.get("");
        Path testImagesPath = Paths.get("");

        Timer programTimer = new Timer();
        programTimer.mark();

        List<Matrix> labels = LabelReader.convertData(trainingLabelsPath);

        programTimer.mark();

        System.out.format("Loaded Labels - Took: %f seconds.%n", programTimer.getLastMarksElapsedTimeInSeconds());

        programTimer.mark();

        List<Matrix> images = ImageReader.convertData(trainingImagesPath);

        programTimer.mark();

        System.out.format("Loaded Images - Took: %f seconds.%n", programTimer.getLastMarksElapsedTimeInSeconds());

        programTimer.mark();

        List<Pair<Matrix,Matrix>> learningData = combineData(images, labels);

        programTimer.mark();

        System.out.format("Converted Training Data - Took: %f seconds.%n", programTimer.getLastMarksElapsedTimeInSeconds());

        programTimer.mark();
        List<Matrix> testLabelData = LabelReader.convertData(testLabelsPath);

        programTimer.mark();

        System.out.format("Loaded Test Labels - Took: %f seconds.%n", programTimer.getLastMarksElapsedTimeInSeconds());

        programTimer.mark();

        List<Matrix> testImageData = ImageReader.convertData(testImagesPath);

        programTimer.mark();

        System.out.format("Loaded Test Images - Took: %f seconds.%n", programTimer.getLastMarksElapsedTimeInSeconds());

        programTimer.mark();

        List<Pair<Matrix, Matrix>> testData = combineData(testImageData, testLabelData);

        programTimer.mark();

        System.out.format("Converted Test Data - Took: %f seconds.%n", programTimer.getLastMarksElapsedTimeInSeconds());

        NeuralNetwork network = new NeuralNetwork(784, 30, 10);

        programTimer.mark();
        network.SGD(learningData, 30, 10, 1.0, testData );

        programTimer.mark();

        System.out.format("Network Training - Took: %f seconds.%n", programTimer.getLastMarksElapsedTimeInSeconds());
        System.out.format("Total Program - Took: %f seconds.%n", programTimer.getTotalElapsedTimeInSeconds());
    }

    public static List<Pair<Matrix,Matrix>> combineData(List<Matrix> first, List<Matrix> second)
    {
        List<Pair<Matrix, Matrix>> result = new ArrayList<>(first.size());

        for(int i = 0; i < first.size(); i++)
        {
            result.add(new Pair<>(first.get(i), second.get(i)));
        }

        return result;
    }

    /**
     * Helper function that prints out a 28x28 image stored in a Matrix of size (28x28 by 1) to the console.
     * Used to test to make sure images were loaded correctly.
     * @param image
     */
    public static void printImage(Matrix image)
    {
        double entry = 0.0;

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                entry = image.getEntry(i * 28 + j, 0);
                if( entry < 75)
                {
                    System.out.print(" ");
                } else if( entry < 150)
                {
                    System.out.print("x");
                } else
                {
                    System.out.print( (char)223);
                }
            }
            System.out.println();
        }
    }
}

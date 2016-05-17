package com.vafilor.neural;

import java.util.Collections;
import java.util.Random;
import java.util.List;
import java.util.ArrayList;

/**
 * Created by Andrey Melnikov on 5/9/16.
 *
 * Based on Michael Nielsen's code/book available at:
 * http://neuralnetworksanddeeplearning.com/chap1.html
 */
public class NeuralNetwork
{
    private int[] sizes;
    private List<Matrix> biases;
    private List<Matrix> weights;

    public NeuralNetwork(int... sizes)
    {
        this.sizes = new int[sizes.length];
        System.arraycopy(sizes, 0, this.sizes, 0, sizes.length);

        this.biases = new ArrayList<>();

        for (int i = 1; i < (this.sizes.length); i++) {
            this.biases.add(NeuralNetwork.gaussianDistributionMatrix(this.sizes[i], 1));
        }

        this.weights = new ArrayList<>();

        for (int i = 0; i < this.sizes.length - 1; i++)
        {
            this.weights.add(NeuralNetwork.gaussianDistributionMatrix(this.sizes[i + 1], this.sizes[i]));
        }
    }

    public Matrix feedforward(Matrix input)
    {
        Matrix result = input;

        for(int i = 0; i < this.biases.size(); i++)
        {
            result = this.weights.get(i).multiply(result).add(this.biases.get(i)).applyFunction(NeuralNetwork::sigmoid);
        }

        return result;
    }

    public void SGD(List<Pair<Matrix, Matrix>> trainingData, int epochs, int miniBatchSize, double eta, List<Pair<Matrix,Matrix>> testData)
    {
        int n = trainingData.size();



        for(int i = 1; i <= epochs; i++)
        {
            List<List<Pair<Matrix,Matrix>>> miniBatches = new ArrayList<>();
            Collections.shuffle(trainingData);

            for(int j = 0; j < n; j += miniBatchSize)
            {
                List<Pair<Matrix,Matrix>> mini = new ArrayList<>();

                for(int k = j; k < j + miniBatchSize; k++ )
                {
                    mini.add(trainingData.get(k));
                }

                miniBatches.add(mini);
            }

            for(List<Pair<Matrix,Matrix>> miniBatch : miniBatches)
            {
                this.updateMiniBatch(miniBatch, eta);
            }

            if(null != testData)
            {
                System.out.format("Epoch %d / %d: %d / %d correct.%n", i, epochs, this.evaluate(testData), testData.size());
            }

        }
    }

    public void updateMiniBatch(List<Pair<Matrix,Matrix>> miniBatch, double eta)
    {
        ArrayList<Matrix> nabla_b = this.createBlankCopy(this.biases);
        ArrayList<Matrix> nabla_w = this.createBlankCopy(this.weights);

        Pair<Matrix, Matrix> pair = null;

        ArrayList<Matrix> delta_nabla_b = null;
        ArrayList<Matrix> delta_nabla_w = null;

        for(int i = 0; i < miniBatch.size(); i++)
        {
            pair = miniBatch.get(i);

            ArrayList<ArrayList<Matrix>> result = this.backprop(pair.getFirstElement(), pair.getSecondElement());

            delta_nabla_b = result.get(0);
            delta_nabla_w = result.get(1);

			this.addInto(nabla_b, delta_nabla_b);
			this.addInto(nabla_w, delta_nabla_w);
        }

        Matrix w = null;
        Matrix nw = null;

        for(int i = 0; i < this.weights.size(); i++)
        {
            nw = nabla_w.get(i);
            this.weights.get(i).subtractInto( nw.scale( eta / miniBatch.size() ));
        }

        Matrix b = null;
        Matrix nb = null;

        for (int i = 0; i < this.biases.size(); i++)
        {
            nb = nabla_b.get(i);
            this.biases.get(i).subtractInto(nb.scale(eta / miniBatch.size()));
        }
    }

    /**
     * Returns a 2 element array where each element is a list of matrices. First list is gradients for biases. Second list is gradient for weights.
     * @param input
     * @param output
     * @return
     */
    //TODO change to use pair
    public ArrayList<ArrayList<Matrix>> backprop(Matrix input, Matrix output)
    {
        ArrayList<Matrix> nabla_b = this.createBlankCopy(this.biases);
		ArrayList<Matrix> nabla_w = this.createBlankCopy(this.weights);
		
		Pair<ArrayList<Matrix>, ArrayList<Matrix>> zsAndActivations = this.calculateActivations(input);

		ArrayList<Matrix> zs = zsAndActivations.getFirstElement();
		ArrayList<Matrix> activations = zsAndActivations.getSecondElement();
        

        //Backward pass
        Matrix delta = this.costDerivative(activations.get(activations.size() - 1), output).
                            multiplyEntries(
                                    zs.get(zs.size()-1).
                                    applyFunction(NeuralNetwork::sigmoidPrime)
                            );


        nabla_b.set(nabla_b.size()-1, delta);
        nabla_w.set(nabla_w.size()-1,  delta.multiply(activations.get(activations.size() - 2).transpose()) );

        Matrix z = null;
        Matrix sp = null;

        for(int i = 2; i < this.sizes.length; i++)
        {
            z = zs.get(zs.size() - i);
            sp = z.applyFunction(NeuralNetwork::sigmoidPrime);

            delta = this.weights.get(this.weights.size() - i + 1).transpose().multiply(delta).multiplyEntries(sp);
            nabla_b.set(nabla_b.size() - i, delta);
            nabla_w.set(nabla_w.size() - i, delta.multiply(activations.get(activations.size() - i - 1).transpose()) );
        }

        ArrayList<ArrayList<Matrix>> results = new ArrayList<ArrayList<Matrix>>();
        results.add(nabla_b);
        results.add(nabla_w);
        //TODO why List<List<Matrix>> not work here?

        return results;
    }

	//Returns the zs and activations for the input matrix using the current biases and weights
	//firstElement is zs, second element is activations
	private Pair<ArrayList<Matrix>, ArrayList<Matrix>> calculateActivations(Matrix input)
	{
		Matrix weight = null;
		Matrix bias = null;
		
		ArrayList<Matrix> zs = new ArrayList<>();
		ArrayList<Matrix> activations = new ArrayList<>();
	
		Matrix activation = input;
		
		activations.add(activation);
	
	    for(int i = 0; i < this.biases.size(); i++)
        {
            weight = this.weights.get(i);
            bias = this.biases.get(i);

            Matrix z = weight.multiply(activation).add(bias);
            zs.add(z);
            activation = z.applyFunction(NeuralNetwork::sigmoid);
            activations.add(activation);
        }
        
        return new Pair(zs, activations);
	}

    public int evaluate(List<Pair<Matrix, Matrix>> testData)
    {
        List<Integer> testResults = new ArrayList<>();

        Matrix result = null;

        for(int i = 0; i < testData.size(); i++)
        {
            result = this.feedforward(testData.get(i).getFirstElement());

            testResults.add(this.getLargestRow(result));
        }

        int totalCorrect = 0;

		//TODO make this a separate method
        for(int i = 0; i < testData.size(); i++)
        {
            if( testResults.get(i) == this.getNonZeroRow(testData.get(i).getSecondElement()) )
            {
                totalCorrect++;
            }
        }

        return totalCorrect;
    }


    /**
     * @param matrix input matrix
     * @return the highest row in the matrix that has a non-zero entry in column 0.
     */
    private int getNonZeroRow(Matrix matrix)
    {
        return this.getNonZeroRowWithThreshold(matrix, 0);
    }

    private int getNonZeroRowWithThreshold(Matrix matrix, double threshold)
    {
        for(int i = matrix.getRows() - 1; i >= 0; i--)
        {
            if( matrix.getEntry(i, 0) > threshold)
            {
                return i;
            }
        }

        return -1;
    }

	private int getLargestRow(Matrix matrix)
    {
        int largest = 0;
        double largestValue = matrix.getEntry(largest, 0);

        for(int i = 0; i < matrix.getRows(); i++)
        {
            if( matrix.getEntry(i, 0) > largestValue)
            {
                largest = i;
                largestValue = matrix.getEntry(largest, 0);
            }
        }

        return largest;
    }

    private Matrix costDerivative(Matrix outputActiviations, Matrix output)
    {
        return outputActiviations.subtract(output);
    }

    @Override
    public String toString()
    {
        StringBuilder builder = new StringBuilder();

        builder.append("Biases:\n");

        for(Matrix bias : this.biases)
        {
            builder.append(bias.toString());
            builder.append("\n");
        }

        builder.append("Weights:\n");

        for(Matrix weights : this.weights)
        {
            builder.append(weights.toString());
            builder.append("\n");
        }

        return builder.toString();
    }

	private ArrayList<Matrix> createBlankCopy(List<Matrix> source)
	{
		ArrayList<Matrix> copy = new ArrayList<>(source.size());
		
		for(int i = 0; i < source.size(); i++)
		{
			copy.add(new Matrix(source.get(i).getRows(), source.get(i).getColumns()));
		}
		
		return copy;
	}

	//Modifies the entries in destination by adding the entries in delta. 
	//At the end, ith entry in destination = ith entry in old + ith entry in delta
	private void addInto(List<Matrix> destination, List<Matrix> delta)
	{
		for(int i = 0; i < destination.size(); i++)
		{
			destination.get(i).addInto(delta.get(i));
		}
	}

    private static double sigmoid(double input)
    {
        return 1.0 /( 1.0 + Math.exp(-input) );
    }

    private static double sigmoidPrime(double input)
    {
        return sigmoid(input) * (1-sigmoid(input));
    }

    public static Matrix gaussianDistributionMatrix(int rows, int columns)
    {
        Random random = new Random();

        Matrix matrix = new Matrix(rows, columns);

        for (int i = 0; i < matrix.getRows(); i++) {
            for (int j = 0; j < matrix.getColumns(); j++) {
                matrix.setEntry(i, j, random.nextGaussian() );
            }
        }

        return matrix;
    }
}
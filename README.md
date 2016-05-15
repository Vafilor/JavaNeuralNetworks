# JavaNeuralNetworks
Playing around with Neural Networks.

Background
------------------
I saw Michael Nielson's book on Neural Networks mentioned on Hacker News, and read through the handwritten digits portion. I thought it'd be fun to see if I could recreate
the Neural Network code as much as possible by hand. To that end, I created a Matrix class to deal with the Linear Algebra, and 
a NeuralNetwork class to handle the... well, Neural Network. It is all written in Java, and I hope to make a C++ version in the 
near future so that I can compare the performance.

Program Notes
-------------------
The Neural Network works, but its a work in progress. Next steps are to clean up the source code, write unit tests, and see if I can't
speed up loading the image data and neural network code itself. 

Also, I'm not getting as good results as Nielson's program, so need to take another look at that.


Acknowledgements
-----------------
This is based on Michael Nielsen's book "Neural Networks and Deep Learning" - Using neural nets to recognize handwritten digits.
Available at Nielsen's website: http://neuralnetworksanddeeplearning.com/ 


Prerequisites
-----------------
1. Download and extract the four files for training data available at http://yann.lecun.com/exdb/mnist/

To Compile
----------------
1. Edit the main file and provide the paths (in the trainNeuralNetwork method) for the training data downloaded above.
  1. Path trainingImagesPath = Paths.get("path_to_train-images.idx3-ubyte");
  2. Path trainingLabelsPath = Paths.get("path_to_train-labels.idx1-ubyte");
  3. Path testLabelsPath = Paths.get("path_to_test-t10k-labels.idx1-ubyte");
  4. Path testImagesPath = Paths.get("path_to_test-t10k-images.idx3-ubyte");
  
2. Open up a terminal/command line and navigate to src directory. 
3. Enter the command
  1. Windows: javac com\vafilor\neural\\*.java 
  2. Linuxy: javac com/vafilor/neural/*.java 
	

To Run
-----------------
1. Open up a terminal/command line and navigate to src directory. 
2. Enter the command:  **java com.vafilor.neural.Main**
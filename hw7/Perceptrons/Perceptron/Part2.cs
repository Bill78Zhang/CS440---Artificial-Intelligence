using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Perceptron
{
    class Part2
    {
        static void Main(string[] args)
        {
            var perceptrons = TrainPerceptrons();

            var accuracy = EvaluatePerceptrons(perceptrons);

            Console.WriteLine("Accuracy on Test Data: " + accuracy);
            Console.ReadLine();
        }

        /// <summary>
        /// Trains Perceptrons </summary>
        /// <returns>
        /// Returns Trained Perceptrons </returns>
        private static List<Perceptron> TrainPerceptrons()
        {
            const string imageFilePath = "data/trainingimages";
            var trainingData = ImportImages(imageFilePath);

            const string labelFilePath = "data/traininglabels";
            var trainingLabels = ImportLabels(labelFilePath);

            var trainingLimit = (int)(trainingData.Count * 0.8);
            const double alpha = 0.95;
            const int epoch = 500;

            // Initialize Perceptrons
            var perceptrons = new List<Perceptron>();
            for (var i = 0; i < 10; i++)
            {
                var p = SetupPerceptron(i, alpha);
                perceptrons.Add(p);
            }

            for (var e = 0; e < epoch; e++)
            {
                var indices = RandomizeSequence(trainingLabels.Length);
                // Training
                for (var i = 0; i < trainingLimit; i++)
                {

                    var xLabel = trainingLabels[indices[i]];

                    for (var j = 0; j < perceptrons.Count; j++)
                    {
                        perceptrons[j].Classify(trainingData[indices[i]], xLabel);
                    }
                }

                var correct = 0.0;
                var numClassified = 0.0;

                // Validation
                for (var i = trainingLimit; i < indices.Length; i++)
                {
                    var x = trainingData[indices[i]];
                    var xLabel = trainingLabels[indices[i]];

                    var best = 0.0;
                    var predict = 0;

                    for (var j = 0; j < perceptrons.Count; j++)
                    {
                        var score = perceptrons[j].Classify(x, xLabel);

                        if (score > best)
                        {
                            best = score;
                            predict = j;
                        }
                    }

                    if (predict == xLabel)
                    {
                        correct++;
                    }

                    numClassified++;
                }

                Console.WriteLine("Epoch: " + e + " Accuracy: " + correct / numClassified * 100);
            }

            return perceptrons;
        }

        /// <summary>
        /// Evaluates Perceptrons on the Test Data </summary>
        /// <param name="perceptrons"> List of Perceptrons </param>
        /// <returns>
        /// Accuracy </returns>
        private static double EvaluatePerceptrons(List<Perceptron> perceptrons)
        {
            var correct = 0.0;
            var numClassified = 0.0;

            const string imageFilePath = "data/trainingimages";
            var testData = ImportImages(imageFilePath);

            const string labelFilePath = "data/traininglabels";
            var testLabels = ImportLabels(labelFilePath);

            for (var i = 0; i < testData.Count; i++)
            {
                var x = testData[i];
                var xLabel = testLabels[i];

                var best = 0.0;
                var predict = 0;

                for (var j = 0; j < perceptrons.Count; j++)
                {
                    var score = perceptrons[j].Classify(x, xLabel);

                    if (score > best)
                    {
                        best = score;
                        predict = j;
                    }
                }

                if (predict == xLabel)
                {
                    correct++;
                }

                numClassified++;
            }

            return correct / numClassified * 100;
        }

        /// <summary>
        /// Imports Images as vectors </summary>
        /// <param name="imageFilePath"> File Path to Image File </param>
        /// <returns>
        /// Returns List of double Array of images </returns>
        private static List<double[]> ImportImages(string imageFilePath)
        {
            var data = new List<double[]>();

            var counter = 0;
            string line;
            var item = "";
            var file = new StreamReader(File.OpenRead(imageFilePath));
            while ((line = file.ReadLine()) != null)
            {
                item += line;

                if (++counter % 28 != 0) continue;

                var itemData = new double[item.Length];

                for (var i = 0; i < item.Length; i++)
                {
                    itemData[i] = ConvertCharacter(item[i]);
                }
                data.Add(itemData);
                item = "";
            }

            return data;
        }

        /// <summary>
        /// Classifies Input Characters </summary>
        /// <param name="inputValue"> Input Character </param>
        /// <returns>
        /// Returns Classified Value </returns>
        private static double ConvertCharacter(char inputValue)
        {
            switch (inputValue)
            {
                case ' ':
                    return 0.0;
                case '+':
                    return 0.5;
                case '#':
                    return 1.0;
            }

            return 0.0;
        }

        /// <summary>
        /// Imports Labels </summary>
        /// <param name="labelFilePath"> File Path to Label File </param>
        /// <returns>
        /// Returns Int Array of labels </returns>
        private static int[] ImportLabels(string labelFilePath)
        {
            var labels = "";
            string line;
            var file = new StreamReader(File.OpenRead(labelFilePath));
            while ((line = file.ReadLine()) != null)
            {
                labels += line;
            }

            var dataLabels = labels.Select(label => label - '0').ToArray();

            return dataLabels;
        }

        /// <summary>
        /// Constructor for Perceptron </summary>
        /// <param name="label"> Label of Perceptron </param>
        /// <param name="alpha"> Learning Rate </param>
        /// <returns>
        /// Returns Initialized Perceptron </returns>
        private static Perceptron SetupPerceptron(int label, double alpha)
        {
            var p = new Perceptron
            {
                Label = label,
                Alpha = alpha,
                Weights = new double[784]
            };

            for (var i = 0; i < p.Weights.Length; i++)
            {
                p.Weights[i] = 0.0;
            }

            return p;
        }

        /// <summary>
        /// Fisher-Yates Shuffle, randomize sequence of numbers </summary>
        /// <param name="count"> number of elements </param>
        /// <returns>
        /// Returns int array of randomized numbers </returns>
        private static int[] RandomizeSequence(int count)
        {
            var rng = new Random();
            var numbers = new int[count];

            for (var i = 0; i < count; i++)
            {
                numbers[i] = i;
            }

            for (var i = 0; i < count - 2; i++)
            {
                var j = rng.Next(i, count);
                var temp = numbers[i];
                numbers[i] = numbers[j];
                numbers[j] = temp;
            }

            return numbers;
        }
    }
}
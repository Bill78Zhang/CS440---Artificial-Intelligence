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

            // Parameters
            var trainingLimit = (int)(trainingData.Count * 0.2);
            const double alpha = 1.0;
            const int epoch = 200;

            // Setup Validation Set
            var validationData = trainingData.Take(trainingLimit).ToList();
            var validationLabels = trainingLabels.Take(trainingLimit).ToList();
            trainingData = trainingData.Skip(trainingLimit).ToList();
            trainingLabels = trainingLabels.Skip(trainingLimit).ToList();
            const double tol = 1e-4;
            var lastAccuracy = 0.0;
            var doneTraining = false;

            // Initialize Perceptrons
            var perceptrons = new List<Perceptron>();
            for (var i = 0; i < 10; i++)
            {
                var p = SetupPerceptron(i, alpha);
                perceptrons.Add(p);
            }

            for (var e = 0; e < epoch; e++)
            {
                if (doneTraining) {
                    Console.WriteLine("Done Training");
                    break;
                }

                // Training
                var indices = RandomizeSequence(trainingLabels.Count);
                for (var i = 0; i < trainingLabels.Count; i++)
                {

                    var xLabel = trainingLabels[indices[i]];

                    for (var j = 0; j < perceptrons.Count; j++)
                    {
                        perceptrons[j].Classify(trainingData[indices[i]], xLabel, true);
                    }
                }

                var correct = 0.0;
                var numClassified = 0.0;

                // TODO: Refactor out to individual function
                // Validation
                for (var i = 0; i < validationLabels.Count; i++)
                {
                    var x = validationData[i];
                    var xLabel = validationLabels[i];

                    var best = 0.0;
                    var predict = 0;

                    for (var j = 0; j < perceptrons.Count; j++)
                    {
                        var score = perceptrons[j].Classify(x, xLabel, false);

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
                var accuracy = correct / numClassified * 100;

                // TODO: Fix This, not quiet accurate
                if (Math.Abs(accuracy - lastAccuracy) < tol) {
                    doneTraining = true;
                } else {
                    lastAccuracy = accuracy;
                }

                Console.WriteLine("Epoch: " + e + " Accuracy: " + accuracy);

                for (var i = 0; i < perceptrons.Count; i++) {
                    perceptrons[i].UpdateAlpha();
                    Console.Write(perceptrons[i].Alpha);
                }
                Console.WriteLine();
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

            const string imageFilePath = "data/testimages";
            var testData = ImportImages(imageFilePath);

            const string labelFilePath = "data/testlabels";
            var testLabels = ImportLabels(labelFilePath);

            for (var i = 0; i < testData.Count; i++)
            {
                var x = testData[i];
                var xLabel = testLabels[i];

                var best = 0.0;
                var predict = 0;

                for (var j = 0; j < perceptrons.Count; j++)
                {
                    var score = perceptrons[j].Classify(x, xLabel, false);

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
        private static List<int> ImportLabels(string labelFilePath)
        {
            var labels = "";
            string line;
            var file = new StreamReader(File.OpenRead(labelFilePath));
            while ((line = file.ReadLine()) != null)
            {
                labels += line;
            }

            var dataLabels = labels.Select(label => label - '0').ToList();

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
                Weights = new double[785]
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
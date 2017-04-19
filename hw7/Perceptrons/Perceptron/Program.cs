using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Perceptron
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            var perceptrons = TrainPerceptrons();

            // TODO: Implement Evaluation Function with Accuracy Measure
        }

        /// <summary>
        /// Trains Perceptrons </summary>
        /// <returns>
        /// Returns Trained Perceptrons </returns>
        private static Perceptron[] TrainPerceptrons()
        {
            const string imageFilePath = "D:\\CS\\440\\hw7\\Perceptrons\\Perceptron\\data\\trainingimages";
            var trainingData = ImportImages(imageFilePath);

            const string labelFilePath = "D:\\CS\\440\\hw7\\Perceptrons\\Perceptron\\data\\traininglabels";
            var trainingLabels = ImportLabels(labelFilePath);

            // TODO: Setup Training Set and Validation Set to determine best parameters
            const double alpha = 0.95;
            const int epoch = 50;

            // TODO: Refactor to use ints instead of chars
            // TODO: Refactor into for loop
            var p0 = SetupPerceptron('0', alpha);
            var p1 = SetupPerceptron('1', alpha);
            var p2 = SetupPerceptron('2', alpha);
            var p3 = SetupPerceptron('3', alpha);
            var p4 = SetupPerceptron('4', alpha);
            var p5 = SetupPerceptron('5', alpha);
            var p6 = SetupPerceptron('6', alpha);
            var p7 = SetupPerceptron('7', alpha);
            var p8 = SetupPerceptron('8', alpha);
            var p9 = SetupPerceptron('9', alpha);

            var perceptrons = new Perceptron[] {p0, p1, p2, p3, p4, p5, p6, p7, p8, p9};

            for (var e = 0; e < epoch; e++)
            {
                // TODO: Randomize order of training elements
                for (var i = 0; i < trainingLabels.Length; i++)
                {
                    var X = trainingData[i];
                    var XLabel = trainingLabels[i];

                    foreach (var p in perceptrons)
                    {
                        p.Classify(X, XLabel);
                    }
                }
            }

            return perceptrons;
        }

        /// <summary>
        /// Imports Images as vectors </summary>
        /// <param name="ImageFilePath"> File Path to Image File </param>
        /// <returns>
        /// Returns List of double Array of images </returns>
        private static List<double[]> ImportImages(string ImageFilePath)
        {
            var data = new List<double[]>();

            var counter = 0;
            string line;
            var item = "";
            var file = new System.IO.StreamReader(File.OpenRead(ImageFilePath));
            while ((line = file.ReadLine()) != null)
            {
                item += line;

                if (++counter % 28 == 0)
                {
                    var itemData = new double[item.Length];

                    for (int i = 0; i < item.Length; i++)
                    {
                        itemData[i] = ConvertCharacter(item[i]);
                    }
                    data.Add(itemData);
                    item = "";
                }
            }

            return data;
        }

        /// <summary>
        /// Classifies Input Characters </summary>
        /// <param name="InputValue"> Input Character </param>
        /// <returns>
        /// Returns Classified Value </returns>
        private static double ConvertCharacter(char InputValue)
        {
            switch (InputValue)
            {
                case ' ':
                    return 0.0;
                case '+':
                    return 0.5;
                case '#':
                    return 1.0;
                default:
                    break;
            }

            return 0.0;
        }

        /// <summary>
        /// Imports Labels </summary>
        /// <param name="LabelFilePath"> File Path to Label File </param>
        /// <returns>
        /// Returns Char Array of labels </returns>
        private static char[] ImportLabels(string LabelFilePath)
        {
            var labels = "";
            string line;
            var file = new System.IO.StreamReader(File.OpenRead(LabelFilePath));
            while ((line = file.ReadLine()) != null)
            {
                labels += line;
            }
            // TODO: Refactor to IntArrays
            var dataLabels = labels.ToCharArray();

            return dataLabels;
        }

        /// <summary>
        /// Constructor for Perceptron </summary>
        /// <param name="label"> Label of Perceptron </param>
        /// <param name="alpha"> Learning Rate </param>
        /// <returns>
        /// Returns Initialized Perceptron </returns>
        private static Perceptron SetupPerceptron(char label, double alpha)
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
    }
}
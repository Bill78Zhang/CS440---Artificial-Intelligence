using System;
using System.Collections.Generic;
using System.Security.Cryptography.X509Certificates;
using System.Text;

namespace Perceptron
{
    internal class Perceptron
    {
        public int Label;
        public double Alpha;
        public double[] Weights;

        /// <summary>
        /// Classifies input vector </summary>
        /// <param name="x"> Input Data Vector </param>
        /// <param name="xLabel"> Label for Input </param>
        public double Classify(double[] x, int xLabel)
        {
 	        const int bias = 1;
            var percep = 0;
            if (DotProduct(x) > 0)
            {
                percep = 1;
            }

            var err = MatchLabel(xLabel) - percep;
            var deltaWeight = MultiplyConstant(Alpha * err, x);
            UpdateWeight(deltaWeight);

	        return percep + bias;
        }

        /// <summary>
        /// Updates Weights </summary>
        /// <param name="deltaWeight"> Vector representing values to update by </param>
        private void UpdateWeight(double[] deltaWeight)
        {
            for (var i = 0; i < Weights.Length; i++)
            {
                Weights[i] += deltaWeight[i];
            }
        }
        
        /// <summary>  
        /// Calculates the dot product of two vectors </summary>
        /// <param name="x"> Input Data Vector </param>
        /// <returns>
        /// Returns results of dot product calculation </returns>
        private double DotProduct(double[] x)
        {
            var value = 0.0;
            for (var i = 0; i < Weights.Length; i++)
            {
                value += Weights[i] * x[i];
            }

            return value;
        }

        /// <summary>  
        /// Multiplies a vector by a constant </summary>
        /// <param name="constant"> Constant to multiply by </param>
        /// <param name="x"> Input Data Vector </param>
        /// <returns>
        /// Returns results of multiplication </returns>
        private double[] MultiplyConstant(double constant, double[] x)
        {
            for (var i = 0; i < x.Length; i++)
            {
                x[i] *= constant;
            }

            return x;
        }

        /// <summary>
        /// Checks if input label matches perceptron label </summary>
        /// <param name="xLabel"> input label </param>
        /// <returns>
        /// Returns 1 if matches, 0 otherwise </returns>
        private int MatchLabel(int xLabel)
        {
            return Label == xLabel ? 1 : 0;
        }
    }
}

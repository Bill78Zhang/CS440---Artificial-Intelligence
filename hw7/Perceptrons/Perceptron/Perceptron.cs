using System;
using System.Collections.Generic;
using System.Security.Cryptography.X509Certificates;
using System.Text;

namespace Perceptron
{
    internal class Perceptron
    {
        public char Label;
        public double Alpha;
        public double[] Weights;

        /// <summary>
        /// Classifies input vector </summary>
        /// <param name="X"> Input Data Vector </param>
        /// <param name="XLabel"> Label for Input </param>
        public void Classify(double[] X, char XLabel)
        {
            var percep = 0;
            if (DotProduct(X) > 0)
            {
                percep = 1;
            }

            var err = MatchLabel(XLabel) - percep;
            var deltaWeight = MultiplyConstant(Alpha * err, X);
            UpdateWeight(deltaWeight);
        }

        /// <summary>
        /// Updates Weights </summary>
        /// <param name="DeltaWeight"> Vector representing values to update by </param>
        private void UpdateWeight(double[] DeltaWeight)
        {
            for (var i = 0; i < Weights.Length; i++)
            {
                Weights[i] += DeltaWeight[i];
            }
        }
        
        /// <summary>  
        /// Calculates the dot product of two vectors </summary>
        /// <param name="X"> Input Data Vector </param>
        /// <returns>
        /// Returns results of dot product calculation </returns>
        private double DotProduct(double[] X)
        {
            var value = 0.0;
            for (var i = 0; i < Weights.Length; i++)
            {
                value += Weights[i] * X[i];
            }

            return value;
        }

        /// <summary>  
        /// Multiplies a vector by a constant </summary>
        /// <param name="Constant"> Constant to multiply by </param>
        /// <param name="X"> Input Data Vector </param>
        /// <returns>
        /// Returns results of multiplication </returns>
        private double[] MultiplyConstant(double Constant, double[] X)
        {
            for (var i = 0; i < X.Length; i++)
            {
                X[i] *= Constant;
            }

            return X;
        }

        /// <summary>
        /// Checks if input label matches perceptron label </summary>
        /// <param name="XLabel"> input label </param>
        /// <returns>
        /// Returns 1 if matches, 0 otherwise </returns>
        private int MatchLabel(char XLabel)
        {
            return Label == XLabel ? 1 : 0;
        }
    }
}

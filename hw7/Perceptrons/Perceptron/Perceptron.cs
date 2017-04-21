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
            var dotProduct = DotProduct(x);
            var percep = 0;
            if (dotProduct > 0)
            {
                percep = 1;
            }

            var err = MatchLabel(xLabel) - percep;
            var loss = Alpha * err;
            var deltaWeight = MultiplyConstant(loss, x);
            UpdateWeight(deltaWeight);

	        return dotProduct + bias;
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
            var temp = new double[x.Length];
            for (var i = 0; i < x.Length; i++)
            {
                temp[i] = x[i] * constant;
            }

            return temp;
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

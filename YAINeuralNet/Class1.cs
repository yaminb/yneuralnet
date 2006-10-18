/// <license>
///     This file is part of YNeuralNet.
///
///    YNeuralNet is free software; you can redistribute it and/or modify
///    it under the terms of the GNU General Public License as published by
///    the Free Software Foundation; either version 2 of the License, or
///    (at your option) any later version.
///
///    YNeuralNet is distributed in the hope that it will be useful,
///    but WITHOUT ANY WARRANTY; without even the implied warranty of
///    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
///    GNU General Public License for more details.
///
///    You should have received a copy of the GNU General Public License
///    along with YNeuralNet; if not, write to the Free Software
///    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
/// 
/// </license>
/// <summary>
///     This is a very easy to use neural net class.  
///     Refer to the sample project (YAITest) for documentation on its use.
/// </summary>
/// <author> Yamin Bismilla </author>
/// <created> Oct 18, 2006 </author>
/// <history> </history>
/// 
using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

//Most of this code is derived from 
//Artificial Intelligence A Modern Approach, 2nd Edition
//By stuart Russel and Peter Norvig
//my text for AI course at UWaterloo 4th year...who knew I'd need it again
namespace YAINeuralNet
{

    public class RandomGenerator : Random
    {
        private RandomGenerator() { }
        public double NextDouble(double min, double max)
        {

            return min + (NextDouble() * (max - min));
        }

        static public RandomGenerator Ins()
        {
            if (m_instance == null)
            {
                m_instance = new RandomGenerator();
            }

            return m_instance;
        }
        static private RandomGenerator m_instance = null;
    }

    public class Layer
    {
        /*layer P.745
         *  a(j)--accessed by m_values[j]
         *  W(j)(k) -- accessed by m_weights[j][k] 
         * 
         * prevLayer 
         *  a(k) --accessed by m_values[k]
         * */

        public void Init(int numNodes, Layer pLayer, Net parent)
        {
            m_values = new double[numNodes];
            m_prevLayer = pLayer;
            if( m_prevLayer != null) m_prevLayer.m_nextLayer = this;
            m_parent = parent;

            m_ini = new double[m_values.Length];
            m_delta = new double[m_values.Length];

            if (m_prevLayer == null)
            {
                m_weights = null;
            }
            else
            {
                m_weights = new double[m_values.Length][];
                for (int i = 0; i < m_weights.Length; i++)
                {
                    m_weights[i] = new double[m_prevLayer.m_values.Length];
                }
            }
        }

        public void setValues(double[] values)
        {
            //for now, make a copy...safer
            if (values.Length != m_values.Length)
            {
                throw new Exception("Wrong length");
            }

            Array.Copy(values, m_values, m_values.Length);
        }

        public double[] m_values = null;
        public double[][] m_weights = null;
        public double[] m_ini = null;
        public double[] m_delta = null;
        public Layer m_prevLayer = null;
        public Layer m_nextLayer = null;
        public Net m_parent;
    }

    public interface TrainerProvider
    {
        void ResetData();
        void Reinit();
        double[] GetInputData();
        double[] GetExpectedOutput();

        void NextData();
        bool HasMoreTrainData();
        bool ShouldStopTraining();

        double GetRandomWeight();
    }

    public interface DicreteTestProvider : TrainerProvider
    {
         bool isEqual(double[] output, double[] expectedOutput);
         bool HasMoreTestData();
         TextWriter GetWriter();
    }

    public class TestResultsDiscrete
    {
        public int m_totalTests;
        public int m_correctCount;

        public double GetHitRate()
        {
            return ((double)m_correctCount) / ((double)m_totalTests);
        }
    }

    public interface NetProvider
    {
        double ApplyActivation(double value); //normally sigmoid
        double ApplyActivationPrime(double value);
        double GetAlpha(); //used in setting weights in backwards propogration
    }

    public class Net
    {
        NetProvider m_netP = null;
        TrainerProvider m_trainerP = null;

        Layer[] m_Layers;
        public void GenerateNetwork(int[] unitlayers, NetProvider netP)
        {
            m_netP = netP;
            m_Layers = new Layer[unitlayers.Length];

            Layer prevLayer = null;
            for (int i = 0; i < unitlayers.Length; i++)
            {
                m_Layers[i] = new Layer();
                m_Layers[i].Init(unitlayers[i], prevLayer, this);
                prevLayer = m_Layers[i];
            }
        }

        private Layer InputLayer() { return m_Layers[0]; }
        private Layer OutputLayer() { return m_Layers[m_Layers.Length - 1]; }



        private void PropagateLayerForward(Layer Lower, Layer Upper)
        {
            int i;
            int j;

            for (i = 0; i < Upper.m_values.Length; i++)
            {
                double sum = 0;
                for (j = 0; j < Lower.m_values.Length; j++)
                {
                     sum += Upper.m_weights[i][j] * Lower.m_values[j];
                }
                Upper.m_ini[i] = sum;
                Upper.m_values[i] = m_netP.ApplyActivation(Upper.m_ini[i]);

            }
        }

        private void GetOutputError(double[] expected)
        {
            Layer output = OutputLayer();
            for (int i = 0; i < output.m_values.Length; i++)
            {
                double err = (expected[i] - output.m_values[i]);
                output.m_delta[i] = err * m_netP.ApplyActivationPrime(output.m_ini[i]);
            }
        }

        private void PropagateLayerBackwards(Layer Upper, Layer Lower)
        {
            for (int j = 0; j < Lower.m_values.Length; j++)
            {
                double sum = 0;
                for (int i = 0; i < Upper.m_delta.Length; i++)
                {
                    sum = sum + (Upper.m_delta[i] * Upper.m_weights[i][j]);
                }
                Lower.m_delta[j] = sum * m_netP.ApplyActivationPrime(Lower.m_ini[j]);

                for (int i = 0; i < Upper.m_values.Length; i++)
                {
                    Upper.m_weights[i][j] += m_netP.GetAlpha() * Lower.m_values[j] * Upper.m_delta[i];
                }
            }
        }

        private void PropagateForward()
        {
            for (int l = 0; l < m_Layers.Length - 1; l++)
            {
                PropagateLayerForward(m_Layers[l], m_Layers[l + 1]);
            }
        }

        private void PropogateBackwards()
        {
            for (int l = m_Layers.Length - 1; l >= 1; l--)
            {
                PropagateLayerBackwards(m_Layers[l], m_Layers[l - 1]);
            }
        }

        private void RandomizeWeights()
        {
            for (int l = 1; l < m_Layers.Length; l++)
            {
                for (int i = 0; i < m_Layers[l].m_weights.Length; i++)
                {
                    for (int j = 0; j < m_Layers[l].m_weights[i].Length; j++)
                    {
                        m_Layers[l].m_weights[i][j] = m_trainerP.GetRandomWeight();
                    }

                }
            }
        }

        public void LearnBackProp(TrainerProvider trainer)
        {
            m_trainerP = trainer;

            RandomizeWeights();

            //Page 746
            trainer.Reinit();
            bool stopLearning = false;
            while (!stopLearning)
            {
                trainer.ResetData();
                while (trainer.HasMoreTrainData())
                {

                    trainer.NextData();
                    double[] inData = m_trainerP.GetInputData();
                    double[] expectedData = m_trainerP.GetExpectedOutput();

                    ApplyNetwork(inData);

                    GetOutputError(expectedData);

                    PropogateBackwards();


                }


                stopLearning = m_trainerP.ShouldStopTraining();
            }

        }

        public TestResultsDiscrete TestNetworkDiscreet(DicreteTestProvider trainer)
        {
            TestResultsDiscrete res = new TestResultsDiscrete();
            res.m_correctCount = 0;
            res.m_totalTests = 0;
            
            trainer.Reinit();
            trainer.ResetData();
            while (trainer.HasMoreTrainData())
            {
                trainer.NextData();
                double[] inData = m_trainerP.GetInputData();
                double[] expectedData = m_trainerP.GetExpectedOutput();
                double[] outdata = ApplyNetwork(inData);

                bool hit = trainer.isEqual(outdata, expectedData);
                if( hit)
                {
                    res.m_correctCount++;
                }
                res.m_totalTests++;

    
            }

            return res;
        }

        public double[] ApplyNetwork(double[] input)
        {
            InputLayer().setValues(input);
            PropagateForward();


            return OutputLayer().m_values;
        }

    }
}

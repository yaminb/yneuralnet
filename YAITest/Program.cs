/// <license>
///     This file is part of YAITest.
///
///    YAITest is free software; you can redistribute it and/or modify
///    it under the terms of the GNU General Public License as published by
///    the Free Software Foundation; either version 2 of the License, or
///    (at your option) any later version.
///
///    YAITest is distributed in the hope that it will be useful,
///    but WITHOUT ANY WARRANTY; without even the implied warranty of
///    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
///    GNU General Public License for more details.
///
///    You should have received a copy of the GNU General Public License
///    along with YAITest; if not, write to the Free Software
///    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
/// 
/// </license>
/// <summary>
///     This is the test program for YAINeuralNet 
/// </summary>
/// <author> Yamin Bismilla </author>
/// <created> Oct 18, 2006 </author>
/// <history> </history>
/// 

using System;
using System.Collections.Generic;
using System.Text;
using YAINeuralNet;
using System.IO;

namespace YAITest
{
	//this test data is generated from 
	//http://www.neural-networks-at-your-fingertips.com/bpn.html (Karsten Kutza)
    class SunSpotData : DicreteTestProvider
    {
        static int m_prevYears = 30;
        static double[] Sunspots = {

                        0.0262,  0.0575,  0.0837,  0.1203,  0.1883,  0.3033,  
                        0.1517,  0.1046,  0.0523,  0.0418,  0.0157,  0.0000,  
                        0.0000,  0.0105,  0.0575,  0.1412,  0.2458,  0.3295,  
                        0.3138,  0.2040,  0.1464,  0.1360,  0.1151,  0.0575,  
                        0.1098,  0.2092,  0.4079,  0.6381,  0.5387,  0.3818,  
                        0.2458,  0.1831,  0.0575,  0.0262,  0.0837,  0.1778,  
                        0.3661,  0.4236,  0.5805,  0.5282,  0.3818,  0.2092,  
                        0.1046,  0.0837,  0.0262,  0.0575,  0.1151,  0.2092,  
                        0.3138,  0.4231,  0.4362,  0.2495,  0.2500,  0.1606,  
                        0.0638,  0.0502,  0.0534,  0.1700,  0.2489,  0.2824,  
                        0.3290,  0.4493,  0.3201,  0.2359,  0.1904,  0.1093,  
                        0.0596,  0.1977,  0.3651,  0.5549,  0.5272,  0.4268,  
                        0.3478,  0.1820,  0.1600,  0.0366,  0.1036,  0.4838,  
                        0.8075,  0.6585,  0.4435,  0.3562,  0.2014,  0.1192,  
                        0.0534,  0.1260,  0.4336,  0.6904,  0.6846,  0.6177,  
                        0.4702,  0.3483,  0.3138,  0.2453,  0.2144,  0.1114,  
                        0.0837,  0.0335,  0.0214,  0.0356,  0.0758,  0.1778,  
                        0.2354,  0.2254,  0.2484,  0.2207,  0.1470,  0.0528,  
                        0.0424,  0.0131,  0.0000,  0.0073,  0.0262,  0.0638,  
                        0.0727,  0.1851,  0.2395,  0.2150,  0.1574,  0.1250,  
                        0.0816,  0.0345,  0.0209,  0.0094,  0.0445,  0.0868,  
                        0.1898,  0.2594,  0.3358,  0.3504,  0.3708,  0.2500,  
                        0.1438,  0.0445,  0.0690,  0.2976,  0.6354,  0.7233,  
                        0.5397,  0.4482,  0.3379,  0.1919,  0.1266,  0.0560,  
                        0.0785,  0.2097,  0.3216,  0.5152,  0.6522,  0.5036,  
                        0.3483,  0.3373,  0.2829,  0.2040,  0.1077,  0.0350,  
                        0.0225,  0.1187,  0.2866,  0.4906,  0.5010,  0.4038,  
                        0.3091,  0.2301,  0.2458,  0.1595,  0.0853,  0.0382,  
                        0.1966,  0.3870,  0.7270,  0.5816,  0.5314,  0.3462,  
                        0.2338,  0.0889,  0.0591,  0.0649,  0.0178,  0.0314,  
                        0.1689,  0.2840,  0.3122,  0.3332,  0.3321,  0.2730,  
                        0.1328,  0.0685,  0.0356,  0.0330,  0.0371,  0.1862,  
                        0.3818,  0.4451,  0.4079,  0.3347,  0.2186,  0.1370,  
                        0.1396,  0.0633,  0.0497,  0.0141,  0.0262,  0.1276,  
                        0.2197,  0.3321,  0.2814,  0.3243,  0.2537,  0.2296,  
                        0.0973,  0.0298,  0.0188,  0.0073,  0.0502,  0.2479,  
                        0.2986,  0.5434,  0.4215,  0.3326,  0.1966,  0.1365,  
                        0.0743,  0.0303,  0.0873,  0.2317,  0.3342,  0.3609,  
                        0.4069,  0.3394,  0.1867,  0.1109,  0.0581,  0.0298,  
                        0.0455,  0.1888,  0.4168,  0.5983,  0.5732,  0.4644,  
                        0.3546,  0.2484,  0.1600,  0.0853,  0.0502,  0.1736,  
                        0.4843,  0.7929,  0.7128,  0.7045,  0.4388,  0.3630,  
                        0.1647,  0.0727,  0.0230,  0.1987,  0.7411,  0.9947,  
                        0.9665,  0.8316,  0.5873,  0.2819,  0.1961,  0.1459,  
                        0.0534,  0.0790,  0.2458,  0.4906,  0.5539,  0.5518,  
                        0.5465,  0.3483,  0.3603,  0.1987,  0.1804,  0.0811,  
                        0.0659,  0.1428,  0.4838,  0.8127 

                      };



        int m_curIndex = 0;
        int m_trainCount = 0;
        int m_totalTrainCount = 0;

        public void ResetData()
        {
            m_curIndex = 0;
            m_trainCount = 0;

        }

        public void Reinit()
        {
            m_totalTrainCount = 0;
        }

        public double[] GetInputData()
        {
            double[] ret = new double[m_prevYears];
            for (int i = 0; i < m_prevYears; i++)
            {
                ret[i] = Sunspots[m_curIndex-i-1];
            }
            return ret;
        }

        public double[] GetExpectedOutput()
        {
            double[] ret = new double[1];
            ret[0] = Sunspots[m_curIndex];
            return ret;
        }

        public void NextData()
        {
            m_curIndex = RandomGenerator.Ins().Next(m_prevYears, Sunspots.Length - 1);
            m_trainCount++;
            m_totalTrainCount++;
        }

        public bool HasMoreTrainData()
        {
            return (m_trainCount < Sunspots.Length);
        }

        public bool HasMoreTestData()
        {
            return HasMoreTrainData();
        }

        public bool ShouldStopTraining()
        {
            return m_totalTrainCount > (Sunspots.Length * 1000);
        }

        public double GetRandomWeight()
        {
            //return 5.0;
            return RandomGenerator.Ins().NextDouble(-5.0, 5.0);
        }

        public bool isEqual(double[] output, double[] expectedOutput)
        {
            if (Math.Abs(output[0] - expectedOutput[0]) < 0.1) return true;
            return false;
        }

        public TextWriter GetWriter()
        {
            return null;
        }

    }

    class SunNet : NetProvider
    {
        public double ApplyActivation(double value)
        {
            return (1.0 / (1.0 + Math.Exp(-value)));

        }
        public double ApplyActivationPrime(double value)
        {
            double sig = ApplyActivation(value);
            return sig * (1.0 - sig);
        }
        public double GetAlpha()
        {
            return 0.001;
        }

    }


    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Starting Neural Net test");

            Net net = new Net();
            SunSpotData data = new SunSpotData();
            SunNet snet = new SunNet();

            int[] netLayers = { 30, 20, 10, 1 };
            net.GenerateNetwork(netLayers, snet);
            net.LearnBackProp(data);

           net.TestNetworkDiscreet(data);
        }
    }
}

/// <license>
///     This file is part of YAICompression.
///
///    YAICompression is free software; you can redistribute it and/or modify
///    it under the terms of the GNU General Public License as published by
///    the Free Software Foundation; either version 2 of the License, or
///    (at your option) any later version.
///
///    YAICompression is distributed in the hope that it will be useful,
///    but WITHOUT ANY WARRANTY; without even the implied warranty of
///    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
///    GNU General Public License for more details.
///
///    You should have received a copy of the GNU General Public License
///    along with YAICompression; if not, write to the Free Software
///    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
/// 
/// </license>
/// <summary>
///     This is an example an attempt for machine learning compressions  
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
using System.Collections;

namespace YAICompression
{
    class Utils
    {
        public static double[] ByteArrayToBinArray(byte[] input)
        {
            double[] ret = new double[input.Length * 8];
            for (int i = 0; i < input.Length; i++)
            {
                int bt = input[i];
                bt = bt & 0x00FF;

                int mask = 0x80;
                for (int j = 0; j < 8; j++)
                {
                    int index = (i*8) + j;
                    int val = bt & mask;
                    if (val != 0) ret[index] = 1.0;
                    else ret[index] = 0.0;
                    mask = mask >> 1;
                }

            }
            return ret;
        }


        public static byte[] BinArrayToByteArray(double[] input)
        {
            byte[] ret = new byte[input.Length/8];

            Array.Clear(ret, 0, ret.Length);

            for (int i = 0; i < input.Length; i++)
            {
                double dt = input[i];
                bool bitSet = false;
                if (dt > 0.50) bitSet = true;

                if (bitSet)
                {
                    int workingByte = i / 8;
                    int curBit = i % 8;
                    int mask = 0x0080 >> curBit;

                    int ibyte = ret[workingByte];
                    ibyte = ibyte | mask;
                    ibyte = ibyte & 0x00FF;
                    ret[workingByte] = (byte)ibyte;
                }

            }
            return ret;
        }

        //modified http://www.yoda.arachsys.com/csharp/readbinary.html
        public static byte[] ReadFully(Stream stream, long initialLength)
        {
            
            // If we've been passed an unhelpful initial length, just
            // try and get stream lengh or use 32K.
            if (initialLength < 1)
            {
                try
                {
                    initialLength = stream.Length;
                }
                catch (Exception e)
                {
                    initialLength = 32768;
                }

            }

            byte[] buffer = new byte[initialLength];
            int read = 0;

            int chunk;
            while ((chunk = stream.Read(buffer, read, buffer.Length - read)) > 0)
            {
                read += chunk;

                // If we've reached the end of our buffer, check to see if there's
                // any more information
                if (read == buffer.Length)
                {
                    int nextByte = stream.ReadByte();

                    // End of stream? If so, we're done
                    if (nextByte == -1)
                    {
                        return buffer;
                    }

                    // Nope. Resize the buffer, put in the byte we've just
                    // read, and continue
                    byte[] newBuffer = new byte[buffer.Length * 2];
                    Array.Copy(buffer, newBuffer, buffer.Length);
                    newBuffer[read] = (byte)nextByte;
                    buffer = newBuffer;
                    read++;
                }
            }
            // Buffer is now too big. Shrink it.
            byte[] ret = new byte[read];
            Array.Copy(buffer, ret, read);
            return ret;
        }


    }

    class ByteData : DicreteTestProvider
    {
        public int m_prevBytes = 10;
        public int m_outBytes = 1;
        public int m_totalTrainReps = 10;

        public int getInputNodeCount() { return m_prevBytes*8; }
        public int getOutputNodeCount() { return m_outBytes*8; }
        
        byte[] m_data = null;

        int m_curIndex = 0;
        int m_trainCount = 0;
        int m_totalTrainCount = 0;
        int m_numSamples = 0;
        

        public bool ReadStream(Stream str, long expectedLength)
        {
            m_data = Utils.ReadFully(str, expectedLength);
            return true;
        }

        public void ResetData()
        {
            m_curIndex = m_prevBytes - m_outBytes;
            m_trainCount = 0;

            int adjustLen = (m_data.Length - m_prevBytes);
            int numSample = adjustLen / m_outBytes;
            m_numSamples = numSample;

        }

        public void Reinit()
        {
            m_totalTrainCount = 0;
        }

        public double[] GetInputData()
        {
            byte[] ret = new byte[m_prevBytes];
            for (int i = 0; i < m_prevBytes; i++)
            {
                ret[i] = m_data[m_curIndex - i - 1];
            }
            return Utils.ByteArrayToBinArray(ret);
        }

        public double[] GetExpectedOutput()
        {
            byte[] ret = new byte[m_outBytes];
            for (int i = 0; i < m_outBytes; i++)
            {
                ret[i] = m_data[m_curIndex + i];
            }
            return Utils.ByteArrayToBinArray(ret);
        }

        public void NextData()
        {
            m_curIndex = m_curIndex + m_outBytes;
            m_trainCount++;
            m_totalTrainCount++;
        }

        
        public bool HasMoreTrainData()
        {
            return m_trainCount < ( m_numSamples);
        }

        public bool HasMoreTestData()
        {
            return m_trainCount < (m_numSamples);
        }

        public bool ShouldStopTraining()
        {
            return m_totalTrainCount > (m_numSamples * m_totalTrainReps);
        }

        public double GetRandomWeight()
        {
            return RandomGenerator.Ins().NextDouble(-5.0, 5.0);
        }

        public bool isEqual(double[] output, double[] expectedOutput)
        {
            byte[] bout = Utils.BinArrayToByteArray(output);
            byte[] bexp = Utils.BinArrayToByteArray(expectedOutput);

            for (int i = 0; i < bout.Length; i++)
            {
                if (bout[i] != bexp[i])
                {
                    return false;
                }
            }
            return true;
        }

        public TextWriter GetWriter()
        {
            return null;
        }

    }

    class CompressionNet : NetProvider
    {
        public double m_alpha = 0.01;

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
            return m_alpha;
        }

    }

    class CmdLineHandler
    {
        private Hashtable m_tab = new Hashtable();
        //all arguments are of the form <name>=<identifier>
        //ex: file=C:\test.dat
        //not c# handles quotations and everything from the command line
        //all names automatically converted to small case
        public String getValue(String key, String defVal)
        {
            String val = (String)m_tab[key]; 
            if (val == null) return defVal;
            return val;
        }
        public void Handle(string[] args)
        {
            m_tab.Clear();
            for (int i = 0; i < args.Length; i++)
            {
                try
                {
                    String a = args[i];
                    int ieq = a.IndexOf("=");
                    String name = a.Substring(0, ieq);
                    name = name.ToLower();
                    String value = a.Substring(ieq + 1);
                    m_tab.Add(name, value);
                }
                catch (Exception e)
                {
                }
                
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Starting Neural Net test");
            CmdLineHandler cmd = new CmdLineHandler();

            cmd.Handle(args);

            
            Net net = new Net();
            FileStream fs = File.OpenRead(cmd.getValue("file",""));
            ByteData data = new ByteData();
            data.ReadStream(fs, -1);
            data.m_prevBytes = int.Parse( cmd.getValue("prevbytes","10"));
            data.m_outBytes = int.Parse(cmd.getValue("outbytes","1"));
            data.m_totalTrainReps = int.Parse(cmd.getValue("totaltrainreps","10"));
                        
            CompressionNet snet = new CompressionNet();
            snet.m_alpha =  double.Parse( cmd.getValue("alpha", "0.02") );

            int ic = data.getInputNodeCount();
            int oc = data.getOutputNodeCount();

            int[] netLayers = new int[ int.Parse(cmd.getValue("hcnt","0")) + 2]; //2 for input/output
            netLayers[0] = ic;
            netLayers[netLayers.Length - 1] = oc;
            for(int j=0; j < netLayers.Length - 2; j++)
            {
                String hkey = "h"+ j;
                netLayers[j + 1] = int.Parse(cmd.getValue(hkey, "10"));
            }

            net.GenerateNetwork(netLayers, snet);
            net.LearnBackProp(data);

            TestResultsDiscrete res = net.TestNetworkDiscreet(data);
            Console.WriteLine("Hit Rate=" + res.GetHitRate() * 100.00 + "%");
        }
    }
}

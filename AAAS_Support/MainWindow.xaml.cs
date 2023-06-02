using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace AAAS_Support
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private List<String> GetListOfValue(String elements)
        {
            List<String> ReturnValue = new List<String>();
            String[] a1 = elements.Split(new string[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);

            foreach (var item in a1)
            {
                String[] a2 = item.Split(';');
                String[] a3 = a2[1].Split(':');
                // Debug.WriteLine(a3[1].Trim());
                ReturnValue.Add(a3[1]);
            }

            return ReturnValue;
        }

        private void btnCreateTable_Click(object sender, RoutedEventArgs e)
        {
            String sRandom         = "episode : 0 ; periodicrewards :  0\r\nepisode : 10000 ; periodicrewards :  402\r\nepisode : 20000 ; periodicrewards :  423\r\nepisode : 30000 ; periodicrewards :  432\r\nepisode : 40000 ; periodicrewards :  407\r\nepisode : 50000 ; periodicrewards :  501\r\nepisode : 60000 ; periodicrewards :  482\r\nepisode : 70000 ; periodicrewards :  460\r\nepisode : 80000 ; periodicrewards :  435\r\nepisode : 90000 ; periodicrewards :  465\r\nepisode : 100000 ; periodicrewards :  410\r\nepisode : 110000 ; periodicrewards :  422\r\nepisode : 120000 ; periodicrewards :  400\r\nepisode : 130000 ; periodicrewards :  492\r\nepisode : 140000 ; periodicrewards :  399\r\nepisode : 150000 ; periodicrewards :  441\r\nepisode : 160000 ; periodicrewards :  434\r\nepisode : 170000 ; periodicrewards :  448\r\nepisode : 180000 ; periodicrewards :  394\r\nepisode : 190000 ; periodicrewards :  384\r\nepisode : 200000 ; periodicrewards :  514";
            String sepsilonzero    = "episode : 0 ; periodicrewards :  0\r\nepisode : 10000 ; periodicrewards :  885\r\nepisode : 20000 ; periodicrewards :  1466\r\nepisode : 30000 ; periodicrewards :  1523\r\nepisode : 40000 ; periodicrewards :  1418\r\nepisode : 50000 ; periodicrewards :  1527\r\nepisode : 60000 ; periodicrewards :  1516\r\nepisode : 70000 ; periodicrewards :  1518\r\nepisode : 80000 ; periodicrewards :  1531\r\nepisode : 90000 ; periodicrewards :  1467\r\nepisode : 100000 ; periodicrewards :  1583\r\nepisode : 110000 ; periodicrewards :  1523\r\nepisode : 120000 ; periodicrewards :  1487\r\nepisode : 130000 ; periodicrewards :  1520\r\nepisode : 140000 ; periodicrewards :  1576\r\nepisode : 150000 ; periodicrewards :  1469\r\nepisode : 160000 ; periodicrewards :  1498\r\nepisode : 170000 ; periodicrewards :  1466\r\nepisode : 180000 ; periodicrewards :  1497\r\nepisode : 190000 ; periodicrewards :  1480\r\nepisode : 200000 ; periodicrewards :  1466";
            String sdecreasingzero = "episode : 0 ; periodicrewards :  0\r\nepisode : 10000 ; periodicrewards :  0\r\nepisode : 20000 ; periodicrewards :  859\r\nepisode : 30000 ; periodicrewards :  1700\r\nepisode : 40000 ; periodicrewards :  2056\r\nepisode : 50000 ; periodicrewards :  2682\r\nepisode : 60000 ; periodicrewards :  4664\r\nepisode : 70000 ; periodicrewards :  6366\r\nepisode : 80000 ; periodicrewards :  5462\r\nepisode : 90000 ; periodicrewards :  5775\r\nepisode : 100000 ; periodicrewards :  6701\r\nepisode : 110000 ; periodicrewards :  5216\r\nepisode : 120000 ; periodicrewards :  4889\r\nepisode : 130000 ; periodicrewards :  6058\r\nepisode : 140000 ; periodicrewards :  7284\r\nepisode : 150000 ; periodicrewards :  5636\r\nepisode : 160000 ; periodicrewards :  6785\r\nepisode : 170000 ; periodicrewards :  7157\r\nepisode : 180000 ; periodicrewards :  5659\r\nepisode : 190000 ; periodicrewards :  7258\r\nepisode : 200000 ; periodicrewards :  5261";

            //String[] aRandom = sRandom.Split(new string[] { "\r\n", "\r", "\n" },StringSplitOptions.None);

            //foreach (var item in aRandom)
            //{
            //    String[] bRandom = item.Split(';');
            //    String[] cRandom = bRandom[1].Split(':');
            //    Debug.WriteLine(cRandom[1].Trim());
            //}

            List<String> randoms = GetListOfValue(sRandom);
            List<String> epsilonzeros = GetListOfValue(sepsilonzero);
            List<String> decreasingzeros = GetListOfValue(sdecreasingzero);

            Debug.WriteLine(@"\begin{center}");
            Debug.WriteLine(@"\begin{tabular}{||c c c c||}");
            Debug.WriteLine(@"\hline");
            Debug.WriteLine(@" Episode & Random & $\epsilon$-0 & Decr-0 \\ [0.5ex] ");
            Debug.WriteLine(@"\hline\hline");
            for (int i = 0; i < randoms.Count; i++)
            {
                String Line = String.Format($"{i} & {randoms[i]} & {epsilonzeros[i]} & {decreasingzeros[i]}" + @" \\");
                Debug.WriteLine(Line);
            }
            Debug.WriteLine(@"\end{tabular}");
            Debug.WriteLine(@"\end{center}");


        }

    }
}

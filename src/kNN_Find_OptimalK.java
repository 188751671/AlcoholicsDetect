//use this kNN function as an example template 

import java.lang.Math;
import java.util.*;
import java.io.File;
import java.io.IOException;

public class kNN_Find_OptimalK
{
    static int TRAIN_SIZE=400; //no. training patterns
    static int VAL_SIZE=200; //no. testing patterns
    static int FEATURE_SIZE=61; //no. of features


    static double[][] train = new double[TRAIN_SIZE][FEATURE_SIZE]; //training data
    static double[][] val = new double[VAL_SIZE][FEATURE_SIZE]; //validation data
    static int[] train_label = new int[TRAIN_SIZE]; //actual target/class label for train data
    static int[] val_label = new int[VAL_SIZE]; //actual target/class label for validation data
    static int optimalK;
    static int HighestK = TRAIN_SIZE / 2;

    public static void main(String[] args) throws IOException {
        Load_Data();
        double K1 = getAccuracy(1, true);

        int MaxK;
        if (TRAIN_SIZE<=2) return;
        if (TRAIN_SIZE < 9)
            MaxK = TRAIN_SIZE;
        else
            MaxK = HighestK;      //   search optimal K from 0 to HighestK

        double[] accuracyList = new double[MaxK];
        accuracyList[0] = K1;
        for (int i=1; i < MaxK; i++) {
            accuracyList[i] = getAccuracy(i + 1, true);
            //System.out.println(accuracyList[i]);
        }
        // Find the biggest accuracy & optimal K
        double maxAccuracy = 0;
        for (int i=0; i<accuracyList.length; i++){
            if (accuracyList[i] > maxAccuracy) {
                maxAccuracy = accuracyList[i];
                optimalK = i+1;
            }
        }

        System.out.println("Optimal K:  " + optimalK + "    Accuracy:" + maxAccuracy);
    }

    public static void Load_Data() throws IOException {

        String train_file="alco_train_data.txt"; //read training data
        try (Scanner tmp = new Scanner(new File(train_file))) {
            for (int i=0; i<TRAIN_SIZE; i++)
                for (int j=0; j<FEATURE_SIZE; j++)
                    if(tmp.hasNextDouble())
                        train[i][j]=tmp.nextDouble();
            tmp.close();
        }

        String train_label_file="alco_train_label.txt"; //read train label
        try (Scanner tmp = new Scanner(new File(train_label_file))) {
            for (int i=0; i<TRAIN_SIZE; i++)
                if(tmp.hasNextInt())
                    train_label[i]=tmp.nextInt();
            tmp.close();
        }

        String val_file="alco_val_data.txt"; //read validation data
        try (Scanner tmp = new Scanner(new File(val_file))) {
            for (int i=0; i<VAL_SIZE; i++)
                for (int j=0; j<FEATURE_SIZE; j++)
                    if(tmp.hasNextDouble())
                        val[i][j]=tmp.nextDouble();
            tmp.close();
        }

        String val_label_file="alco_val_label.txt"; //read validation label (to obtain classification accuracy)
        try (Scanner tmp = new Scanner(new File(val_label_file))) {
            for (int i=0; i<VAL_SIZE; i++)
                if(tmp.hasNextInt())
                    val_label[i]=tmp.nextInt();
            tmp.close();
        }
    }

    private static double getAccuracy(int NUM_NEIGHBOUR, boolean useManhattan) {

        double[] y = new double[FEATURE_SIZE]; //temp variable holding one pattern of validation data
        double[] x = new double[FEATURE_SIZE]; //temp variable holding one pattern of train data
        double[][] dist_label = new double[TRAIN_SIZE][2]; // Storing distance & label pairs. After sorting, K neighbours are the first K items

        int[] neighbour = new int[NUM_NEIGHBOUR];
        int[] predicted_class = new int[VAL_SIZE];

        for (int j = 0; j < VAL_SIZE; j++) {           // Each pattern of 200 vali_data  compares with    Each of 400 train_data

            for (int f = 0; f < FEATURE_SIZE; f++)   // every feature
                y[f] = val[j][f];                                                     // get a pattern (one line) of 200 vali_data to y[]

            for (int i = 0; i < TRAIN_SIZE; i++)                                        // loop 400 times
            {
                for (int f = 0; f < FEATURE_SIZE; f++) {
                    x[f] = train[i][f];                                                 // get a pattern (one line) of 400 train_data to x[]
                }

                if (useManhattan){
                    // Manhattan Distance
                    double sum = 0.0;
                    for (int f = 0; f < FEATURE_SIZE; f++) {
                        sum = sum + Math.abs(x[f] - y[f]);
                    }
                    dist_label[i][0] = sum;                                                 // dist_label[400][2]  storing one of 200 comparison results
                    dist_label[i][1] = train_label[i];
                }else {
                    // Euclidean Distance
                    double sum=0.0;
                    for (int f=0; f<FEATURE_SIZE; f++)
                        sum=sum + ((x[f]-y[f])*(x[f]-y[f]));
                    dist_label[i][0] = Math.sqrt(sum);
                    dist_label[i][1] = train_label[i];
                }
            }

            Sort(dist_label, 1); //sorting

            for (int n = 0; n < NUM_NEIGHBOUR; n++) {
                neighbour[n] = (int) dist_label[n][1];                                  // get K neighbours' classes
            }
            predicted_class[j] = Mode(neighbour);                                       // get the most frequent number( class ) in the array

        }   // predicted_class[200]  storing 200 results


        // Accuracy Computation:    Compare predicted_class with Validation labels
        int success = 0;
        for (int j = 0; j < VAL_SIZE; j++)
            if (predicted_class[j] == val_label[j])
                success = success + 1;
        double accuracy = (success * 100.0) / VAL_SIZE;

        return accuracy;
    }


    public static void Sort (double[][] sort_array, final int column_sort) {  //sorting function
        Arrays.sort(sort_array, new Comparator<double[]>()
        {
            @Override
            public int compare(double[] a, double[] b)
            {
                if(a[column_sort-1] > b[column_sort-1]) return 1;
                else return -1;
            }
        });
    }


    public static int Mode(int neigh[]) {    // return the most frequent number in an array
        int modeVal=0;
        int maxCnt=0;

        for (int i = 0; i < neigh.length; ++i)
        {
            int count = 0;
            for (int j = 0; j < neigh.length; ++j) {
                if (neigh[j] == neigh[i])
                    count=count+1;
            }
            if (count > maxCnt) {
                maxCnt = count;
                modeVal = neigh[i];
            }
        }
        return modeVal;
    }


}
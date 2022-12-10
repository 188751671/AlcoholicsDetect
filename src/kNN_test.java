import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Scanner;

public class kNN_test {
    static int TRAIN_SIZE = 400; //no. training patterns
    static int TEST_SIZE = 200; //no. testing patterns
    static int FEATURE_SIZE = 61; //no. of features


    static double[][] train = new double[TRAIN_SIZE][FEATURE_SIZE]; //training data
    static double[][] test = new double[TEST_SIZE][FEATURE_SIZE]; //validation data
    static int[] train_label = new int[TRAIN_SIZE]; //actual target/class label for train data
    static int K = 14;

    public static void main(String[] args) throws IOException {
        Load_Data();
        kNN_output(getPredicted_Class());
    }

    private static int[] getPredicted_Class() {
        double[] y = new double[FEATURE_SIZE]; //temp variable holding one pattern of validation data
        double[] x = new double[FEATURE_SIZE]; //temp variable holding one pattern of train data
        double[][] dist_label = new double[TRAIN_SIZE][2]; // Storing distance & label pairs. After sorting, K neighbours are the first K items

        int[] neighbour = new int[K];
        int[] predicted_class = new int[TEST_SIZE];

        for (int j = 0; j < TEST_SIZE; j++) {           // Each pattern of 200 vali_data  compares with    Each of 400 train_data

            for (int f = 0; f < FEATURE_SIZE; f++)   // every feature
                y[f] = test[j][f];                                                     // get a pattern (one line) of 200 vali_data to y[]

            for (int i = 0; i < TRAIN_SIZE; i++)                                        // loop 400 times
            {
                for (int f = 0; f < FEATURE_SIZE; f++) {
                    x[f] = train[i][f];                                                 // get a pattern (one line) of 400 train_data to x[]
                }

                // Manhattan Distance
                double sum = 0.0;
                for (int f = 0; f < FEATURE_SIZE; f++) {
                    sum = sum + Math.abs(x[f] - y[f]);
                }
                dist_label[i][0] = sum;                                                 // dist_label[400][2]  storing one of 200 comparison results
                dist_label[i][1] = train_label[i];
            }

            Sort(dist_label, 1); //sorting

            for (int n = 0; n < K; n++) {
                neighbour[n] = (int) dist_label[n][1];                                  // get K neighbours' classes
            }
            predicted_class[j] = Mode(neighbour);                                       // get the most frequent number( class ) in the array

        }
        return predicted_class;                                                // predicted_class[200]  storing 200 final results
    }

    private static void Load_Data() throws IOException {

        String train_file = "alco_train_data.txt"; //read training data
        try (Scanner tmp = new Scanner(new File(train_file))) {
            for (int i = 0; i < TRAIN_SIZE; i++)
                for (int j = 0; j < FEATURE_SIZE; j++)
                    if (tmp.hasNextDouble())
                        train[i][j] = tmp.nextDouble();
            tmp.close();
        }

        String train_label_file = "alco_train_label.txt"; //read train label
        try (Scanner tmp = new Scanner(new File(train_label_file))) {
            for (int i = 0; i < TRAIN_SIZE; i++)
                if (tmp.hasNextInt())
                    train_label[i] = tmp.nextInt();
            tmp.close();
        }

        String val_file = "alco_test_data.txt"; //read test data
        try (Scanner tmp = new Scanner(new File(val_file))) {
            for (int i = 0; i < TEST_SIZE; i++)
                for (int j = 0; j < FEATURE_SIZE; j++)
                    if (tmp.hasNextDouble())
                        test[i][j] = tmp.nextDouble();
            tmp.close();
        }
    }

    private static void Sort(double[][] sort_array, final int column_sort) {  //sorting function
        Arrays.sort(sort_array, new Comparator<double[]>() {
            @Override
            public int compare(double[] a, double[] b) {
                if (a[column_sort - 1] > b[column_sort - 1]) return 1;
                else return -1;
            }
        });
    }

    private static int Mode(int neigh[]) {    // return the most frequent number in an array
        int modeVal = 0;
        int maxCnt = 0;

        for (int i = 0; i < neigh.length; ++i) {
            int count = 0;
            for (int j = 0; j < neigh.length; ++j) {
                if (neigh[j] == neigh[i])
                    count = count + 1;
            }
            if (count > maxCnt) {
                maxCnt = count;
                modeVal = neigh[i];
            }
        }
        return modeVal;
    }

    // write  kNN_output.txt
    private static void kNN_output(int[] predicted_class) {
        try {
            PrintWriter writer = new PrintWriter("kNN_output.txt", "UTF-8");
            for (int j = 0; j < TEST_SIZE; j++)
                writer.print(predicted_class[j] + " ");
            writer.close();
        } catch (Exception e) {
            System.out.println(e);
        }
    }

}
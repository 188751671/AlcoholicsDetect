import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.util.Scanner;

public class kNN_GA_Test implements Runnable{
    private static Random random = new Random();
    public static int TRAIN_SIZE=400; //no. training patterns
    public static int VAL_SIZE=200; //no. validation patterns
    public static int FEATURE_SIZE=61; //no. of features

    static class FitnessAndFeatures implements Cloneable{
        double accuracy = 0.0;
        double fitness = 0.0;
        boolean[] best_sol = new boolean[FEATURE_SIZE];
        boolean Roulette1_OR_Tournament0;
        int Tournament_Selection_K;
        boolean Single0_or_Multi1_Points_CrossOver;
        int HowMany_PointsTo_CrossOver;
        int CrossOver_Possibility;
        int Mutation_Possibility;
        @Override
        public Object clone() throws CloneNotSupportedException {           // shallow copy
            FitnessAndFeatures newObj = (FitnessAndFeatures) super.clone();
            return newObj;
        }
    }

    public static double[][] train = new double [TRAIN_SIZE][FEATURE_SIZE]; //data to train
    public static double[][] val = new double [VAL_SIZE][FEATURE_SIZE]; //validation data
    public static int[] train_label=new int[TRAIN_SIZE]; //actual target/class label for train data
    public static int[] val_label=new int[VAL_SIZE]; //actual target/class label for validation data

    //TODO   Parameters
    static int maxNumberOfThreads =80;
    static int ThreadMakerInterval = 100; // Millisecond
    static int kNN_Optimal_K = 14; //replace with optimal k value

    static int POP_SIZE=100;
    static int MAX_GEN=25;

    static FitnessAndFeatures TopFit_Features = new FitnessAndFeatures();   // top Fitness from all threads' results

    public static void main(String[] args) throws IOException, InterruptedException {
        Load_Data(TRAIN_SIZE, VAL_SIZE, FEATURE_SIZE); //load data

        // WriteFile Thread
        new Thread(() -> {                                                        // a specialised Thread writing the best result into file  every 10 seconds
            try{
                double onChange = 0.1;
                while (true) {

                    Thread.sleep(1000 * 10);
                    //System.out.println("Write Down Thread " + Thread.currentThread().getName() +" ID:"+ Thread.currentThread().getId());

                    if (TopFit_Features.accuracy > onChange) {

                        double accuracySum = 0;         // get the average of 10 times run, to make sure valid
                        for (int i=0; i<10; i++)
                            accuracySum += getAccuracy(TopFit_Features.best_sol);
                        double averageAccuracy = accuracySum/10;

                        if (onChange < averageAccuracy){
                            onChange = averageAccuracy;

                            PrintWriter writer = new PrintWriter("topParameters.txt", "UTF-8");
                            writer.println("10 times AverageAccuracy: " + averageAccuracy);
                            writer.println("Fitness: " + TopFit_Features.fitness);
                            writer.println("Roulette1_OR_Tournament0:  " + TopFit_Features.Roulette1_OR_Tournament0);
                            writer.println("Tournament_Selection_K: " + TopFit_Features.Tournament_Selection_K);
                            writer.println("Single0_or_Multi1_Points_CrossOver: " + TopFit_Features.Single0_or_Multi1_Points_CrossOver);
                            writer.println("HowMany_PointsTo_CrossOver: " + TopFit_Features.HowMany_PointsTo_CrossOver);
                            writer.println("CrossOver_Possibility: " + TopFit_Features.CrossOver_Possibility);
                            writer.println("Mutation_Possibility: " + TopFit_Features.Mutation_Possibility);

                            for (int j = 0; j < FEATURE_SIZE; j++)       // Write Features
                                if (TopFit_Features.best_sol[j])
                                    writer.print("1 ");
                                else
                                    writer.print("0 ");

                            writer.close();
                        }

                    }
                }
            }
            catch(Exception e) {
                System.out.println(e);
            }
        }).start();


        // Thread Maker
        while (true) {
            int count = Thread.activeCount();
            //System.out.println(Thread.activeCount() + " threads are running");
            if (count < maxNumberOfThreads) {
                Thread GA_Thread = new Thread(new kNN_GA_Test());
                GA_Thread.start();
            }
            Thread.sleep(ThreadMakerInterval);
        }
    }

    public void run()
    {
        Instant start = Instant.now();

        //TODO  Parameters Range Section
        FitnessAndFeatures fitnessAndFeatures = new FitnessAndFeatures();
        fitnessAndFeatures.Roulette1_OR_Tournament0 = random.nextBoolean();                   // True or False
        fitnessAndFeatures.Tournament_Selection_K = random.nextInt(6) + 2;            // 2-7
        fitnessAndFeatures.Single0_or_Multi1_Points_CrossOver = random.nextBoolean();        // True or False        10*2*20*7*5   = 14000 possibilities
        fitnessAndFeatures.HowMany_PointsTo_CrossOver = random.nextInt(15) + 1;          // 1-15

        int CrossoverPossibility_base = 50;     // 50 - 90 中间的数
        int CrossoverPossibility_range= 40;     // random 此数,来获得       本次 CroPoss
        int MutationPossibility_base  = 10;     // 10 - 40 中间的数
        int MutationPossibility_range = 30;     // random 此数,来获得       本次 MutaPoss

        fitnessAndFeatures.CrossOver_Possibility = CrossoverPossibility_base + random.nextInt(CrossoverPossibility_range);
        fitnessAndFeatures.Mutation_Possibility = MutationPossibility_base + random.nextInt(MutationPossibility_range);

        fitnessAndFeatures = GA(fitnessAndFeatures);    // 使用同一个变量做 参数+接收, 传给GA时  包含了上面的参数   ,而返回时  填充好了 accuracy + fitness + best_sol

        if (fitnessAndFeatures.accuracy > TopFit_Features.accuracy){
            try {
                TopFit_Features = (FitnessAndFeatures) fitnessAndFeatures.clone();
            } catch (CloneNotSupportedException e) {
                throw new RuntimeException(e);
            }
        }

        Instant end = Instant.now();
        System.out.println("ending Thread: "+Thread.currentThread().getName() +" ID: "+Thread.currentThread().getId()+" Duration: "+Duration.between(start,end).getSeconds() + " seconds");
    }


    public static void Sort (double[][] sort_array, final int column_sort) {
        Arrays.sort(sort_array, new Comparator<double[]>() {
            @Override
            public int compare(double[] a, double[] b) {
                if(a[column_sort-1] > b[column_sort-1]) return 1;
                else return -1;
            }
        });
    }


    public static int Mode(int neigh[]) {
        int modeVal=0;
        int maxCnt=0;

        for (int i = 0; i < neigh.length; ++i) {
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


    public static void Load_Data(int TRAIN_SIZE, int VAL_SIZE, int FEATURE_SIZE) throws IOException {

        String train_file="alco_train_data.txt"; //read training data
        try (Scanner tmp = new Scanner(new File(train_file))) {
            for (int i=0; i<TRAIN_SIZE; i++)
                for (int j=0; j<FEATURE_SIZE; j++)
                    if(tmp.hasNextDouble())
                        train[i][j]=tmp.nextDouble();
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

        String train_label_file="alco_train_label.txt"; //read train label
        try (Scanner tmp = new Scanner(new File(train_label_file))) {
            for (int i=0; i<TRAIN_SIZE; i++)
                if(tmp.hasNextInt())
                    train_label[i]=tmp.nextInt();
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


    public static FitnessAndFeatures GA(FitnessAndFeatures allParameters) {

        double[] fitness =new double[POP_SIZE];
        double[] accuracy =new double[POP_SIZE];

        boolean[][] sol = new boolean[POP_SIZE][FEATURE_SIZE];
        boolean[][] new_sol = new boolean[POP_SIZE][FEATURE_SIZE];

        boolean[] temp_sol = new boolean[FEATURE_SIZE];

        //create initial population
        for(int j=0; j<POP_SIZE; j++){
            int count=0;
            for(int k=0; k<FEATURE_SIZE; k++) {
                if (count<40) {
                    sol[j][k] = (Math.random() > 0.5);     // random true/false filling sol[100][61]
                    temp_sol[k] = sol[j][k];              // temp_sol[61] = sol[one][61]
                    if (temp_sol[k])
                        count++;
                }
            }                                       // count is how many true in one population: sol[one][61]
            //modify fitness to include both increasing accuracy and minimising features
            //fitness[j]=(double) getAccuracy(train, val, train_label, val_label, temp_sol)   -   (1.0*count/FEATURE_SIZE);       // double fitness[100]  holding accuracies
        }

        System.arraycopy(sol, 0, new_sol, 0, sol.length); //copy initial array, new_sol=sol;


        for (int gen=0; gen<MAX_GEN; gen++) { //do for many generations       -   (1.0*count/FEATURE_SIZE)

            System.arraycopy(new_sol, 0, sol, 0, new_sol.length); //sol=new_sol;  put new gen(new_sol) on Sol, for later altering new_sol

            //compute fitness
            for (int j = 0; j < POP_SIZE; j++) {
                int count = 0;
                for (int k = 0; k < FEATURE_SIZE; k++) {
                    temp_sol[k] = sol[j][k];
                    if (temp_sol[k])
                        count++;
                }
                accuracy[j] = getAccuracy(temp_sol);
                fitness[j] = accuracy[j] - (1.0 * count / FEATURE_SIZE);
                //System.out.print(fitness[j] + " ");
            }


            // Find max Fitness of this generation, and compare with the BestFitness from all generations
            int maxAt = 0;
            for (int j = 0; j < POP_SIZE; j++)
                maxAt = fitness[j] > fitness[maxAt] ? j : maxAt;

            if (fitness[maxAt] > allParameters.fitness) {
                allParameters.accuracy = accuracy[maxAt];
                allParameters.fitness = fitness[maxAt];

                for (int k = 0; k < FEATURE_SIZE; k++)
                    allParameters.best_sol[k] = sol[maxAt][k];
            }
            // System.out.println("Best fitness in Gen"+ gen + " = " + fitness[maxAt]);
            // System.out.println("Accuracy =  " + getAccuracy(train, val, train_label, val_label, Best_Sol));


            if (allParameters.Roulette1_OR_Tournament0) {

                // Roulette Wheel Selection
                double AccuracySum = 0;
                for (int j = 0; j < POP_SIZE; j++)
                    AccuracySum += accuracy[j];

                random.nextDouble(AccuracySum);
                for (int i = 1; i < POP_SIZE; i++) {
                    double temp_sum = AccuracySum;
                    for (int j = 0; j < POP_SIZE; j++) {
                        temp_sum -= accuracy[j];
                        if (temp_sum < 0) {
                            new_sol[i] = sol[j];
                            break;
                        }
                    }
                }
            }else {

                // Tournament Selection with K.   select K population to compare, and select randomly 100 such pairs
                for (int i = 0; i < POP_SIZE; i++) {
                    double selectedPop[][] = new double[allParameters.Tournament_Selection_K][2];   // [0] storing ramdom index , [1] storing Its fitness value which will be compared
                    for (int u = 0; u < allParameters.Tournament_Selection_K; u++) {
                        selectedPop[u][0] = random.nextInt(POP_SIZE);

                        selectedPop[u][1] = fitness[(int) selectedPop[u][0]];
                    }
                    Sort(selectedPop, 2);
                    int winner = (int) selectedPop[allParameters.Tournament_Selection_K - 1][0];

                    new_sol[i] = sol[winner];                           // because here we need select winners from old Sol, and put it in NewSol, we need to maintain NewSol & Sol
                }
            }


            // 如果出现 Cros + Muta  > 100时,  那就random(他们的和)  大于就去 其中一个,否则另一个.
            // < 100时  poss = random(100),   poss < 其中一个时, 去它     < 两个的和, 另一个      否则去neither

            int Cross1_or_Mutate2_or_Neither0;
            if (allParameters.CrossOver_Possibility + allParameters.Mutation_Possibility > 100){
                if (random.nextInt(allParameters.CrossOver_Possibility + allParameters.Mutation_Possibility) < allParameters.CrossOver_Possibility) {
                    // Go Cross
                    Cross1_or_Mutate2_or_Neither0 = 1;
                }else {
                    // Go Mutate
                    Cross1_or_Mutate2_or_Neither0 = 2;
                }
            }else {
                int poss = random.nextInt(100);
                if (poss < allParameters.CrossOver_Possibility) {
                    // Go Cross
                    Cross1_or_Mutate2_or_Neither0 = 1;
                }else if (poss < allParameters.CrossOver_Possibility + allParameters.Mutation_Possibility){
                    // Go Mutate
                    Cross1_or_Mutate2_or_Neither0 = 2;
                }else{
                    // Go neither
                    Cross1_or_Mutate2_or_Neither0 = 0;
                }
            }


            if (Cross1_or_Mutate2_or_Neither0 == 1) {

                // Multiple Points Crossover
                if (allParameters.Single0_or_Multi1_Points_CrossOver) {

                    for (int i = 0; i < POP_SIZE / 2; i++) {
                        for (int k = 0; k < allParameters.HowMany_PointsTo_CrossOver; k++) {
                            int point = random.nextInt(FEATURE_SIZE);

                            int random1 = random.nextInt(POP_SIZE);
                            int random2 = random.nextInt(POP_SIZE);
                            System.arraycopy(sol[random2], point, new_sol[random1], point, 1);
                            System.arraycopy(sol[random1], point, new_sol[random2], point, 1);
                        }
                    }
                } else{

                    // Single Point Crossover
                    int Crossover_StartAt = random.nextInt(FEATURE_SIZE);
                    for (int i = 0; i < POP_SIZE / 2; i++) {
                        int random1 = random.nextInt(POP_SIZE);
                        int random2 = random.nextInt(POP_SIZE);
                        // srcPos through srcPos+length-1 in the source array are copied into positions destPos through destPos+length-1
                        System.arraycopy(sol[random2], Crossover_StartAt, new_sol[random1], Crossover_StartAt, FEATURE_SIZE - Crossover_StartAt);
                    }
                }
            }else if(Cross1_or_Mutate2_or_Neither0 == 2){

                // Mutation
                if (random.nextInt(100) < allParameters.Mutation_Possibility) {
                    int MutateAt = random.nextInt(FEATURE_SIZE);
                    int whichToMutate = random.nextInt(POP_SIZE);
                    new_sol[whichToMutate][MutateAt] = !new_sol[whichToMutate][MutateAt];
                }
            }else {
                // neither Crossover, nor Mutation
            }

        } //end of gen loop


        return allParameters;

    }


    public static double getAccuracy(boolean[] sol) {
        double[][] dist_label = new double[TRAIN_SIZE][2]; //distance array, no of columns+1 to accomodate distance
        double[] y = new double[FEATURE_SIZE];
        double[] x = new double[FEATURE_SIZE];

        int[] neighbour = new int[kNN_Optimal_K];
        int[] predicted_class = new int[VAL_SIZE];

        for (int j=0; j<VAL_SIZE; j++) //for every validation data
        {
            for (int f=0; f<FEATURE_SIZE; f++)
                if (sol[f])
                    y[f]=val[j][f];             // (random generated) IF sol[f]==true   give Y  the vali Feature value,  else give 0
                else
                    y[f]=0.0;

            for (int i=0; i<TRAIN_SIZE; i++)
            {
                for (int f=0; f<FEATURE_SIZE; f++)
                    if (sol[f])
                        x[f]=train[i][f];               // sol[f]==true   give X  the train Feature value,  else give 0
                    else
                        x[f]=0.0;                               // according sol[61] true or false, part of train + val data are zero ed
                                                                // then get the predicted_class & vali_lable pairs
                                                                // Sort, get K(14) items , get and return accuracy
                // Manhattan Distance
                double sum = 0.0;
                for (int f = 0; f < FEATURE_SIZE; f++) {
                    sum = sum + Math.abs(x[f] - y[f]);
                }
                dist_label[i][0] = sum;                                                 // dist_label[400][2]  storing one of 200 comparison results
                dist_label[i][1] = train_label[i];
            }

            Sort(dist_label,1); //Sorting distance

            for (int n=0; n<kNN_Optimal_K; n++) //training label from required neighbours
                neighbour[n]=(int) dist_label[n][1];

            predicted_class[j]=Mode(neighbour);

        } //end val data loop

        int success=0;
        for (int j=0; j<VAL_SIZE; j++)
            if (predicted_class[j]==val_label[j])
                success++;

        double accuracy = (success*1.0)/VAL_SIZE;
        return accuracy;
        //System.out.print(accuracy + " ");
    }

} //end class loop
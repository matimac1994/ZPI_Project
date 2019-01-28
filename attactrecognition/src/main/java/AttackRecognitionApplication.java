import utils.Paths;

import java.util.Arrays;

public class AttackRecognitionApplication {


    public static void main(String[] args) {
        DataManagerContract dataManagerContract = new CSVManager();
        dataManagerContract.read(Paths.PIMA_DATA_CSV);
        dataManagerContract.saveTrainData(Paths.PIMA_TRAIN_DATA_CSV);
        dataManagerContract.saveTestData(Paths.PIMA_TEST_DATA_CSV);

        dataManagerContract.readByIndexes(Paths.PIMA_TRAIN_DATA_CSV, Arrays.asList("0", "1", "2", "3")).forEach(
                record -> System.out.println(record.getValues() + "\n")
        );

    }
}

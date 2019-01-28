import model.DataHeaders;
import model.Record;
import utils.Paths;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class AttackRecognitionApplication {

    private static final DataHeaders dataHeaders = new DataHeaders("9", Arrays.asList("0", "1", "2", "3", "4", "5", "6", "7", "8"));

    public static void main(String[] args) throws IOException {
        //read data
        IDataReader csvDataReader = new CSVReader(Paths.PIMA_DATA_CSV);
        //write train, test data
        IDataWriter csvTrain = new CSVWriter(Paths.PIMA_TRAIN_DATA_CSV, dataHeaders.getAllHeaders());
        IDataWriter csvTest = new CSVWriter(Paths.PIMA_TEST_DATA_CSV, dataHeaders.getAttributes());
        csvTrain.write(prepareTrain(csvDataReader));
        csvTest.write(prepareTest(csvDataReader));
        //show examples
        IDataReader trainDataReader = new CSVReader(Paths.PIMA_TRAIN_DATA_CSV);
        trainDataReader.readByIndexes(Arrays.asList("0", "1", "2", "3")).forEach(
                record -> System.out.println(record.getValues() + "\n")
        );
    }

    private static List<Record> prepareTrain(IDataReader csvDataReader) {
        return SplitData.splitToTrain(csvDataReader.readByIndexes(dataHeaders.getAllHeaders()));
    }

    private static List<Record> prepareTest(IDataReader csvDataReader) {
        return SplitData.spliToTest(csvDataReader.readByIndexes(dataHeaders.getAttributes()));
    }
}

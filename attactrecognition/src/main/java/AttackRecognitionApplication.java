import model.DataHeaders;
import model.Record;

import java.io.IOException;
import java.util.List;

import static utils.Paths.*;

public class AttackRecognitionApplication {

    private static final DataHeaders dataHeaders = new DataHeaders(NET_CLASS_NAME, NET_ATTRS);

    public static void main(String[] args) throws IOException {
        //read data
        IDataReader csvDataReader = new CSVReader(NET_DATA_CSV);
        //write train, test data
        IDataWriter csvTrain = new CSVWriter(NET_TRAIN_DATA_CSV, dataHeaders.getAllHeaders());
        IDataWriter csvTest = new CSVWriter(NET_TEST_DATA_CSV, dataHeaders.getAttributes());
        csvTrain.write(prepareTrain(csvDataReader));
        csvTest.write(prepareTest(csvDataReader));
    }

    private static List<Record> prepareTrain(IDataReader csvDataReader) {
        return SplitData.splitToTrain(csvDataReader.readByIndexes(dataHeaders.getAllHeaders()));
    }

    private static List<Record> prepareTest(IDataReader csvDataReader) {
        return SplitData.spliToTest(csvDataReader.readByIndexes(dataHeaders.getAttributes()));
    }
}

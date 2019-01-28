import model.DataHeaders;
import model.Record;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CSVManager implements DataManagerContract {
    private static final float TRAIN_SPLIT_RATIO = 0.6f;
    private DataHeaders dataHeaders = new DataHeaders("9", Arrays.asList("0", "1", "2", "3", "4", "5", "6", "7", "8"));
    private List<CSVRecord> csvRecords = new ArrayList<>();

    @Override
    public void read(String path) {
        try {
            CSVReader csvReader = new CSVReader(path);
            csvRecords = csvReader.getAllRecords();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public List<Record> readByIndexes(String path, List<String> indexes) {
        List<Record> records = new ArrayList<>();
        try {
            CSVReader csvReader = new CSVReader(path);
            records = csvReader.getRecordsByIndexes(indexes);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return records;
    }

    @Override
    public void saveTrainData(String path) {
        if (!csvRecords.isEmpty()) {
            List<CSVRecord> trainRecords = csvRecords.subList(0, getAmountOfTrainData());
            try {
                CSVWriter csvTrainWriter = new CSVWriter(path, dataHeaders.getAllHeaders());
                csvTrainWriter.saveRecordsByIndexes(trainRecords);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void saveTestData(String path) {
        if (!csvRecords.isEmpty()) {
            List<CSVRecord> testRecords = csvRecords.subList(getAmountOfTrainData(), csvRecords.size());
            try {
                CSVWriter csvTestWriter = new CSVWriter(path, dataHeaders.getAttributes());
                csvTestWriter.saveRecordsByIndexes(testRecords);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private int getAmountOfTrainData() {
        int amountOfRecords = csvRecords.size();
        return Float.valueOf(TRAIN_SPLIT_RATIO * amountOfRecords).intValue();
    }
}

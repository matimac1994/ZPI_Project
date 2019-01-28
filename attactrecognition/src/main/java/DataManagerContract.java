import model.Record;

import java.util.List;

public interface DataManagerContract {
    void read(String path);

    List<Record> readByIndexes(String path, List<String> indexes);

    void saveTrainData(String path);

    void saveTestData(String path);
}

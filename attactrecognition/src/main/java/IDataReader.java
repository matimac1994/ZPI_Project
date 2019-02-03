import model.Record;

import java.util.List;

public interface IDataReader {

    List<Record> readAll();

    List<Record> readByIndexes(List<String> indexes);
}

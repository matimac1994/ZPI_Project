import model.Record;

import java.io.IOException;
import java.util.List;

public interface IDataWriter {

    void write(List<Record> records) throws IOException;
}

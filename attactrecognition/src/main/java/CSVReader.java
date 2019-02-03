import model.Record;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

class CSVReader implements IDataReader {

    private List<CSVRecord> allRecords;

    CSVReader(String path) throws IOException {
        Reader reader = Files.newBufferedReader(Paths.get(path));
        CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader()
                .withIgnoreHeaderCase()
                .withTrim());
        allRecords = csvParser.getRecords();
    }

    @Override
    public List<Record> readAll() {
        return allRecords.stream().map(record ->
                new Record(new ArrayList<String>(record.toMap().values()))).collect(Collectors.toList()
        );
    }

    @Override
    public List<Record> readByIndexes(List<String> indexes) {
        List<Record> records = new ArrayList<>();
        for (CSVRecord csvRecord : allRecords) {
            records.add(new Record(indexes.stream().map(csvRecord::get).collect(Collectors.toList())));
        }
        return records;
    }
}

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

class CSVReader {

    private CSVParser csvParser;

    CSVReader(String path) throws IOException {
        Reader reader = Files.newBufferedReader(Paths.get(path));
        this.csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader()
                .withIgnoreHeaderCase()
                .withTrim());
    }

    List<CSVRecord> getAllRecords() throws IOException {
        return csvParser.getRecords();
    }

    List<Record> getRecordsByIndexes(List<String> indexes) throws IOException {
        List<Record> records = new ArrayList<>();
        for (CSVRecord csvRecord : getAllRecords()) {
            records.add(new Record(indexes.stream().map(csvRecord::get).collect(Collectors.toList())));
        }
        return records;
    }
}

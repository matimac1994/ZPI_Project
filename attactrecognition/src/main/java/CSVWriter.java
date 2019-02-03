import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

class CSVWriter {

    private CSVPrinter csvPrinter;
    private List<String> headers;

    CSVWriter(String path, List<String> headers) throws IOException {
        BufferedWriter writer = Files.newBufferedWriter(Paths.get(path));
        csvPrinter = new CSVPrinter(writer, CSVFormat.DEFAULT.withHeader(headers.toArray(new String[0])));
        this.headers = headers;
    }

    void saveRecordsByIndexes(List<CSVRecord> records) throws IOException {
        for (CSVRecord csvRecord : records) {
            csvPrinter.printRecord(headers.stream().map(csvRecord::get).collect(Collectors.toList()));
        }
        csvPrinter.flush();
    }
}

import model.Record;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

class CSVWriter implements IDataWriter {
    private CSVPrinter csvPrinter;

    CSVWriter(String path, List<String> headers) throws IOException {
        BufferedWriter writer = Files.newBufferedWriter(Paths.get(path));
        csvPrinter = new CSVPrinter(writer, CSVFormat.DEFAULT.withHeader(headers.toArray(new String[0])).withIgnoreHeaderCase().withTrim());
    }

    @Override
    public void write(List<Record> records) throws IOException {
        for (Record record : records) {
            try {
                csvPrinter.printRecord(record.getValues());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        csvPrinter.flush();
    }
}

package model;

import java.util.List;

public class Record {
    private List<String> values;

    public Record(List<String> values) {
        this.values = values;
    }

    public List<String> getValues() {
        return values;
    }
}

package model;

import java.util.ArrayList;
import java.util.List;

public class DataHeaders {
    private String label;
    private List<String> attributes;

    public DataHeaders(String label, List<String> attributes) {
        this.label = label;
        this.attributes = attributes;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public List<String> getAttributes() {
        return attributes;
    }

    public void setAttributes(List<String> attributes) {
        this.attributes = attributes;
    }

    public List<String> getAllHeaders() {
        List<String> allHeaders = new ArrayList<>(attributes);
        allHeaders.add(label);
        return allHeaders;
    }
}

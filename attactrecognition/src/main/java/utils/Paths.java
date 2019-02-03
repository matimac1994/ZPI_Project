package utils;

import java.util.Arrays;
import java.util.List;

public class Paths {
    //NET
    public static final List<String> NET_ATTRS = Arrays.asList("time", "duration", "source_computer", "source_port", "destination_computer", "destination_port", "protocol", "packet_count", "byte_count");
    public static final String NET_CLASS_NAME = "class";
    public static final String NET_DATA_CSV = "src/main/NETData.csv";
    public static final String NET_TRAIN_DATA_CSV = "./NETtrain.csv";
    public static final String NET_TEST_DATA_CSV = "./NETtest.csv";

    //PROC
    public static final List<String> PROC_ATTRS = Arrays.asList("time", "user@domain", "computer", "process_name", "start_end");
    public static final String PROC_CLASS_NAME = "class";
    public static final String PROC_DATA_CSV = "src/main/PROCData.csv";
    public static final String PROC_TRAIN_DATA_CSV = "./PROCtrain.csv";
    public static final String PROC_TEST_DATA_CSV = "./PROCtest.csv";
}

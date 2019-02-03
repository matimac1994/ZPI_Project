import model.Record;

import java.util.List;

class SplitData {
    private static final float TRAIN_SPLIT_RATIO = 0.55f;

    static List<Record> splitToTrain(List<Record> allRecords) {
        return allRecords.subList(0, getAmountOfTrainData(allRecords));
    }

    static List<Record> spliToTest(List<Record> allRecords) {
        return allRecords.subList(getAmountOfTrainData(allRecords), allRecords.size());
    }

    private static int getAmountOfTrainData(List<Record> allRecords) {
        return Float.valueOf(TRAIN_SPLIT_RATIO * allRecords.size()).intValue();
    }
}

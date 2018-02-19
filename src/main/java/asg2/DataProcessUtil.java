package asg2;

import java.io.*;
import java.util.*;

/**
 * Author: baojianfeng
 * Date: 2018-02-02
 * Description: this class is used for processing data from csv file
 */
public class DataProcessUtil {
    private String path;
    private Map<String, List<String>> attrMap;
    private Map<Integer, String> attrLabelMap; // store attribute position and name
    private Map<Integer, List<String>> attrValMap = new HashMap<>();
    private List<String> labels = new ArrayList<>();
    private List<String[]> instanceList = new ArrayList<>();

    public DataProcessUtil(String path) {
        this.path = path;
    }

    /**
     * process data from a file
     */
    public void processData() {
        FileInputStream fis = null;
        BufferedReader br = null;
        try {
            fis = new FileInputStream(path);
            br = new BufferedReader(new InputStreamReader(fis));
            String line = br.readLine();
            String[] attrArr = line.split(",");
            attrLabelMap = getAttrLabelMap(attrArr);

            line = br.readLine();
            // TODO test whether it can skip an empty line
            while (line != null) {
                if (!line.isEmpty()) {
                    String[] valArr = line.split(",");
                    attrValMap = getAttrValMap(valArr, attrValMap);
                    labels.add(valArr[valArr.length - 1]); // store the class label information at the last position
                    instanceList.add(valArr); // store instances
                }

                line = br.readLine();
            }

            attrMap = formAttrMap(attrLabelMap, attrValMap);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * extract attribute information from a string
     * @param attrArr input string array
     * @return attribute label map
     */
    private Map<Integer, String> getAttrLabelMap(String[] attrArr) {
        Map<Integer, String> map = new HashMap<>();
        // why attrArr.length - 1? aims to exclude 'Class' label
        for (int i = 0; i < attrArr.length - 1; i++)
            map.put(i, attrArr[i]);

        return map;
    }

    /**
     * extract attribute values from a string
     * @param valArr input string array
     * @param map input map
     * @return attribute values map
     */
    private Map<Integer, List<String>> getAttrValMap(String[] valArr, Map<Integer, List<String>> map) {
        // why attrArr.length - 1? aims to exclude 'Class' label
        for (int i = 0; i < valArr.length - 1; i++) {
            List<String> valueList;
            if (!map.containsKey(i))
                valueList = new ArrayList<>();
            else
                valueList = map.get(i);

            valueList.add(valArr[i]);
            map.put(i, valueList);
        }

        return map;
    }

    /**
     * compose attribute map using attribute label map and attribute value map
     * @param attrLabelMap attribute label map
     * @param attrValMap attribute value map
     * @return attribute map
     */
    private Map<String, List<String>> formAttrMap(Map<Integer, String> attrLabelMap, Map<Integer, List<String>> attrValMap) {
        Map<String, List<String>> attrMap = new HashMap<>();
        Set<Integer> keySet = attrLabelMap.keySet();
        for (int key : keySet) {
            if (attrValMap.containsKey(key))
                attrMap.put(attrLabelMap.get(key), attrValMap.get(key));
        }

        return attrMap;
    }

    public Map<String, List<String>> getAttrMap() {
        return attrMap;
    }

    public List<String> getLabels() {
        return labels;
    }

    public Map<Integer, String> getAttrLabelMap() {
        return attrLabelMap;
    }

    public List<String[]> getInstanceList() {
        return instanceList;
    }
}

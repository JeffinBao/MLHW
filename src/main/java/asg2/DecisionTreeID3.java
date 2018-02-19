package asg2;

import java.util.*;

/**
 * Author: baojianfeng
 * Date: 2018-02-02
 * Description: implementing of decision tree using ID3 algorithm
 */
public class DecisionTreeID3 {
    private String trainDsPath;
    private String validationDsPath;
    private String testDsPath;
    private double pruneFactor;
    private TreeNode root;
    private int nodeId = 0; // label a node with unique node id
    private int leafNodeCount = 0;

    /**
     * constructor
     * @param trainDsPath the complete path of training data set
     * @param validationDsPath the complete path of validation data set
     * @param testDsPath the complete path of test data set
     * @param pruneFactor pruning factor
     */
    public DecisionTreeID3(String trainDsPath, String validationDsPath,
                           String testDsPath, double pruneFactor) {
        this.trainDsPath = trainDsPath;
        this.validationDsPath = validationDsPath;
        this.testDsPath = testDsPath;
        this.pruneFactor = pruneFactor;

        initialization();
    }

    /**
     * get training data set file path
     * @return training data set file path
     */
    public String getTrainDsPath() {
        return trainDsPath;
    }

    /**
     * get validation data set file path
     * @return validation data set file path
     */
    public String getValidationDsPath() {
        return validationDsPath;
    }

    /**
     * get test data set file path
     * @return test data set file path
     */
    public String getTestDsPath() {
        return testDsPath;
    }

    /**
     * get the prune factor
     * @return prune factor
     */
    public double getPruneFactor() {
        return pruneFactor;
    }

    /**
     * initialise
     */
    private void initialization() {
        DataProcessUtil dataUtil = new DataProcessUtil(trainDsPath);
        dataUtil.processData();
    }

    /**
     * public method: construct a tree
     * @param attrMap attributes map: key is attribute name , value is attribute values
     * @param labels class label array
     */
    public void constructTree(Map<String, List<String>> attrMap, List<String> labels) {
        root = constructTree(attrMap, labels, 0);
    }

    /**
     * prune a decision tree according to given pruning factor
     * @param factor factor
     * @return pruned root node
     */
    public TreeNode pruneTree(double factor) {
        int pruneNodeCount = (int) (factor * nodeId);
        TreeNode copiedNode = copyTree(root);
        pruneTree(copiedNode, pruneNodeCount);

        return copiedNode;
    }

    /**
     * prune a bunch of nodes in a tree
     * @param node root node
     * @param pruneNodeCount the number of nodes to be pruned
     */
    private void pruneTree(TreeNode node, int pruneNodeCount) {
        List<Integer> pruneNodesIdList = new ArrayList<>();
        for (int i = 0; i < pruneNodeCount; i++) {
            int index = (int) (nodeId * Math.random());
            TreeNode curNode = findNode(node, index);

            // make sure the node has not been pruned yet and it is not the leaf node
            while (pruneNodesIdList.contains(index) || (curNode != null && curNode.classLabel != null)) {
                index = (int) (nodeId * Math.random());
                curNode = findNode(node, index);
            }

            if (curNode != null) {
                curNode.left = null;
                curNode.right = null;
                int[] labelZeroOne = new int[2];
                labelZeroOne[0] = curNode.labelZeroCount;
                labelZeroOne[1] = curNode.labelOneCount;
                curNode.classLabel = getClassLabel(labelZeroOne);
                curNode.attribute = null; // current node becomes leaf node
            }
            pruneNodesIdList.add(index);
        }

    }

    /**
     * count the number of all nodes and leaf nodes
     * @param node root node
     * @return arr[0]: the number of all nodes, arr[1]: the number of leaf nodes
     */
    public int[] countNodes(TreeNode node) {
        int[] count = new int[2];
        if (node == null)
            return count;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(node);
        int countAll = 0, countLeaf = 0;
        while (!queue.isEmpty()) {
            TreeNode curNode = queue.poll();
            countAll++;
            if (curNode.classLabel != null)
                countLeaf++;

            if (curNode.left != null)
                queue.add(curNode.left);
            if (curNode.right != null)
                queue.add(curNode.right);
        }

        count[0] = countAll;
        count[1] = countLeaf;
        return count;
    }

    /**
     * find a node according to its id
     * @param node root node
     * @param id id
     * @return node with specific id
     */
    private TreeNode findNode(TreeNode node, int id) {
        if (node == null)
            return null;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(node);
        while (!queue.isEmpty()) {
            TreeNode curNode = queue.poll();
            if (curNode.id == id)
                return curNode;

            if (curNode.left != null)
                queue.add(curNode.left);
            if (curNode.right != null)
                queue.add(curNode.right);
        }

        return null;
    }

    /**
     * copy the current tree
     * @param node root node
     * @return a copied tree
     */
    public TreeNode copyTree(TreeNode node) {
        if (node == null)
            return null;

        TreeNode copiedNode = new TreeNode(node.id, node.height, node.attribute, node.classLabel, node.labelZeroCount, node.labelOneCount);
        copiedNode.left = copyTree(node.left);
        copiedNode.right = copyTree(node.right);

        return copiedNode;
    }

    /**
     * public method: print tree
     */
    public void printTree() {
        printTree(root);
    }

    /**
     * calculate accuracy
     * @param instanceList instance list
     * @param attrPosMap attribute position --> name map
     * @param rootNode root node
     * @param totalSize data set size
     * @return accuracy
     */
    public double calAccuracy(List<String[]> instanceList, Map<Integer, String> attrPosMap, TreeNode rootNode, int totalSize) {
        return getCorrectCount(instanceList, attrPosMap, rootNode) / totalSize;
    }

    /**
     * compute how many correct predictions are mode by the model
     * @param instanceList instance list
     * @param attrPosMap attribute position --> name map
     * @param rootNode root node
     * @return number of correct predictions
     */
    private double getCorrectCount(List<String[]> instanceList, Map<Integer, String> attrPosMap, TreeNode rootNode) {
        double correctCount = 0.0;
        for (String[] strings : instanceList) {
            if (checkPrediction(strings, rootNode, attrPosMap))
                correctCount++;
        }

        return correctCount;
    }

    /**
     * check whether the decision tree model predict the correct class label
     * @param instance instance
     * @param node node
     * @param attrPosMap attribute position --> name map
     * @return true if prediction is equal to class label, otherwise return false
     */
    private boolean checkPrediction(String[] instance, TreeNode node, Map<Integer, String> attrPosMap) {
        if (node.classLabel != null)
            return instance[instance.length - 1].equals(node.classLabel);
        else {
            // TODO It is not efficient to find the attribute position in the data set
            int pos = getPosFromAttrVal(attrPosMap, node.attribute);
            if (pos == -1)
                return false;
            else {
                if (instance[pos].equals("0"))
                    return checkPrediction(instance, node.left, attrPosMap);
                else
                    return checkPrediction(instance, node.right, attrPosMap);
            }
        }
    }

    /**
     * find the position of an attribute in the file
     * @param attrPosMap attribute position and name map
     * @param attr attribute
     * @return the position
     */
    private int getPosFromAttrVal(Map<Integer, String> attrPosMap, String attr) {
        for (Map.Entry<Integer, String> entry : attrPosMap.entrySet()) {
            if (attr.equals(entry.getValue()))
                return entry.getKey();
        }

        return -1;
    }

    /**
     * private method: print tree
     * @param node node
     */
    private void printTree(TreeNode node) {
        if (node == null)
            return;

        if (node.classLabel != null) {
            System.out.println(node.classLabel);
        } else {
            System.out.println();
            if (node.left != null) {
                for (int i = 0; i < node.height; i++)
                    System.out.print("| ");
                System.out.print(node.attribute + "=0:");
                printTree(node.left);
            }
            if (node.right != null) {
                for (int i = 0; i < node.height; i++)
                    System.out.print("| ");
                System.out.print(node.attribute + "=1:");
                printTree(node.right);
            }
        }
    }

    /**
     * private method to construct a tree, need to test, especially when to stop building the tree
     * @param attrMap attributes map
     * @param labels class labels
     * @param height the height of a node
     * @return constructed node
     */
    private TreeNode constructTree(Map<String, List<String>> attrMap,
                                   List<String> labels, int height) {
        if (labels.size() == 0 || attrMap.size() == 0)
            return null;

        double entropyParent = calParentEntropy(labels);
        int[] labelZeroOne = calZeroOneLabels(labels);
        if (entropyParent == 0.0) {
            leafNodeCount++;
            return new TreeNode(nodeId++, height, null, getClassLabel(labelZeroOne), labelZeroOne[0], labelZeroOne[1]);
        } else {
            String splitAttr = getSplitAttr(attrMap, labels, entropyParent);
            if (splitAttr.isEmpty()) {
                leafNodeCount++;
                return new TreeNode(nodeId++, height, null, getClassLabel(labelZeroOne), labelZeroOne[0], labelZeroOne[1]);
            } else {
                TreeNode node = new TreeNode(nodeId++, height, splitAttr, null, labelZeroOne[0], labelZeroOne[1]);
                List<List<Integer>> leftRightIndexList = getChildLevelDataIndexList(attrMap, splitAttr);
                List<Integer> leftChildIndex = leftRightIndexList.get(0);
                List<Integer> rightChildIndex = leftRightIndexList.get(1);
                attrMap.remove(splitAttr); // remove selected split attribute
                node.left = constructTree(purifyMap(attrMap, leftChildIndex), purifyLabels(leftChildIndex, labels), height + 1);
                node.right = constructTree(purifyMap(attrMap, rightChildIndex), purifyLabels(rightChildIndex, labels), height + 1);
                return node;
            }
        }
    }

    /**
     * get split attribute
     * @param map attribute map
     * @param labels class labels
     * @param entropyParent entropy of current data set
     * @return split attribute
     */
    private String getSplitAttr(Map<String, List<String>> map, List<String> labels, double entropyParent) {
        String splitAttr = "";
        Set<String> keySet = map.keySet();
        double infoGainMax = 0.0;
        for (String key : keySet) {
            List<String> attrValList = map.get(key);
            double entropyChild = calEntropy(attrValList, labels);
            double infoGain = entropyParent - entropyChild; // there maybe several attributes whose information gain are the same, just pick one attribute as the split attribute
            if (infoGain > infoGainMax) {
                infoGainMax = infoGain;
                splitAttr = key;
            }
        }

        return splitAttr;
    }

    /**
     * get class label sign
     * @param labelZeroOne class label array, labelZeroOne[0] stores the number of class 0, labelZeroOne[1] stores the number of class 1
     * @return "0" if the number of label 0 is greater than or equal to the number of label 1, otherwise return "1"
     */
    private String getClassLabel(int[] labelZeroOne) {
        if (labelZeroOne[0] >= labelZeroOne[1])
            return "0";
        else
            return "1";
    }

    /**
     * split the attribute value into two groups: 0 and 1
     * @param map map
     * @param splitAttr split attributes
     * @return index lists: left index list contains all 0, right index list contains all 1
     */
    private List<List<Integer>> getChildLevelDataIndexList(Map<String, List<String>> map,
                                                    String splitAttr) {
        List<List<Integer>> leftRightIndexList = new ArrayList<>(); // store left and right index list
        List<String> originalList = map.get(splitAttr); // original list, waiting to be split
        List<Integer> leftIndex = new ArrayList<>();
        List<Integer> rightIndex = new ArrayList<>();
        for (int i = 0; i < originalList.size(); i++) {
            if (originalList.get(i).equals("0"))
                leftIndex.add(i);
            else
                rightIndex.add(i);
        }

        leftRightIndexList.add(leftIndex);
        leftRightIndexList.add(rightIndex);

        return leftRightIndexList;

    }

    /**
     * eliminate attribute and its values according to previously split attribute
     * @param map map
     * @param listIndex child level needed index list
     * @return new map
     */
    private Map<String, List<String>> purifyMap(Map<String, List<String>> map, List<Integer> listIndex) {
        Map<String, List<String>> newMap = new HashMap<>();

        Set<String> keySet = map.keySet();
        for (String key : keySet) {
            List<String> originList = map.get(key);
            List<String> newList = new ArrayList<>();
            for (Integer val : listIndex)
                newList.add(originList.get(val));

            newMap.put(key, newList);
        }

        return newMap;
    }

    /**
     * get the child level class label's
     * @param listIndex child level needed index list
     * @param labels parent level class labels
     * @return child level class labels
     */
    private List<String> purifyLabels(List<Integer> listIndex, List<String> labels) {
        List<String> newLabelsList = new ArrayList<>();
        for (Integer val : listIndex)
            newLabelsList.add(labels.get(val));

        return newLabelsList;
    }


    /**
     * calculate the initial entropy of all data
     * @param labels label class list
     * @return entropy of all data
     */
    private double calParentEntropy(List<String> labels) {
        double length = labels.size();
        if (length == 0.0)
            return 0.0;

        double countZero = 0.0;
        for (String value : labels) {
            if (value.equals("0"))
                countZero++;
        }

        double countOne = length - countZero;
        double zeroP = countZero / length;
        double oneP = countOne / length;
        double logZero = log2(zeroP);
        double logOne = log2(oneP);
        double v1 = zeroP * logZero;
        double v2 = oneP * logOne;
        double result = -v1 - v2;

        return result;
    }

    /**
     * calculate the entropy if we choose a specific attribute to split
     * @param attributes attribute values
     * @param labels label class list
     * @return entropy after splitting with a specific attribute
     */
    private double calEntropy(List<String> attributes, List<String> labels) {
        if (attributes.size() == 0)
            return 0.0;

        double countZero = 0.0, countZeroZero = 0.0, countZeroOne = 0.0;
        double countOne = 0.0, countOneZero = 0.0, countOneOne = 0.0;
        double length = attributes.size();
        for (int i = 0; i < length; i++) {
            if (attributes.get(i).equals("0")) {
                countZero++;
                if (labels.get(i).equals("0"))
                    countZeroZero++;
                else
                    countZeroOne++;
            } else {
                countOne++;
                if (labels.get(i).equals("0"))
                    countOneZero++;
                else
                    countOneOne++;
            }
        }

        // calculate possibilities
        // TODO check whether the following values are zero or not
        double zeroP = countZero / length;
        double zeroZeroP, zeroOneP;
        if (countZero == 0.0) {
            zeroZeroP = 0.0;
            zeroOneP = 0.0;
        } else {
            zeroZeroP = countZeroZero / countZero;
            zeroOneP = countZeroOne / countZero;
        }

        double oneP = countOne / length;
        double oneZeroP, oneOneP;
        if (countOne == 0.0) {
            oneZeroP = 0.0;
            oneOneP = 0.0;
        } else {
            oneZeroP = countOneZero / countOne;
            oneOneP = countOneOne / countOne;
        }


        // calculate entropy
        return zeroP * (-zeroZeroP * log2(zeroZeroP) - zeroOneP * log2(zeroOneP))
                + oneP * (-oneZeroP * log2(oneZeroP) - oneOneP * log2(oneOneP));
    }

    /**
     * calculate how many zero and one labels in the class label array
     * @param labels class label array
     * @return int[0] = size of zero class, int[1] = size of one class
     */
    private int[] calZeroOneLabels(List<String> labels) {
        int[] result = new int[2];
        int countZero = 0;
        for (String val : labels) {
            if (val.equals("0"))
                countZero++;
        }

        result[0] = countZero;
        result[1] = labels.size() - countZero;
        return result;
    }

    /**
     * calculate log, base is 2
     * @param a input value
     * @return log value
     */
    private double log2(double a) {
        if (a == 0.0) // if a == 0.0, log2(a) = 0.0
            return 0.0;

        return Math.log(a) / Math.log(2);
    }

    private class TreeNode {
        int id;
        int height; // store the height of a node
        String attribute;
        int labelZeroCount; // store how many class zero at a node
        int labelOneCount; // store how many class one at a node
        String classLabel;
        TreeNode left; // means attribute = 0
        TreeNode right; // means attribute = 1

        TreeNode(int id, int height, String attribute, String classLabel, int labelZeroCount, int labelOneCount) {
            this.id = id;
            this.height = height;
            this.attribute = attribute;
            this.classLabel = classLabel;
            this.labelZeroCount = labelZeroCount;
            this.labelOneCount = labelOneCount;
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("Please input the file name of training data set: ");
        String trainDsPath = sc.nextLine();
        System.out.println("Please input the file name of validation data set: ");
        String validationDsPath = sc.nextLine();
        System.out.println("Please input the file name of test data set: ");
        String testDsPath = sc.nextLine();
        System.out.println("Please input the prune factor(less than 1.0): ");
        double pruneFactor = Double.valueOf(sc.nextLine());

        DecisionTreeID3 dtID3 = new DecisionTreeID3(trainDsPath,
                validationDsPath,
                testDsPath, pruneFactor);
        // process data
        DataProcessUtil dpTrain = new DataProcessUtil(dtID3.getTrainDsPath());
        dpTrain.processData();
        DataProcessUtil dpValidation = new DataProcessUtil(dtID3.getValidationDsPath());
        dpValidation.processData();
        DataProcessUtil dpTest = new DataProcessUtil(dtID3.getTestDsPath());
        dpTest.processData();

        // construct tree
        dtID3.constructTree(dpTrain.getAttrMap(), dpTrain.getLabels());
        dtID3.printTree();
        System.out.println("Pre-Pruned Accuracy");
        System.out.println("---------------------------------------------------------------------");
        System.out.println("Number of training instances = " + dpTrain.getLabels().size());
        System.out.println("Number of training attributes = " + dpTrain.getAttrLabelMap().size());
        System.out.println("Total number of nodes in the tree = " + dtID3.nodeId);
        System.out.println("Total number of leaf nodes in the tree = " + dtID3.leafNodeCount);
        System.out.println("Accuracy of the model on the training data set = " +
                dtID3.calAccuracy(dpTrain.getInstanceList(), dpTrain.getAttrLabelMap(), dtID3.root, dpTrain.getLabels().size()));
        System.out.println();
        System.out.println("Number of validation instances = " + dpValidation.getLabels().size());
        System.out.println("Number of validation attributes = " + dpValidation.getAttrLabelMap().size());
        System.out.println("Accuracy of the model on the validation data set before pruning = " +
                dtID3.calAccuracy(dpValidation.getInstanceList(), dpValidation.getAttrLabelMap(), dtID3.root, dpValidation.getLabels().size()));
        System.out.println();
        System.out.println("Number of testing instances = " + dpTest.getLabels().size());
        System.out.println("Number of testing attributes = " + dpTest.getAttrLabelMap().size());
        System.out.println("Accuracy of the model on the testing data set before pruning = " +
                dtID3.calAccuracy(dpTest.getInstanceList(), dpTest.getAttrLabelMap(), dtID3.root, dpTest.getLabels().size()));

        System.out.println();

        double TrainAccuracy = dtID3.calAccuracy(dpValidation.getInstanceList(), dpValidation.getAttrLabelMap(), dtID3.root, dpValidation.getLabels().size());
        TreeNode prunedTree = null;
        double prunedAccuracy = 0;
        System.out.println("Calculating an better pruned tree...");
        int i = 0;
        //At least improve 0.02 in accuracy.
        while(prunedAccuracy <= TrainAccuracy + 0.02){

            prunedTree = dtID3.pruneTree(dtID3.getPruneFactor());
            prunedAccuracy = dtID3.calAccuracy(dpValidation.getInstanceList(), dpValidation.getAttrLabelMap(), prunedTree, dpValidation.getLabels().size());
//            System.out.print("--old--"+TrainAccuracy+" ---new---"+prunedAccuracy);
            i++;

            if(prunedAccuracy > TrainAccuracy && i >= 10000){
                break;
            }
        }
        System.out.println("After "+ i +" loops, reach the pruned tree with " + (prunedAccuracy - TrainAccuracy) + " accuracy improvement.");
        dtID3.printTree(prunedTree);

        System.out.println("Post-Pruned Accuracy");
        System.out.println("---------------------------------------------------------------------");
        System.out.println("Number of training instances = " + dpTrain.getLabels().size());
        System.out.println("Number of training attributes = " + dpTrain.getAttrLabelMap().size());
        int[] nodeCount = dtID3.countNodes(prunedTree);
        System.out.println("Total number of nodes in the tree = " + nodeCount[0]);
        System.out.println("Total number of leaf nodes in the tree = " + nodeCount[1]);
        System.out.println("Accuracy of the model on the training data set = " +
                dtID3.calAccuracy(dpTrain.getInstanceList(), dpTrain.getAttrLabelMap(), prunedTree, dpTrain.getLabels().size()));
        System.out.println();
        System.out.println("Number of validation instances = " + dpValidation.getLabels().size());
        System.out.println("Number of validation attributes = " + dpValidation.getAttrLabelMap().size());
        System.out.println("Accuracy of the model on the validation data set after pruning = " +
                dtID3.calAccuracy(dpValidation.getInstanceList(), dpValidation.getAttrLabelMap(), prunedTree, dpValidation.getLabels().size()));
        System.out.println();
        System.out.println("Number of testing instances = " + dpTest.getLabels().size());
        System.out.println("Number of testing attributes = " + dpTest.getAttrLabelMap().size());
        System.out.println("Accuracy of the model on the testing data set after pruning = " +
                dtID3.calAccuracy(dpTest.getInstanceList(), dpTest.getAttrLabelMap(), prunedTree, dpTest.getLabels().size()));
    }
}

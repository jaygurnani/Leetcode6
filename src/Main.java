import javax.swing.tree.TreeNode;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Main {
    public static void main(String[] args) {
        //System.out.println("Hello, World!");
        //String[] input = new String[]{"eat","tea","tan","ate","nat","bat"};
        //var output = groupAnagrams(input);

        //int[] input = new int[]{0,1,0,3,12};
        //moveZeroes(input);
        //System.out.println(output);

        //int[] input = new int[]{7,1,5,3,6,4};
        //int output = maxProfit(input);

        //String input = "babad";
        //String output = longestPalindrome(input);

//        StringBuffer sb = new StringBuffer();
//        sb.append("This is an");
//        var output = expandBySpace(sb, 2, 8, 16);
//        System.out.println(output.toString());

//        String[] input = new String[]{"Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"};
//        List<String> output = fullJustify(input, 16);
//
//        for(String s: output) {
//            System.out.println(s);
//        }

//        int[] input = new int[] {3,1,4,3,2,2,4};
//        long output = countGood(input, 2);

//        int[][] input = new int[][] {{11, 9}, {9, 4},{1,5}, {4, 1}, };
//        String output = printPathWay(input);

        //String output = removeKdigits("33526221184202197273", 19);
        //int[] input = new int[]{99,99};
        //String output = String.valueOf(containsNearbyDuplicate(input, 2));
        //int[] nums = new int[]{0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1};
        //int output = longestOnes(nums, 3);
        //int output = lengthOfLongestSubstringTwoDistinct("1123123aaaaaa990606");

        String output = longestPalindrome("1123123aaaaaa990606");
        System.out.println(output);

    }

    public static List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> result = new ArrayList<>();
        Map<String, List<String>> mapping = new HashMap<>();

        for(String s: strs) {
            var charArray = s.toCharArray();
            Arrays.sort(charArray);
            var sorted = Arrays.toString(charArray);

            if(mapping.containsKey(sorted)) {
                var map = mapping.get(sorted);
                map.add(s);
            } else {
                var newList = new ArrayList<String>();
                newList.add(s);
                mapping.put(sorted, newList);
            }
        }

        for(String s: mapping.keySet()) {
            result.add(mapping.get(s));
        }

        return result;
    }

    public static void moveZeroes(int[] nums) {
        int currentCount = 0;
        for(int n = 0; n < nums.length; n++) {
            if (nums[n] != 0) {
                nums[currentCount] = nums[n];
                currentCount++;
            }
        }
        while (currentCount < nums.length){
            nums[currentCount] = 0;
            currentCount++;
        }

        System.out.println(Arrays.toString(nums));

    }

    public static int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<>();
        int largestWindow = 0;
        int left = 0;

        var charArray = s.toCharArray();
        for(int right = 0; right < charArray.length; right++) {
            var item =charArray[right];
            if (!set.contains(item)) {
                set.add(item);
                largestWindow = Math.max(largestWindow, right - left + 1);
            } else {
                while(set.contains(item)) {
                    set.remove(charArray[left]);
                    left++;
                }
                set.add(item);
            }
        }

        return largestWindow;
    }

    public static int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for(int n: nums){
            map.put(n, map.getOrDefault(n,0)+1);
        }

        PriorityQueue<Map.Entry<Integer, Integer>>  maxHeap =
                new PriorityQueue<>((a,b) -> b.getValue()-a.getValue());
        maxHeap.addAll(map.entrySet());

        List<Integer> res = new ArrayList<>();
        while(res.size() < k){
            Map.Entry<Integer, Integer> entry = maxHeap.poll();
            res.add(entry.getKey());
        }
        return res.stream().mapToInt(i-> i).toArray();
    }

    public static List<List<Integer>> kSmallestPairs1(int[] nums1, int[] nums2, int k) {
        PriorityQueue<int[]> minHeap = new PriorityQueue<>(Comparator.comparingInt(a -> (a[0] + a[1])));
        List<List<Integer>> result = new ArrayList<>();

        for(int i = 0; i < nums1.length; i++) {
            for(int j = 0; j < nums2.length; j++) {
                var item1 = nums1[i];
                var item2 = nums2[j];

                int[] input = new int[2];
                input[0] = item1;
                input[1] = item2;
                minHeap.add(input);
            }
        }

        for (int i = 0; i < k; i++) {
            int[] currentItem = minHeap.poll();
            List<Integer> list = Arrays.stream(currentItem).boxed().toList();
            result.add(list);
        }

        return result;
    }

    public static List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        PriorityQueue<int[]> minHeap = new PriorityQueue<>((a, b) -> a[0] - b[0]);
        List<List<Integer>> result = new ArrayList<>();

        for (int x : nums1) {
            minHeap.offer(new int[]{x + nums2[0], 0});
        }

        // Pop the k smallest pairs from the priority queue
        while (k > 0 && !minHeap.isEmpty()) {
            int[] pair = minHeap.poll();
            int sum = pair[0]; // Get the smallest sum
            int pos = pair[1]; // Get the index of the second element in nums2

            List<Integer> currentPair = new ArrayList<>();
            currentPair.add(sum - nums2[pos]);
            currentPair.add(nums2[pos]);
            result.add(currentPair); // Add the pair to the result list

            // If there are more elements in nums2, push the next pair into the priority queue
            if (pos + 1 < nums2.length) {
                minHeap.offer(new int[]{sum - nums2[pos] + nums2[pos + 1], pos + 1});
            }

            k--; // Decrement k
        }


        return result;
    }

    public static int maxSubArray(int[] nums) {
        int currentMax = nums[0];
        int currentSubArray = nums[0];

        for (int i = 1; i < nums.length; i++) {
            int num = nums[i];
            currentSubArray = Math.max(currentSubArray + num, num);
            currentMax = Math.max(currentMax, currentSubArray);
        }

        return currentMax;
    }

    public static int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);

        for(int i = 0; i < nums.length; i++) {
            for(int j = 0; j < i; j++) {
                if(nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }

        int longest = 0;
        for (int num : dp) {
            longest = Math.max(num, longest);
        }

        return longest;
    }

    public static int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
        int result = 0;

        for(int i = 0; i < nums.length; i++) {
            pq.add(nums[i]);
        }

        for(int i = 0; i < k; i++){
            result = pq.poll();
        }

        return result;
    }

    public static int maxProfit(int[] prices) {
        int n = prices.length;
        int[] maxProfit = new int[n];
        int maxProfitTotal = 0;

        for(int i = 0; i < prices.length; i++) {
            int j = i+1;
            while (j < n) {
                if (prices[j] > prices[i]) {
                    maxProfit[i] = Math.max(prices[j] - prices[i], maxProfit[i]);
                }
                j++;
            }
        }

        for(int i = 0; i < n; i++) {
            maxProfitTotal = Math.max(maxProfit[i], maxProfitTotal);
        }

        return maxProfitTotal;
    }

    public int maxProfit2(int[] prices) {
        int minprice = Integer.MAX_VALUE;
        int maxprofit = 0;
        for(int i = 0; i < prices.length; i++) {
            if (prices[i] < minprice) {
                minprice = prices[i];
            }
            else if (prices[i] - minprice > maxprofit) {
                maxprofit = prices[i] - minprice;
            }
        }
        return maxprofit;
    }


    public static int[] dailyTemperatures(int[] temperatures) {
        int[] result = new int[temperatures.length];
        Stack<Integer> stack = new Stack<>();

        for (int i = 0; i < temperatures.length; i++) {
            int currentTemp = temperatures[i];
            while (!stack.isEmpty() && temperatures[stack.peek()] < currentTemp) {
                int prevDay = stack.pop();
                result[prevDay] = i - prevDay;
            }
            stack.push(i);
        }
        return result;
    }

    public static boolean isPalindrome(String s) {
        for(int i = 0, j = s.length()-1; j < s.length(); i++, j--) {
            if (s.charAt(i) != s.charAt(j)) {
                return false;
            }
        }
        return true;
    }

    public static List<String> fullJustify(String[] words, int maxWidth) {
        List<String> result = new ArrayList<String>();
        int lineSpaceCount = 0;
        int lineCharCount = 0;
        StringBuffer sb = new StringBuffer();

        for(String w: words) {
            if (sb.length() + w.length() > maxWidth+1) {
                StringBuffer expanded = expandBySpace(sb.delete(sb.length()-1, sb.length()), lineSpaceCount-1, lineCharCount, maxWidth);
                result.add(expanded.toString());
                sb = new StringBuffer();
                sb.append(w);
                sb.append(' ');

                lineCharCount = w.length();
                lineSpaceCount = 1;

            } else if (sb.length() + w.length() == maxWidth) {
                sb.append(w);
                result.add(sb.toString());
                sb = new StringBuffer();
                lineCharCount = 0;
                lineSpaceCount = 0;
            } else {
                sb.append(w);
                sb.append(' ');

                lineCharCount = lineCharCount + w.length();
                lineSpaceCount = lineSpaceCount + 1;
            }
        }
        if (!sb.isEmpty()) {
            while(sb.length() != maxWidth){
                sb.append(' ');
            }
            result.add(sb.toString());
        }


        return result;
    }

    public static StringBuffer expandBySpace(StringBuffer input, int spaceCount, int charCount, int maxWidth) {
        StringBuffer sb = new StringBuffer();
        int remainingToFill = maxWidth - charCount;
        if (spaceCount == 0) {
            while(input.length() != maxWidth){
                input.append(' ');
            }

            return input;
        }

        int toExpand = remainingToFill / spaceCount;

        for(int i = 0; i < input.length(); i++) {
            if (input.charAt(i) == ' ') {
                for(int j = 0; j < toExpand; j++) {
                    sb.append(' ');
                }
            } else {
                sb.append(input.charAt(i));
            }
        }

        if(sb.length() != maxWidth) {
            for(int i = 1; i < sb.length(); i++) {
                if (sb.charAt(i-1) == ' ' && sb.charAt(i) != ' ') {
                    sb.insert(i, ' ');
                    if (sb.length() == maxWidth) {
                        break;
                    }
                }
            }
        }

        return sb;
    }


    public static int minMeetingRooms(int[][] intervals) {
        if (intervals.length == 0) {
            return 0;
        }

        PriorityQueue<Integer> allocated = new PriorityQueue<>(intervals.length);

        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });

        allocated.add(intervals[0][1]);

        for(int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] >= allocated.peek()) {
                allocated.poll();
            }
            allocated.add(intervals[i][1]);
        }

        return allocated.size();
    }


    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        inorderTraversalResult(root, result);
        return result;
    }

    public void inorderTraversalResult(TreeNode node, List<Integer> result) {
        if (node == null) {
            return;
        }
        if (node.left != null) {
            inorderTraversalResult(node.left, result);
            //result.add(node.left.val);
        }
        result.add(node.val);
        if (node.right != null) {
            inorderTraversalResult(node.right, result);
            //result.add(node.right.val);
        }
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> toReturn = new ArrayList<>();
        HashMap<Integer, List<Integer>> map = new HashMap<>();
        HashMap<Integer, List<Integer>> result = levelOrderWithMap(root, 0, map);

        var keySet = map.keySet();
        var sorted = keySet.stream().sorted().toList();
        for(int k : sorted) {
            toReturn.add(result.get(k));
        }

        return toReturn;
    }

    public HashMap<Integer, List<Integer>> levelOrderWithMap(TreeNode root, int level, HashMap<Integer, List<Integer>> map) {
        if (root == null) {
            return map;
        }
        var current = map.computeIfAbsent(level, k -> new ArrayList<>());
        current.add(root.val);

        levelOrderWithMap(root.left, level + 1, map);
        levelOrderWithMap(root.right, level + 1, map);
        return map;
    }

    public static long countGood(int[] nums, int k) {
        long result = 0;
        HashMap<Integer, Integer> freq = new HashMap<>();
        int left = 0;
        long pairCount = 0;
        for (int right = 0; right < nums.length; right++) {
            int currentFreq = freq.getOrDefault(nums[right], 0);
            // If we get a one here, the pair count becomes 1
            pairCount = pairCount + currentFreq;
            currentFreq = currentFreq+1;
            freq.put(nums[right], currentFreq);

            while (pairCount >= k) {
                // We shortcut a quick way to calculate all the sub-arrays
                // Anything more added will just add the sub-array count
                result = result + (nums.length - right);
                int leftNum = nums[left];
                var leftFreq = freq.get(leftNum);
                leftFreq--;
                freq.put(leftNum, leftFreq);
                pairCount = pairCount - leftFreq;
                left++;
            }
        }

        return result;
    }

    public static String printPathWay(int[][] paths) {
        StringBuffer sb = new StringBuffer();
        // First we need to find the starting point
        Set<Integer> set = new HashSet<>();
        // Then we need to keep a map of source destinations
        HashMap<Integer, Integer> map = new HashMap<>();

        // Add all the sources
        for(int i = 0; i < paths.length; i++) {
            set.add(paths[i][0]);
            map.put(paths[i][0], paths[i][1]);
        }
        // Remove all destinations
        for(int i = 0; i < paths.length; i++) {
            set.remove(paths[i][1]);
        }

        List<Integer> list = new ArrayList<Integer>(set);
        int next = list.getFirst();

        while (map.containsKey(next)) {
            next = map.get(next);
            sb.append(next);
            sb.append("->");
        }


        return sb.substring(0, sb.length()-2);
    }

    public String destCity(List<List<String>> paths) {
        List<String> list = new ArrayList<>();

        for(int i = 0; i < paths.size(); i++) {
            list.add(paths.get(i).get(1));
        }

        for(int j = 0; j < paths.size(); j++) {
            list.remove(paths.get(j).get(0));
        }
        return list.getFirst();

    }

    // This approach won't work for very long string since we don't have enough space for int / long
    // See solution bottom from leetcode on how to solve this using a stack taken from
    // - https://leetcode.com/problems/remove-k-digits/editorial/
    public static String removeKdigits2(String num, int k) {
        long min = Long.MAX_VALUE;
        StringBuffer sb = new StringBuffer(num);

        if (num.length() == k) {
            return "0";
        }

        for(int i = 0; i < k; i++) {
            for(int j = 0; j < num.length(); j++) {
                char toAddBack = num.charAt(j);
                sb.deleteCharAt(j);
                String current = sb.toString();
                if (current.isEmpty()) {
                    return "0";
                }

                long currentValue = Long.parseLong(current);
                if (currentValue < min) {
                    min = currentValue;
                }
                sb.insert(j, toAddBack);
            }
            num = Long.toString(min);
            sb = new StringBuffer(num);
        }

        return Long.toString(min);
    }

    public static String removeKdigits(String num, int k) {
        Stack<Character> stack = new Stack<>();

        for (char digit : num.toCharArray()) {
            while (!stack.isEmpty() && k > 0 && stack.peek() > digit) {
                stack.pop();
                k--;
            }
            stack.push(digit);
        }

        // Remove remaining k digits from the end of the stack
        while (k > 0 && !stack.isEmpty()) {
            stack.pop();
            k--;
        }


        // Construct the resulting string from the stack
        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            sb.append(stack.pop());
        }
        sb.reverse(); // Reverse to get the correct order

        // Remove leading zeros
        while (sb.length() > 0 && sb.charAt(0) == '0') {
            sb.deleteCharAt(0);
        }

        // Handle edge case where result might be empty
        return sb.length() > 0 ? sb.toString() : "0";
    }

    public static int[] twoSum(int[] numbers, int target) {
        int[] result = new int[2];
        HashMap<Integer, Integer> map = new HashMap<>();

        for(int i = 0; i < numbers.length; i++) {
            map.put(numbers[i], i);
        }
        for(int j = 0; j < numbers.length; j++) {
            if (map.containsKey(target - numbers[j])) {
                result[0] = j+1;
                result[1] = map.get(target - numbers[j]) +1;;
                return result;
            }

        }

        return result;
    }

    public static boolean containsNearbyDuplicate(int[] nums, int k) {
        HashSet<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            if (set.contains(nums[i])) {
                return true;
            }
            set.add(nums[i]);
            if (set.size() > k) {
                set.remove(nums[i-k]);
            }
        }
        return false;
    }

    public int maxFrequency(int[] nums, int k) {
        return 0;
    }

    public static int longestOnes(int[] nums, int k) {
        int left = 0;
        int right = 0;

        while (right < nums.length) {
            if (nums[right] == 0) {
                k = k-1;
            }
            if (k < 0) {
                if (nums[left] == 0) {
                    k = k+1;
                }
                left++;
            }
            right++;
        }
        return right-left;

    }

    public static int lengthOfLongestSubstringTwoDistinct(String s) {
        char[] input = s.toCharArray();
        int left = 0;
        int right = 0;
        int max = Integer.MIN_VALUE;
        HashMap<Character, Integer> map = new HashMap<>();

        while(right < input.length) {
            // If we have seen this already, just continue
            if (map.containsKey(input[right])) {
                map.put(input[right], map.get(input[right]) + 1);
                right++;
            } else {
                // This is a new character
                while(map.size() >= 2) {
                    map.put(input[left], map.get(input[left]) - 1);
                    if (map.get(input[left]) == 0) {
                        map.remove(input[left]);
                    }
                    left++;
                }
                map.put(input[right], 1);
                right++;

            }
            max = Math.max(max, right-left);
        }

        return max;
    }

    public static String longestPalindromeBruteForce(String s) {
        if (s.length() <= 1) {
            return s;
        }
        int maxLength = 1;
        String maxStr = s.substring(0,1);
        char[] sCharArray = s.toCharArray();

        for(int i = 0; i < s.length(); i++) {
            for(int j = i + maxLength; j <= s.length(); j++) {
                if (j-i > maxLength && CheckPalindrone(s.substring(i, j))) {
                    maxLength = j-i;
                    maxStr = s.substring(i,j);
                }
            }
        }

        return maxStr;
    }

    public static boolean CheckPalindrone(String s) {
        int left = 0;
        int right = s.length() -1;
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }

        return true;
    }

    public static String longestPalindrome(String s) {
        if (s.length() <= 1) {
            return s;
        }

        String maxStr = s.substring(0,1);
        for(int i = 0; i < s.length()-1; i++) {
            String odd = expandFromCentre(s,i,i);
            String even = expandFromCentre(s,i,i+1);

            if (odd.length() > maxStr.length()) {
                maxStr = odd;
            }
            if (even.length() > maxStr.length()) {
                maxStr = even;
            }
        }
        return maxStr;
    }

    public static String expandFromCentre(String s, int left, int right) {
        while(left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return s.substring(left+1, right);
    }

    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        }

        if (Math.abs(height(root.left) - height(root.right)) < 2 && isBalanced(root.left) && isBalanced(root.right)) {
            return true;
        }
        return false;
    }

    public int height(TreeNode root) {
        if (root == null) {
            return -1;
        }

        return 1+ Math.max(height(root.left), height(root.right));
    }

    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }

        if (root.left == null) {
            return 1+ minDepth(root.right);
        }
        if (root.right == null) {
            return 1+minDepth(root.left);
        }

        int left = minDepth(root.left);
        int right = minDepth(root.right);

        return 1 + Math.min(left, right);
    }

    int totalSum = 0;
    public int sumNumbers(TreeNode root) {
        preOrder(root, 0);
        return totalSum;
    }

    public void preOrder(TreeNode root, int currNumber) {
        if (root != null) {
            currNumber = currNumber * 10 + root.val;
            if (root.left == null && root.right == null) {
                totalSum = totalSum + currNumber;
            }
            preOrder(root.left, currNumber);
            preOrder(root.right, currNumber);
        }
    }

    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> curr =  new ArrayList<Integer>();
        preOrder(root, curr);
        return curr;
    }

    public void preOrder(TreeNode root, List<Integer> curr) {
        if (root == null) {
            return;
        }
        curr.add(root.val);
        preOrder(root.left, curr);
        preOrder(root.right, curr);
    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
}
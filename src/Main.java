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

        String input = "babad";
        String output = longestPalindrome(input);
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

    public static String longestPalindrome(String s) {

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



}
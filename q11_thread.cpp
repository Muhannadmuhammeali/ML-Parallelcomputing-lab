#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <mutex>
using namespace std;
using namespace std::chrono;

long long totalSum = 0;
bool found = false;
int foundIndex = -1;
mutex mtx;

// Generate array
void generateArray(vector<int>& arr, int n) {
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;
}

// Thread function for sum
void partialSum(vector<int>& arr, int start, int end) {
    long long localSum = 0;
    for (int i = start; i < end; i++)
        localSum += arr[i];

    lock_guard<mutex> lock(mtx);
    totalSum += localSum;
}

// Thread function for search
void partialSearch(vector<int>& arr, int start, int end, int key) {
    for (int i = start; i < end; i++) {
        if (arr[i] == key) {
            lock_guard<mutex> lock(mtx);
            found = true;
            foundIndex = i;
            return;
        }
    }
}

int main() {
    int n, key, numThreads;

    cout << "Enter array size: ";
    cin >> n;

    cout << "Enter number of threads: ";
    cin >> numThreads;

    vector<int> arr(n);
    generateArray(arr, n);

    cout << "Enter key to search: ";
    cin >> key;

    vector<thread> threads;
    int partSize = n / numThreads;

    // ---- SUM USING THREADS ----
    auto start = high_resolution_clock::now();

    for (int i = 0; i < numThreads; i++) {
        int startIdx = i * partSize;
        int endIdx = (i == numThreads - 1) ? n : startIdx + partSize;
        threads.push_back(thread(partialSum, ref(arr), startIdx, endIdx));
    }

    for (auto& t : threads)
        t.join();

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    cout << "Threaded Sum = " << totalSum << endl;
    cout << "Threaded Sum Time = " << duration.count() << " microseconds\n";

    // Clear for search
    threads.clear();

    // ---- SEARCH USING THREADS ----
    start = high_resolution_clock::now();

    for (int i = 0; i < numThreads; i++) {
        int startIdx = i * partSize;
        int endIdx = (i == numThreads - 1) ? n : startIdx + partSize;
        threads.push_back(thread(partialSearch, ref(arr), startIdx, endIdx, key));
    }

    for (auto& t : threads)
        t.join();

    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);

    if (found)
        cout << "Key found at index " << foundIndex << endl;
    else
        cout << "Key not found\n";

    cout << "Threaded Search Time = " << duration.count() << " microseconds\n";

    return 0;
}

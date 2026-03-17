#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
using namespace std;
using namespace std::chrono;

// Function to generate random array
void generateArray(vector<int>& arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;   // Random numbers between 0–999
    }
}

// Function to compute sum
long long findSum(vector<int>& arr) {
    long long sum = 0;
    for (int i = 0; i < arr.size(); i++)
        sum += arr[i];
    return sum;
}

// Function to search key
int searchKey(vector<int>& arr, int key) {
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] == key)
            return i;  // return index
    }
    return -1;
}

int main() {
    int n, key;
    cout << "Enter array size: ";
    cin >> n;

    vector<int> arr(n);
    generateArray(arr, n);

    cout << "Enter key to search: ";
    cin >> key;

    // Measure sum time
    auto start = high_resolution_clock::now();
    long long sum = findSum(arr);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    cout << "Sum = " << sum << endl;
    cout << "Sequential Sum Time = " << duration.count() << " microseconds\n";

    // Measure search time
    start = high_resolution_clock::now();
    int index = searchKey(arr, key);
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);

    if (index != -1)
        cout << "Key found at index " << index << endl;
    else
        cout << "Key not found\n";

    cout << "Sequential Search Time = " << duration.count() << " microseconds\n";

    return 0;
}

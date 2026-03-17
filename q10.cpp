#include <iostream>
#include <thread>

using namespace std;

// Function to print first n natural numbers
void printNumbers(int n) {
    for(int i = 1; i <= n; i++) {
        cout << i << " ";
    }
    cout << endl;
}

int main() {
    int n;
    cout << "Enter value of n: ";
    cin >> n;

    // Create thread
    thread t1(printNumbers, n);

    // Wait for thread to finish
    t1.join();

    return 0;
}

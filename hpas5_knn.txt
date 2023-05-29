//Import header files
#include <bits/stdc++.h>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

//Create structure for point depiction
struct Point {
    int val;
    double x, y;
    double distance;
};

//Create comparator for std sorting
bool comparison(Point a, Point b) {
    return (a.distance < b.distance);
}

//Standard kNN function
int kNearestNeighbor(vector<Point> arr, int n, int k, Point p) {
    // Calculate distances between points in arr and point p
    for (int i = 0; i < n; i++) {
        arr[i].distance = sqrt((arr[i].x - p.x) * (arr[i].x - p.x) + (arr[i].y - p.y) * (arr[i].y - p.y));
    }

    // Sort the points based on distance in ascending order
    sort(arr.begin(), arr.end(), comparison);

    int freq1 = 0;
    int freq2 = 0;
    // Count frequencies of the nearest k points
    for (int i = 0; i < k; i++) {
        if (arr[i].val == 0) {
            freq1++;
        } else if (arr[i].val == 1) {
            freq2++;
        }
    }

    // Return the class label based on the majority vote
    return (freq1 > freq2 ? 0 : 1);
}

//Parallel kNN function
int kNearestNeighborParallel(vector<Point> arr, int n, int k, Point p) {
    //Run parallel loop
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        arr[i].distance = sqrt((arr[i].x - p.x) * (arr[i].x - p.x) + (arr[i].y - p.y) * (arr[i].y - p.y));
    }

    // Sort the points based on distance in parallel
    sort(arr.begin(), arr.end(), comparison);

    int freq1 = 0;
    int freq2 = 0;
    #pragma omp parallel for reduction(+: freq1, freq2)
    // Count frequencies of the nearest k points
    for (int i = 0; i < k; i++) {
        if (arr[i].val == 0) {
            freq1++;
        } else if (arr[i].val == 1) {
            freq2++;
        }
    }

    // Return the class label based on the majority vote
    return (freq1 > freq2 ? 0 : 1);
}

int main() {
    vector<Point> data_points;
    //Set number of points in training data
    int data_quantity = 1000;

    //Generate random training data with random classes 0 and 1
    for(int i = 0; i < data_quantity; i++){
        Point t_point;
        t_point.x = rand()%100;
        t_point.y = rand()%100;
        t_point.val = rand()%2;
        data_points.push_back(t_point);
    }

    Point p;
    cout << "Enter x and y coordinates of query point P: ";
    cin >> p.x >> p.y;

    int k;
    cout << "Enter value of k: ";
    cin >> k;

    //Compare times
    cout << "For serial k-NN:" << endl;
    auto start = chrono::high_resolution_clock::now();
    cout << "The value classified for the unknown point (belongs to group): " << kNearestNeighbor(data_points, data_quantity, k, p) << endl;
    auto end = chrono::high_resolution_clock::now();
    double dur = chrono::duration_cast<chrono::microseconds>(end - start).count();
    cout << "Time duration: " << dur << " microseconds" << endl;

    //Compare times
    cout << "For parallel k-NN:" << endl;
    start = chrono::high_resolution_clock::now();
    cout << "The value classified for the unknown point (belongs to group): " << kNearestNeighborParallel(data_points, data_quantity, k, p) << endl;
    end = chrono::high_resolution_clock::now();
    dur = chrono::duration_cast<chrono::microseconds>(end - start).count();
    cout << "Time duration: " << dur << " microseconds" << endl;

    return 0;
}

   

#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace std;


int foo(int x, int y) {
    return bar(x, y);
}
int bar(int x, int y) {
    return x-y;
}


int main()
{
    int ans = bar(8,18);
    cout << ans << endl;
    return 0;
}
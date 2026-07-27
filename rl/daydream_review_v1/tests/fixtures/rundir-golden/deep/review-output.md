# Review

## Issues
1. [tests/test_calc.py:13] `test_mean` cannot distinguish correct float division from a floor-division regression because the test input [1, 2, 3] produces an evenly-divisible mean (2.0), which would pass identically with floor division (//). Recommend adding a test case with non-integer mean, e.g. mean([1, 2]) == 1.5

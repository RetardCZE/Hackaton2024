"""
TASK 7
Improve retrieval by grouping small texts.

First improvement of our large dataset search would be to merge verses into batches, so we don't look for most
similar verse, but rather for most similar passage of the bible. When we take 10-20 verses at once, it is much easier
to differentiate between such groups.

Here copy your code from Task 6 and change the dict flattening part so it merges multiple verses into one
embedded and indexed element. The metadata will now be different as you are saving multiple verses under one
embedding. Also the print of the retrieved items may be different.

Compare subjective quality of results with Task 6 (it won't be great, but you will see some improvement)
"""





# Entrance

/train/RunDemo.py

# Notes

- DIP liked algorithm has no stop condition based on index. It is necessary to manually select the picture with the best visual effect among the pictures saved during the iteration.
  
  

- These reasons may cause the indicators and papers to be not exactly the same:
  1. Random noise.
     - Gaussian noise is added to input per iteration, which may lead to a slight change in the result.
  2. Index calculation platform.
     - Python implementation / Matlab implementation of index is not exactly the same.
     
     - Index in paper comes from Matlab implementation which is the same as in NSTMR, SupReME, MuSA,for a fairer comparison.
  3. Cutting.
     - Some comparison methods need to discard the edges of the image, which results in inconsistencies in size. For comparison, we cut the picture to the same size.Cropping is not included in demo.

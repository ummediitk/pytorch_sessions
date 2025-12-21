
## Use pytorch docs and google / stackoverflow / etc to solve question, Don't use LLMs or chatgpt (it's for your own good)
## In case of stuck ask me , ping me or ask other students but don't use LLMs

## 1. Generate 120 random numbers with torch.set_seed(123) (I haven't talked about it, but use this seed for now)
## 2. In the above convert this (reshape) this with .reshape and einops with batch of 5, channel 2, height 4 and width 3
## 3. Genearte mean, stdev, variance for above question 2 for each batch and each channel (use einops and torch.mean etc.)
## 4. Generate again mean, stdev, variance for above question 2 for eatch batch, eatch channel and each row(height) (use einops and torch.mean, etc)

## 5. Convert the question 1. to dimension of 15 x 8 , 15 rows and 8 columns, now convert it back to numpy array and try plotting with matplotlib , how does it look?
## 6. Create a tensor of 10x10 with same seed as question 1. but this time every alternate cells are either 1 or 0. the very first cell at (0,0) would 1 
## 7. Plot the question 6 by converting it to numpy array and see how it looks (do you see a chessboard?)
## 8. Can you convert the question 6 so that you will only see blue/red/green (any of these colors) instead of black
## 9. From question 5 , take the input and slice every alternate columns and fill it with zeros and then convert it to numpy and plot it again with matplotlib
## 10. Finally write your understanding of the class in a notebook and give it to chatgpt and verify if the understanding of yours are correct

## Bonus question:
## 11. Can you create math symbols like plus , minus, multiplication etc in a 9x9 grid
#  where plus would be defined as entire fifth colum and fifth row in black(1) rest of the cells are white(0), same for other symbols. Try it , it would be fun

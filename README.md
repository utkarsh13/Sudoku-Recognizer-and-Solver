# Sudoku-Recognizer-and-Solver
A Sudoku recognizer and solver code in Python

#### TESTED ON WINDOWS 8

### Versions used:
python 2.7
opencv 2.4.12
numpy 1.11.0

### About sudoku_train.py:
This file is used to train dataset.
It contains a list of path of images named as 'images' which are used for training.
If some other images are to be used for training, then list of path must be updated.
At the end, it saves all the data to files.

### About sudoku_solve.py :
This file is used to solve sudoku image.
It requires input as name of image.
Both sudoku_solve.py and image must be in same folder otherwise you need to adjust the path.
Dataset made during training is used for testing.
Digit recognition is done using KNN model.

### About training_images :
All the images for training are taken fron www.theguardian.com

### About test_images :
All the images for testing are also taken from www.theguardian.com.

### Result :
As the training set and testing set was from the images of www.theguardian.com, so the accuracy of sudoku recognition is 100%.

### Steps for training :
1. Run the file sudoku_train.py
2. For each image in list of path, 81 images are generated one by one and we need to provide input for them by pressing digit represented by them.
3. For blank squares, we need to press 0.
4. We do not need to provide input for all blank squares. Only a few will work fine.
5. To skip input for any particular square, press Enter.
6. Trainig will be complete after input for all images are done.

### Steps for solving :
1. Run the file sudoku_solve.py
2. Provide name of image as input for which we want to solve.
3. Sudoku is solved and displayed.

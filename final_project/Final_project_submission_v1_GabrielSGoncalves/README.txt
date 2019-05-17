Changes needed to be done to run original scripts on folder using Python3 and 
latest scikit-learn release:

# tester.py 

1) Add parenthesis to print statement;
2) Change import of module StratifiedShuffleSplit from sklearn.cross_validation
   to sklearn.model_selection (line 15)
3) Add 'n_splits=10, test_size=0.3' as parameter for StratifiedShuffleSplit 
   (line 29)
3) Add '.split(features, labels)' to 'cv' to iterate over it (line 34)


# poi_id.py

1) Change parameter 'with open()' to used 'rb' instead of just 'r'


# To run the scripts open the terminal and type:

> python3 poi_id.py

# It will create 3 pickle files necessary to run the next script:

> python3 tester.py

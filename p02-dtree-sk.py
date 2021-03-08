"""
Anna Spiro 
In this lab, we'll go ahead and use the sklearn API to learn a decision tree over some actual data!
 
Documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
 
We'll need to install sklearn.
Either use the GUI, or use pip:
 
   pip install scikit-learn
   # or: use install everything from the requirements file.
   pip install -r requirements.txt
"""

# We won't be able to get past these import statments if you don't install the library!
from sklearn.tree import DecisionTreeClassifier

import json  # standard python
from shared import dataset_local_path, TODO  # helper functions I made

#%% load up the data
examples = []
feature_names = set([])

with open(dataset_local_path("poetry_id.jsonl")) as fp:
    for line in fp:
        info = json.loads(line)
        # Note: the data contains a whole bunch of extra stuff; we just want numeric features for now.
        keep = info["features"]
        # make a big list of all the features we have:
        for name in keep.keys():
            feature_names.add(name)
        # whether or not it's poetry is our label.
        keep["y"] = info["poetry"]
        # hold onto this single dictionary.
        examples.append(keep)

#%% Convert data to 'matrices'
# NOTE: there are better ways to do this, built-in to scikit-learn. We will see them soon.

# turn the set of 'string' feature names into a list (so we can use their order as matrix columns!)
feature_order = sorted(feature_names)

# Set up our ML problem:
train_y = []
train_X = []

# Put every other point in a 'held-out' set for testing...
test_y = []
test_X = []

for i, row in enumerate(examples):
    # grab 'y' and treat it as our label.
    example_y = row["y"]
    # create a 'row' of our X matrix:
    example_x = []
    for feature_name in feature_order:
        example_x.append(float(row[feature_name]))

    # put every fourth page into the test set:
    if i % 4 == 0:
        test_X.append(example_x)
        test_y.append(example_y)
    else:
        train_X.append(example_x)
        train_y.append(example_y)

print(
    "There are {} training examples and {} testing examples.".format(
        len(train_y), len(test_y)
    )
)

"""
## Actual 'practical' assignment.
TODO(
    "1. Figure out what all of the parameters I listed for the DecisionTreeClassifier do."
)

splitter (best): how to choose the split at each node (best chooses best split, or feature with the highest importance)
max_features (None)
criterion (gini): how to measure the quality of a split/ feature importance (gini impurity measure)
max_depth (None): maximum depth of tree (None means that nodes are expanded until all leaves are pure)
random_state (13): controls randomness of estimator 

# Consult the documentation: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
TODO("2. Pick one parameter, vary it, and find some version of the 'best' setting.")

chosen parameter: max_depth

# Default performance:
# There are 2079 training examples and 693 testing examples.
# Score on Training: 1.000
# Score on Testing: 0.889
TODO("3. Leave clear code for running your experiment!")
"""

if __name__ == "__main__":
    # chosen parameter: max_depth (note: with parameter None, max_depth = 13)

    # create dictionary with max_depth as key, [training, testing] performance as value
    performance_dict = {}

    for i in range(1, 14):
        current_max_depth = i

        # change parameters for decision tree object
        f = DecisionTreeClassifier(
            splitter="best",
            max_features=None,
            criterion="gini",
            max_depth=current_max_depth,
            random_state=13,
        )  # type:ignore

        # train new tree!
        f.fit(train_X, train_y)

        performance_dict[current_max_depth] = [
            f.score(train_X, train_y),
            f.score(test_X, test_y),
        ]

    # find "best" max_depth and print results

    # initialize
    best_depth = 0
    best_testing = 0

    for key in performance_dict:
        # determine best results
        if performance_dict[key][1] > best_testing:
            best_depth = key
            best_testing = performance_dict[key][1]

        # print results
        print("max_depth set to " + str(key))
        print("score on training: " + str(performance_dict[key][0]))
        print("score on testing: " + str(performance_dict[key][1]))
        print("\n")

    print(
        "Best depth was found to be "
        + str(best_depth)
        + ", which is associated with a testing score of "
        + str(best_testing)
        + "."
    )
# %%

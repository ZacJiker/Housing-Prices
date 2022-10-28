from features.build_features import BasicPreprocessing
from models.train_model import NEATModel

# Preprocessing raw date
preprocessing = BasicPreprocessing()
X_train, X_test, y_train, y_test = preprocessing.run()

# Run NEAT algorithm
neat = NEATModel(X_train, y_train)
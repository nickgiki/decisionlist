from decisionlist._base import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class DecisionListClassifier:
    """Fits a Decision List Classifier implementing sequential covering principles"""
    
    def __init__(self, max_vars = 2, 
                 min_confidence = 0.8, 
                 min_support = 0.03,
                 num_trees = 20,
                 max_features_per_split = 0.2
                 ):
        """Initializes the classifier"""
        self.max_vars = max_vars
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.num_trees = num_trees
        self.max_features_per_split = max_features_per_split
    
    def fit(self, X, y):
        """Fit the classifier"""
        pass
    
    def predict(self, X_test):
        """Predicts labels for new samples"""
        pass
    
    def predict_proba(self, X_test):
        """Predicts probabilities for new samples"""
        pass
        
    def score(self, X_test, y_test):
        """Returns the accuracy for new X, y data"""
        pass
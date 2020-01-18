from sklearn.ensemble import RandomForestClassifier
import numpy as np
from ML_UtilsModule import Data_Management, Normalization
from boruta import BorutaPy

# load X and y
# NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
X, y = Data_Management.load_csv_types_features("pokemon.csv", ["hp", "attack", "defense", "sp_attack", "sp_defense","speed", "height_m", "weight_kg"
    , "percentage_male", "generation"])
y = y.ravel()

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

# find all relevant features - 5 features should be selected
feat_selector.fit(X, y)

# check selected features - first 5 features are selected
feat_selector.support_

# check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)
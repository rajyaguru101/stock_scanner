
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def select_algorithms():
    model1 = DecisionTreeClassifier(random_state=42)
    model2 = RandomForestClassifier(random_state=42, n_estimators=100)
    return [model1, model2]

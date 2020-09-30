# https://stackoverflow.com/questions/47399350/how-does-sample-weight-compare-to-class-weight-in-scikit-learn?rq=1
from sklearn.utils.class_weight import compute_sample_weight
y = [1,1,1,1,0,0,1]
print(compute_sample_weight(class_weight='balanced', y=y))
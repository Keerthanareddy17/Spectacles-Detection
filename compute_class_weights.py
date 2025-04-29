from sklearn.utils import class_weight
import numpy as np
import pickle

count_no_glasses = 152249
count_glasses = 10521

# Create a full label list
train_labels = [0] * count_no_glasses + [1] * count_glasses

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

# Convert to a dictionary {0: weight_0, 1: weight_1}
class_weights = dict(enumerate(class_weights))
print("Computed class weights:", class_weights)

# Save it to a file for later use 
with open('class_weights.pkl', 'wb') as f:
    pickle.dump(class_weights, f)

print("Class weights saved to class_weights.pkl")

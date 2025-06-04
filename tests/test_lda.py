from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from gda.lda import LDA

# Create dataset
X, y = make_classification(n_samples=1000, n_classes=2, n_informative=2,
                           n_features=2, n_redundant=0, n_repeated=0, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

# Train model
model = LDA()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

#  Optional: Visualization
from utils.plot_utils import plot_decision_boundary
plot_decision_boundary(model, X, y, title="Linear Discriminant Analysis")

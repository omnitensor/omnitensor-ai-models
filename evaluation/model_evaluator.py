
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ModelEvaluator:
    def __init__(self, model, test_data, test_labels):
        '''
        Initialize the evaluator with a model and test dataset.

        Args:
            model: The trained model to evaluate.
            test_data (np.ndarray): Input features for testing.
            test_labels (np.ndarray): True labels corresponding to the test data.
        '''
        self.model = model
        self.test_data = test_data
        self.test_labels = test_labels

    def evaluate(self):
        '''
        Evaluates the model on the test dataset and computes performance metrics.

        Returns:
            dict: A dictionary containing evaluation metrics such as accuracy, precision,
                  recall, F1 score, and confusion matrix.
        '''
        print("Running model predictions...")
        predictions = self.model.predict(self.test_data)
        
        if predictions.shape != self.test_labels.shape:
            raise ValueError("Predictions shape does not match test labels shape.")

        metrics = {
            "accuracy": accuracy_score(self.test_labels, predictions),
            "precision": precision_score(self.test_labels, predictions, average='weighted'),
            "recall": recall_score(self.test_labels, predictions, average='weighted'),
            "f1_score": f1_score(self.test_labels, predictions, average='weighted'),
            "confusion_matrix": confusion_matrix(self.test_labels, predictions).tolist()
        }
        return metrics

    def report_metrics(self, metrics):
        '''
        Prints the evaluation metrics in a readable format.

        Args:
            metrics (dict): The dictionary of computed metrics.
        '''
        print("Model Evaluation Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"Recall: {metrics['recall']:.2f}")
        print(f"F1 Score: {metrics['f1_score']:.2f}")
        print("Confusion Matrix:")
        for row in metrics["confusion_matrix"]:
            print(row)

# Example usage:
if __name__ == "__main__":
    # First example
    class FirstModel:
        def predict(self, data):
            return np.random.choice([0, 1], size=data.shape[0])
    
    # Generate synthetic test data
    test_data = np.random.rand(100, 10)
    test_labels = np.random.choice([0, 1], size=100)

    # Instantiate the model and evaluator
    model = FirstModel()
    evaluator = ModelEvaluator(model, test_data, test_labels)

    # Evaluate and print metrics
    metrics = evaluator.evaluate()
    evaluator.report_metrics(metrics)

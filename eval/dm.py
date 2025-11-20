from eval.evaluator import Evaluator

def check_actions(pred, gt):
  return pred.strip().lower() == gt.strip().lower()

# Example usage
predicted_action = "confirmation(pizza_ordering)"
ground_truth_action = "confirmation(pizza_ordering)"
print(check_actions(predicted_action, ground_truth_action))
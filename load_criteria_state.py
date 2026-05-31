import os
import pickle

def load_criteria_state(model_name, data_dir):
    state_path = os.path.join(data_dir, f"activation_coverage_{model_name}",
                              f"{model_name}_criteria_state.pkl")

    if not os.path.exists(state_path):
        print(f"WARNING: Criteria state file not found at {state_path}")
        return None

    try:
        with open(state_path, 'rb') as f:
            criteria_state = pickle.load(f)
        print(f"Loaded criteria states from {state_path}")
        return criteria_state
    except Exception as e:
        print(f"WARNING: Failed to load criteria state: {e}")
        return None

from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    """Abstract base class for a recommender system that suggests actions based on context (e.g., BERT QA)."""

    def __init__(self):
        """Initialize the recommender system (load models, etc.)."""
        # For example, load a pre-trained BERT QA model or other resources.
        super().__init__()

    @abstractmethod
    def recommend(self, state, action_space=None):
        """
        Analyze the current state (and optional list of possible actions) and return a recommended action.
        :param state: The current state of the environment (could be any representation, e.g., dict or vector).
        :param action_space: Optional list of available actions to choose from.
        :return: A recommendation for the next action (could be an action index or name).
        """
        raise NotImplementedError("This method should be implemented by a subclass.")


# Example subclass outline (not implemented, just for context):
class BertQARecommender(BaseRecommender):
    """Recommender that uses a BERT-based QA model to suggest actions."""

    def __init__(self, model_path):
        # Load a BERT QA model (e.g., DistilBERT fine-tuned for Q&A) from model_path
        super().__init__()
        # self.model = load_bert_model(model_path)

    def recommend(self, state, action_space=None):
        """
        Use the BERT QA model to suggest an action.
        For example, form a question from the state (like "What is the best exploit?") and context (vuln info),
        get an answer from the model, and map that to an action.
        """
        # question = formulate_question_from_state(state)
        # context = compile_context_from_state(state)
        # answer = self.model.ask(question, context)
        # recommended_action = interpret_answer_as_action(answer, action_space)
        # return recommended_action
        return None  # Placeholder

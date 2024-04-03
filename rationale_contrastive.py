import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    @staticmethod
    def encode_text(text):
        # Placeholder for your text encoding function
        # Should return a fixed-size vector for the given text
        pass

    @staticmethod
    def cosine_similarity(a, b):
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)

    def forward(self, small_model_output, large_models_outputs):
        """
        Compute the contrastive loss.

        Parameters:
        - small_model_output: dict, output of the small model.
        - large_models_outputs: list of dicts, outputs of the large models.

        Returns:
        - loss: torch.Tensor, the computed contrastive loss.
        """
        small_rationale_vec = self.encode_text(small_model_output["rationale"])
        loss = torch.tensor(0.0, requires_grad=True)

        for output in large_models_outputs:
            large_rationale_vec = self.encode_text(output["rationale"])
            similarity = self.cosine_similarity(small_rationale_vec, large_rationale_vec)

            if output["prediction"] == str(output["label"]):  # Positive example
                # Minimize the distance for positive examples
                positive_loss = 1 - similarity
                loss += F.relu(positive_loss)
            else:  # Negative example
                # Maximize the distance for negative examples, hence minimize negative similarity
                negative_loss = similarity + self.margin
                loss += F.relu(negative_loss)

        # Normalize the loss by the number of large model outputs
        return loss / len(large_models_outputs)


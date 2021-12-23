from typing import Sequence
from typing import Any, Callable, Sequence, Optional
from transformers import BertTokenizer, FlaxBertForPreTraining
import flax.linen as nn

MODEL_TYPE = 'bert-base-cased'

def CreateTokenizer():
    """Creates a tokenizer for retrieval"""
    return BertTokenizer.from_pretrained(MODEL_TYPE)

def CreateEncoder():
    """Creates a common encoder for retrieval"""
    return FlaxBertForPreTraining.from_pretrained(MODEL_TYPE)

class DocumentEncoder(nn.Module):
    """One encoder for a document."""
    dimensions: Sequence[int]

    def setup(self):
        """Initialize dense modules"""


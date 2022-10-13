from torch import nn, Tensor
from positional_encodings import PositionalEncodingPermute1D, Summer



class Embedder(nn.Module):
    '''
    This class takes care of generating the embeddings for each text using a pretrained
    Longformer, then the number of channels is reduced using a 1x1 convolutional layer.
    The output are the embeddings plus the positional encodings.
    Input:
      embedder  : The model used for generating the embeddings
      tokenizer : The tokenizer required by the model
      text      : The input text to embed
      d         : The number of output channel. (the name follows the covention of the paper)
    Output:
      embeddings    : document embeddings  [batch_size, text_length, channels]
      posit_encoder : positional encodings [batch_size, text_length, channels]
    '''

    def __init__(self, embedder, tokenizer):
        super(Embedder, self).__init__()
        self.embedder = embedder
        self.tokenizer = tokenizer

    def forward(self, text: str):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        embeddings = self.embedder(input_ids)['last_hidden_state']

        '''
        Maybe for the encodings we can use the Learned Positional Encodings of LED Model ???
        '''
        pos_encoder = PositionalEncodingPermute1D(embeddings.size()[2])  # Numbers of channels
        pos_encodings = pos_encoder(embeddings)

        return embeddings, pos_encodings

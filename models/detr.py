from lib2to3.pgen2 import token
import torch
from torch import nn
import torch.nn.functional as F


class Backbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.from_pretrained("allenai/longformer-base-4096")

    def forward(self, input_ids):
        embeddings = self.backbone(input_ids)
        return embeddings


class Transformer(nn.Module):
    def __init__(self, model, config):
        super().__init__()

        self.model = model(config)

        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

    def forward(self, enc_input_ids, query_embed, glob_enc_attn, glob_dec_attn):

        enc = self.encoder(enc_input_ids, global_attention_mask=glob_enc_attn)
        tgt = torch.zeros_like(query_embed)
        tgt = tgt.unsqueeze(-1).permute(2, 0, 1)
        tgt = tgt + query_embed
        dec = self.decoder(
            inputs_embeds=tgt,
            encoder_hidden_states=enc["last_hidden_state"],
            global_attention_mask=glob_dec_attn,
        )

        return dec

    def reset_parameters(self):
        print("Initializing transformer weights...")
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("Weights initialized using Xavier.")


class DETR(nn.Module):
    def __init__(
        self,
        model,
        config,
        backbone,
        num_classes,
        num_queries,
        hidden_dim,
        class_depth,
        bbox_depth,
        dropout=0,
        class_biases=None,
        init_weight=None,
        transformer_hidden_dim=768,
    ):
        super().__init__()

        self.backbone = Backbone(backbone)

        self.transformer = Transformer(model, config)

        self.linear_class = MLP(
            transformer_hidden_dim, hidden_dim, num_classes + 1, class_depth, dropout
        )
        if class_biases is not None:
            self.linear_class.layers[-1].bias.data = torch.Tensor(class_biases)
            # self.linear_class.bias.data = torch.Tensor(class_biases)

        if init_weight == "xavier":
            print("Initializing MLP weights for classes...")
            (torch.nn.init.xavier_uniform_(self.linear_class.layers[i].weight) for i in range(self.linear_class.num_layers))  # type: ignore
            print("Weights initialized using Xavier.")

        self.linear_bbox = MLP(
            transformer_hidden_dim, hidden_dim, 2, bbox_depth, dropout
        )
        if init_weight == "xavier":
            print("Initializing MLP weights for bbox...")
            (torch.nn.init.xavier_uniform_(self.linear_bbox.layers[i].weight) for i in range(self.linear_bbox.num_layers))  # type: ignore
            print("Weights initialized using Xavier.")

        self.query_embed = nn.Embedding(num_queries, transformer_hidden_dim)
        # output positional encodings (object queries)
        self.query_pos = nn.parameter.Parameter(torch.rand(100, transformer_hidden_dim))
        self.num_queries = num_queries

    def forward(self, inputs_ids, glob_enc_attn, glob_dec_attn):
        h = self.transformer(
            inputs_ids, self.query_embed.weight, glob_enc_attn, glob_dec_attn
        )["last_hidden_state"]

        # finally project transformer outputs to class labels and bounding boxes
        return {
            "pred_logits": self.linear_class(h),
            "pred_boxes": self.linear_bbox(h).sigmoid(),
        }

    def set_transformer_trainable(self, trainable: bool):
        for param in self.transformer.parameters():
            param.requires_grad = trainable

    def set_backbone_trainable(self, trainable: bool):
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def transformer_parameters(self):
        return (
            p
            for n, p in self.named_parameters()
            if "transformer" in n and p.requires_grad
        )

    def last_layers_parameters(self):
        return (
            p
            for n, p in self.named_parameters()
            if "transformer" not in n and "backbone" not in n and p.requires_grad
        )

    def backbone_parameters(self):
        return (
            p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad
        )

    def reset_parameters(self):
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = (
                self.dropout(F.relu(layer(x)))
                if i < self.num_layers - 1
                else self.dropout(layer(x))
            )
        return x


class PrepareInputs:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, docs):
        return self.tokenizer(docs, padding=True, return_tensors="pt").input_ids

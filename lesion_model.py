import torch
import torch.nn as nn
import torch.nn.functional as F

class LesionEmbeddingModel(nn.Module):
    def __init__(self, lesion_names, glom_dim=768, hidden_dim=512, embed_dim=256, dropout=0.1):
        super().__init__()
        self.lesion_names = lesion_names
        self.num_lesions = len(lesion_names)
        self.embed_dim = embed_dim
        
        # Separate encoder for each lesion type
        self.lesion_encoders = nn.ModuleDict()
        self.lesion_classifiers = nn.ModuleDict()
        
        for lesion_name in lesion_names:
            # Encoder network for this lesion
            self.lesion_encoders[lesion_name] = nn.Sequential(
                nn.Linear(glom_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            
            # Binary classifier for this lesion
            self.lesion_classifiers[lesion_name] = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
    
    def forward(self, glom_features):
        embeddings = {}
        logits_list = []
        
        for lesion_name in self.lesion_names:
            # Get embedding for this lesion
            embed = self.lesion_encoders[lesion_name](glom_features)
            embeddings[lesion_name] = embed
            
            # Get classification logit
            logit = self.lesion_classifiers[lesion_name](embed)
            logits_list.append(logit)
        
        logits = torch.cat(logits_list, dim=-1)  # [B, num_lesions]
        
        return embeddings, logits
    
    def get_embedding(self, glom_features, lesion_name):
        """Extract embedding for specific lesion"""
        return self.lesion_encoders[lesion_name](glom_features)
    
    def save_lesion_module(self, lesion_name, save_path):
        """Save encoder and classifier for a specific lesion"""
        torch.save({
            'encoder_state': self.lesion_encoders[lesion_name].state_dict(),
            'classifier_state': self.lesion_classifiers[lesion_name].state_dict(),
            'lesion_name': lesion_name,
            'embed_dim': self.embed_dim
        }, save_path)
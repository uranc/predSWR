import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization

class PrototypicalMemory(Layer):
    def __init__(self, num_prototypes=4, feature_dim=128, temperature=0.1):
        super().__init__()
        self.num_prototypes = num_prototypes  # onset, peak, offset, noise
        self.temperature = temperature
        
        # Initialize prototypes (learnable)
        self.prototypes = self.add_weight(
            shape=(num_prototypes, feature_dim),
            initializer='orthogonal',
            trainable=True,
            name='prototypes'
        )
        
    def call(self, features, labels=None, training=True):
        # Compute distances to prototypes
        # Normalize features and prototypes
        features_norm = tf.nn.l2_normalize(features, axis=-1)
        protos_norm = tf.nn.l2_normalize(self.prototypes, axis=-1)
        
        # Calculate cosine similarities
        similarities = tf.matmul(features_norm, protos_norm, transpose_b=True)
        
        # Scale by temperature
        logits = similarities / self.temperature
        
        if training and labels is not None:
            # Prototypical loss
            loss = self.prototypical_loss(logits, labels)
            self.add_loss(loss)
            
        return logits, similarities

    def prototypical_loss(self, logits, labels):
        # Convert numeric labels to one-hot
        labels_onehot = tf.one_hot(labels, self.num_prototypes)
        
        # Cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_onehot,
            logits=logits
        )
        return tf.reduce_mean(loss)

class HistoryAwareTCNMixerBlock(Layer):
    def __init__(self, dim, tcn_params, num_prototypes=4):
        super().__init__()
        # ...existing TCNMixerBlock init code...
        
        self.proto_memory = PrototypicalMemory(
            num_prototypes=num_prototypes,
            feature_dim=dim
        )
        
    def call(self, x, phase_labels=None, training=True):
        # Regular mixing operations
        x = super().call(x)
        
        # Prototypical memory operations
        proto_logits, similarities = self.proto_memory(
            x, 
            labels=phase_labels,
            training=training
        )
        
        # Weight features based on prototype similarities
        attention = tf.nn.softmax(similarities, axis=-1)
        proto_context = tf.matmul(attention, self.proto_memory.prototypes)
        
        return x + proto_context, proto_logits

class PatchMixerTCN(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        # ...existing init code...
        
        # Additional head for phase classification
        self.phase_head = Dense(4, activation='softmax')  # 4 phases
        
    def call(self, inputs, phase_labels=None, training=True):
        x = self.patch_tokenizer(inputs)
        
        all_logits = []
        for block in self.blocks:
            x, proto_logits = block(
                x, 
                phase_labels=phase_labels,
                training=training
            )
            all_logits.append(proto_logits)
        
        # Average prototype logits across blocks
        avg_logits = tf.reduce_mean(tf.stack(all_logits), axis=0)
        
        # Final predictions
        main_output = self.fc(self.pool(x))
        phase_output = self.phase_head(self.pool(x))
        
        return {
            'main': main_output,
            'phase': phase_output,
            'proto_logits': avg_logits
        }

def proto_phase_loss(y_true, proto_logits, phase_labels, lambda_proto=0.1):
    # Main prediction loss
    main_loss = tf.keras.losses.binary_crossentropy(y_true, proto_logits)
    
    # Phase classification loss
    phase_loss = tf.keras.losses.sparse_categorical_crossentropy(
        phase_labels, proto_logits)
    
    return main_loss + lambda_proto * phase_loss

# Usage example:
"""
model = PatchMixerTCN(
    patch_size=32,
    stride=16,
    embed_dim=128,
    num_blocks=4,
    num_prototypes=4  # onset, peak, offset, noise
)

# Training
outputs = model(inputs, phase_labels=phase_labels, training=True)
loss = proto_phase_loss(
    y_true=targets,
    proto_logits=outputs['proto_logits'],
    phase_labels=phase_labels
)
"""

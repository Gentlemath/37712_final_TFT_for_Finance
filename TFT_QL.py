import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for feature selection and gating mechanisms"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate(x))
    

    
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, batch_first=False):
        super(GatedResidualNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        if self.input_dim!=self.output_dim:
            self.skip_layer = nn.Linear(self.input_dim, self.output_dim)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.glu = GatedLinearUnit(output_dim, output_dim) # Use GLU instead of simple linear
        self.norm = nn.LayerNorm(output_dim)   #not sure LayerNorm 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.input_dim!=self.output_dim:   #last dim of x is not the same as output
            res = self.skip_layer(x)  #(len, B, output_dim)
        else:
            res = x
        x = self.fc1(x)     #(len, B, hidden_dim)
        x = self.elu(x)
        x = self.fc2(x)    #(len, B, output_dim)
        x = self.dropout(x)
        x = self.glu(x)  # Apply GLU
        x = x + res  # Residual connection
        return self.norm(x)
    
class VSN(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size, dropout):
        '''
        input_size: generally, concatenatation of variable embddings
        num_inputs: number of variables involved in the input embedddings
        hidden_size: the hidden size of LSTM
        '''
        super(VSN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size =input_size
        self.num_inputs = num_inputs
        self.dropout = dropout

        self.flattened_grn = GatedResidualNetwork(self.num_inputs*self.input_size, self.hidden_size, self.num_inputs, self.dropout)

        self.single_variable_grns = nn.ModuleList()
        for i in range(self.num_inputs):
            self.single_variable_grns.append(GatedResidualNetwork(self.input_size, self.hidden_size, self.hidden_size, self.dropout))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedding):
        
        sparse_weights = self.flattened_grn(embedding)   #(len, B, num_variables)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)  #(len, B, 1, num_variables)

        var_outputs = []     
        for i in range(self.num_inputs):    #num_inputs(num_variables) of (len, B, hidden_size)
            var_outputs.append(self.single_variable_grns[i](embedding[:,:, (i*self.input_size) : (i+1)*self.input_size]))

        var_outputs = torch.stack(var_outputs, axis=-1)     # torch.stack along axis=-1 (len, B, hidden_size, num_variables)
        outputs = var_outputs*sparse_weights         # (len, B, hidden_size, num_variables)
        outputs = outputs.sum(axis=-1)   #(len, B, hidden_size)

        return outputs, sparse_weights


class PositionEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = hidden_dim

        positional_encodings = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000) / hidden_dim)
        )
        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)
        positional_encodings = positional_encodings.unsqueeze(0)    # (1, max_len, hidden_dim)

        #positional_encodings is stored as part of the model but not updated during training
        self.register_buffer(
            "positional_encodings", positional_encodings, persistent=False
        )     

    def forward(self, x):
        x = x * math.sqrt(self.hidden_dim)
        seq_len = x.size(0)
        x = x + self.positional_encodings[:, : seq_len].view(seq_len, 1, self.hidden_dim)
        return self.dropout(x)

class AddNorm(nn.Module):
    def __init__(self, feature_dim, eps=1e-6):
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(feature_dim, eps=eps)

    def forward(self, x, sublayer_output):
        """
        Args:
            x: Tensor of shape (seq_len, batch, feature_dim)
            sublayer_output: Tensor of shape (seq_len, batch, feature_dim)
        Returns:
            Normalized tensor of shape (seq_len, batch, feature_dim)
        """
        return self.layer_norm(x + sublayer_output)
    

class TFT(nn.Module):
    def __init__(self, config):
        super(TFT, self).__init__()

        self.device = config['device']
        self.batch_size=config['batch_size']

        self.seq_length = config['seq_length']           # whole TS length
        self.encoder_len = config['num_encoder_steps']   #TS length of encoder

        self.num_cat =  config['num_categoical_variables']   #total number of categoical variables/features
        self.cat_embedding_vocab_sizes = config['cat_embedding_vocab_sizes']
        self.num_real_encoder =  config['num_real_encoder'] #number of real variables in encoer
        self.num_real_decoder =  config['num_real_decoder'] #number of real variables in decoer
        self.num_input_to_mask = config['num_masked_vars']  #number of real variables masked in decoder ('Close', 'Volume')
        self.embedding_dim = config['embedding_dim']  #embeddding dim of each variable

        self.hidden_size = config['lstm_hidden_dimension']   #LSTM params
        self.lstm_layers = config['lstm_layers']
        self.dropout = config['dropout']
        self.attn_heads = config['attn_heads']    #attention params
        self.num_quantiles = config['num_quantiles']
        self.valid_quantiles = config['vailid_quantiles']


        # Embeddings for categorical variables
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_categories, self.embedding_dim).to(self.device) for num_categories in self.cat_embedding_vocab_sizes
        ])

        # Linear layers for continuous variables
        self.continuous_linear = nn.ModuleList([
            nn.Linear(1, self.embedding_dim).to(self.device) for _ in range(self.num_real_encoder)
        ])

        self.encoder_variable_selection = VSN(self.embedding_dim,
                                (self.num_real_encoder +  self.num_cat),
                                self.hidden_size,
                                self.dropout)

        self.decoder_variable_selection = VSN(self.embedding_dim,
                                (self.num_real_decoder +  self.num_cat),
                                self.hidden_size,
                                self.dropout)
        
        self.position_encoding = PositionEncoding(self.hidden_size, self.dropout, self.seq_length)
        
        self.lstm_encoder_input_size = self.embedding_dim*(self.num_real_encoder + self.num_cat)
        
        self.lstm_decoder_input_size = self.embedding_dim*(self.num_real_decoder + self.num_cat)                                     

        self.lstm_encoder = nn.LSTM(input_size=self.hidden_size, 
                            hidden_size=self.hidden_size,
                           num_layers=self.lstm_layers,
                           dropout=self.dropout, batch_first=False)
        
        self.lstm_decoder = nn.LSTM(input_size=self.hidden_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.lstm_layers,
                                   dropout=self.dropout, batch_first=False)

        self.post_lstm_gate = GatedLinearUnit(self.hidden_size, self.hidden_size)
        self.addnorm = AddNorm(self.hidden_size)
          
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.attn_heads, batch_first=False)

        self.post_attn_gate = GatedLinearUnit(self.hidden_size, self.hidden_size)
        self.attn_dropout = nn.Dropout(self.dropout)
        #self.post_attn_norm = TimeDistribution(nn.BatchNorm1d(self.hidden_size, self.hidden_size))

        self.pos_wise_ff = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout)

        #self.pre_output_norm = TimeDistribution(nn.BatchNorm1d(self.hidden_size, self.hidden_size))
        self.pre_output_gate = GatedLinearUnit(self.hidden_size, self.hidden_size)

        self.output_layer = nn.Linear(self.hidden_size, self.num_quantiles)

    def apply_embedding(self, x, apply_masking = True):
        '''
            x : (batch_size, timesteps, features)
            features: 'Close','Volume', 'Day_of_Week', 'Day_of_Month', 'Week_of_Year','Month'   
           apply_masking: mask variables that should not be accessed after the encoding steps
        '''

        #real variable embeddings  ['Close','Volume']
        if apply_masking:
            time_varying_real_vectors = []
            for i in range(self.num_real_decoder):
                emb = self.continuous_linear[i+self.num_input_to_mask](x[:,:,i+self.num_input_to_mask].view(x.size(0), -1, 1))
                time_varying_real_vectors.append(emb)           
            if time_varying_real_vectors == []:
                time_varying_real_embedding = []
            else:
                time_varying_real_embedding = torch.cat(time_varying_real_vectors, dim=2)

        else: 
            time_varying_real_vectors = []
            for i in range(self.num_real_encoder):
                emb = self.continuous_linear[i](x[:,:,i].view(x.size(0), -1, 1))  #(B, len, emb_dim)
                time_varying_real_vectors.append(emb)
            time_varying_real_embedding = torch.cat(time_varying_real_vectors, dim=2) #(B, len, emb_dim*num_real)
        
        
         ##categorical embeddings ['Day_of_Week', 'Day_of_Month', 'Week_of_Year','Month']
        time_varying_categoical_vectors = []
        for i in range(self.num_cat):
            cat_feature = x[:, :, self.num_real_encoder + i].long()
            emb = self.cat_embeddings[i](cat_feature)
            #emb = self.cat_embeddings[i](x[:, :,self.num_real_encoder+i].view(x.size(0), -1, 1).long()) #(B, len, emb_dim)
            time_varying_categoical_vectors.append(emb)
        time_varying_categoical_embedding = torch.cat(time_varying_categoical_vectors, dim=2)  #(B, len, emb_dim*num_cat)

        ##concatenate all embeddings  #(B, len, emb_dim*(num_cat + num_real)
        if time_varying_real_embedding == []:
            embeddings = time_varying_categoical_embedding
        else:
            embeddings = torch.cat([time_varying_categoical_embedding,time_varying_real_embedding], dim=2)
        
        return embeddings.view( -1, x.size(0), embeddings.size(2))
    
    def init_lstm_hidden(self, bs, hs):
        '''
        initialize hidden state: (num_layers * num_directions, batch_size, hidden_size)
        initialize cell state: (num_layers * num_directions, batch_size, hidden_size)
        num_directions: 1
        '''
        return torch.zeros(self.lstm_layers, bs, hs, device=self.device)
        
    def encode(self, x, hidden=None):    
        if hidden is None:
            hidden = self.init_lstm_hidden(x.shape[1], x.shape[2])    
        if x.shape[1] != hidden.shape[1]:
            breakpoint()
        output, (hidden, cell) = self.lstm_encoder(x, (hidden, hidden))      
        return output, hidden
        
    def decode(self, x, hidden=None):       
        if hidden is None:
            hidden = self.init_lstm_hidden(x.shape[1], x.shape[2])            
        output, (hidden, cell) = self.lstm_decoder(x, (hidden, hidden))        
        return output, hidden
    

    def forward(self, x):
        '''
        x: iunput data, size of (B, seq_len, num_features)
        '''
        #Embedding and variable selection
        embeddings_encoder = self.apply_embedding(x[:,:self.encoder_len,:],apply_masking=False)  #(len, B, emb_dim*(num_cat + num_real))
        embeddings_decoder = self.apply_embedding(x[:,self.encoder_len:,:], apply_masking=True)  #(len, B, emb_dim*(num_cat))
        embeddings_encoder, encoder_sparse_weights = self.encoder_variable_selection(embeddings_encoder)  #(len_encode, B, hidden_dim)
        embeddings_decoder, decoder_sparse_weights = self.decoder_variable_selection(embeddings_decoder)  #(len_decode, B, hidden_dim)

        scale_factor = torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float, device=embeddings_encoder.device))
        embeddings_encoder = embeddings_encoder * scale_factor  # Scale the embeddings  
        embeddings_encoder_pos = self.position_encoding(embeddings_encoder)

        embeddings_decoder = embeddings_decoder * scale_factor  # Scale the embeddings  
        embeddings_decoder_pos = self.position_encoding(embeddings_decoder)

         ## LSTM
        lstm_input = torch.cat([embeddings_encoder_pos,embeddings_decoder_pos], dim=0)   #(seq_len, B, hidden_dim)
        encoder_output, hidden = self.encode(embeddings_encoder)
        decoder_output, _ = self.decode(embeddings_decoder, hidden)
        lstm_output = torch.cat([encoder_output, decoder_output], dim=0)

        ## Gate
        lstm_output = self.post_lstm_gate(lstm_output)  #(seq_len, B, hidden_dim)
        ## Add and norm
        attn_input = self.addnorm(lstm_input, lstm_output)   #(seq_len, B, hidden_dim)

        ### Encoder-Decoder Attention
        attn_output, attn_output_weights = self.multihead_attn(
            attn_input[self.encoder_len:,:,:], attn_input[:self.encoder_len,:,:], attn_input[:self.encoder_len,:,:])

        ## Gate
        attn_output = self.post_attn_gate(attn_output)
        ## Add and norm
        attn_output = self.addnorm(attn_input[self.encoder_len:,:,:], attn_output)  #(decode_len, B, hidden_dim)

        ## Position-wise feed forward
        output = self.pos_wise_ff(attn_output)    #(decode_len, B, hidden_dim)

        ## Gate
        output = self.pre_output_gate(output)     #(decode_len, B, hidden_dim)

        # Add and norm
        output = self.addnorm(lstm_output[self.encoder_len:,:,:], output)   #(decode_len, B, hidden_dim)

        bs = output.shape[1]
        output = self.output_layer(output.view(bs, -1, self.hidden_size))   #(B, 1, 1)
        
        # Fully connected output
        out = output[:, -1, :] 
        return out.squeeze()





369
id_col = 'categorical_id'
time_col='hours_from_start'
input_cols =['power_usage', 'hour', 'day_of_week', 'hours_from_start', 'categorical_id']
target_col = 'power_usage'
time_steps=192
num_encoder_steps = 168
output_size = 1
max_samples = 1000
input_size = 5

elect = ts_dataset.TSDataset(id_col, time_col, input_cols,
                      target_col, time_steps, max_samples,
                     input_size, num_encoder_steps, output_size, train)


batch_size=64
loader = DataLoader(
            elect,
            batch_size=batch_size,
            num_workers=2,
            shuffle=True
        )

static_cols = []
categorical_cols = ['Day_of_Week', 'Day_of_Month', 'Week_of_Year','Month']
real_cols = ['Close','Volume']
config = {}
config['static_variables'] = len(static_cols)
config['time_varying_categoical_variables'] = 4
config['time_varying_real_variables_encoder'] = 2
config['time_varying_real_variables_decoder'] = 0
config['num_masked_series'] = 2
config['static_embedding_vocab_sizes'] = [369]
config['time_varying_embedding_vocab_sizes'] = [369]
config['embedding_dim'] = 8
config['lstm_hidden_dimension'] = 160
config['lstm_layers'] = 1
config['dropout'] = 0.05
config['device'] = 'cpu'
config['batch_size'] = 64
config['encode_length'] = 168
config['attn_heads'] = 4
config['num_quantiles'] = 3
config['vailid_quantiles'] = [0.1,0.5,0.9]
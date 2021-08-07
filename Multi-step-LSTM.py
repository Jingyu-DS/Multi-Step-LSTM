def split_sequences(sequences, n_steps_in, n_steps_out):
  X, y = list(), list()
  for i in range(len(sequences)):
    end_ix = i + n_steps_in
    out_end_ix = end_ix + n_steps_out - 1
    if out_end_ix > len(sequences):
      break 
      
    seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
    X.append(seq_x)
    y.append(seq_y)
    
  return array(X), array(y)

model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(3, return_sequences=False))
model.add(Dense(n_steps_out))
opt = SGD(learning_rate=0.01, momentum=0.9, clipvalue=0.5)
model.compile(loss='mean_squared_error', optimizer=opt)

def plot_forecasts(series, forecasts, n_test):
  plt.figure(figsize=(20, 5))
  plt.plot(series.values)
  
  for i in range(len(forecasts)):
    off_s = len(series) - n_test + i - 1
    off_e = off_s + len(forecasts[i]) +1 
    xaxis = [x for x in range(off_s, off_e)]
    yaxis = [series.values[off_s]] + forecasts[i].tolist()
    plt.plot(xaxis, yaxis, color = 'red')
    
  plt.show()
    

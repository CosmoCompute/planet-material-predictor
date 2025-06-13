from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_sarimax(df, column='max_temp', forecast_days=30):
    train = df[column].iloc[:-forecast_days]
    test = df[column].iloc[-forecast_days:]

    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    result = model.fit(disp=False)

    forecast = result.get_forecast(steps=forecast_days)
    conf = forecast.conf_int()
    conf['forecast'] = forecast.predicted_mean
    conf['actual'] = test.values
    return result, conf

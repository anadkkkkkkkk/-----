import numpy as np, pandas as pd, yfinance as yf, datetime, time, requests, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

TELEGRAM_TOKEN = '8540803234:AAHXdvF2-GW4vcnDnWoD9Mn42r6_rJq_yic'
CHAT_ID = '7644255708'
def send_telegram(msg):
    try: requests.post(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage', data={'chat_id': CHAT_ID, 'text': msg}, timeout=10)
    except: pass

print("🥇 بوت الذهب الأقوى – 4 نماذج محسّنة بالكامل")
send_telegram("🟢 بوت الذهب الأقوى بدأ (4 نماذج محسّنة)")

SYMBOL = "GC=F"
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE = 0.01
LEVERAGE = 5
STOP_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
MIN_CONFIDENCE = 0.55
LSTM_LOOKBACK = 30
MODEL_XGB = 'gold_xgb.json'
MODEL_LSTM = 'gold_lstm.h5'
MODEL_RF = 'gold_rf.pkl'
MODEL_CAT = 'gold_cat.cbm'
CAPITAL_FILE = 'capital_mtf.txt'
STATE_FILE = 'state.txt'

def fetch_data(interval='5m', days=120):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)
    df = yf.download(SYMBOL, start=start, end=end, interval=interval, progress=False)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    df = df[['Open','High','Low','Close','Volume']].copy()
    df.columns = ['open','high','low','close','volume']
    df.dropna(inplace=True)
    return df

def compute_features(df):
    if df.empty: return df
    df = df.copy()
    df['ema_9']  = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']   = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr_14'] = df['tr'].rolling(14).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['up'] = df['high'] - df['high'].shift(1)
    df['down'] = df['low'].shift(1) - df['low']
    df['plus_dm']  = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
    df['minus_dm'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)
    df['plus_di']  = 100 * (df['plus_dm'].rolling(14).mean() / (df['atr_14'] + 1e-9))
    df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / (df['atr_14'] + 1e-9))
    df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-9)
    df['adx'] = df['dx'].rolling(14).mean()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(50).mean()
    df['trend'] = np.where(df['close'] > df['ema_200'], 1, -1)
    # هدف أكبر لتقليل الضوضاء (0.4% تحرك بدلاً من 0.05%)
    df['target'] = (df['close'].shift(-1)/df['close'] - 1 > 0.004).astype(int)
    df.dropna(inplace=True)
    return df

def detect_order_block(df, i, direction='bull'):
    if i < 5: return None
    if direction == 'bull':
        for j in range(i-1, max(i-20, 0), -1):
            if df['close'].iloc[j] < df['open'].iloc[j] and df['high'].iloc[j+1] > df['high'].iloc[j]:
                return df['high'].iloc[j]
    return None

df_5m  = compute_features(fetch_data('5m', 120))
df_4h  = compute_features(fetch_data('4h', 60))
df_1h  = compute_features(fetch_data('1h', 60))

if df_5m.empty or df_4h.empty:
    send_telegram("❌ لا توجد بيانات")
    raise SystemExit

features = ['ema_9','ema_21','macd','macd_signal','atr_14','adx','volume_ratio','trend','close']
lstm_features = ['close', 'ema_9', 'ema_21', 'macd', 'rsi', 'atr_14', 'adx', 'volume_ratio']

# XGBoost (قوي)
if os.path.exists(MODEL_XGB):
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(MODEL_XGB)
    xgb_model.fit(df_5m[features], df_5m['target'], xgb_model=xgb_model.get_booster())
else:
    xgb_model = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.03)
    xgb_model.fit(df_5m[features], df_5m['target'])
xgb_model.save_model(MODEL_XGB)

# LSTM (قوي)
if os.path.exists(MODEL_LSTM):
    lstm_model = tf.keras.models.load_model(MODEL_LSTM)
else:
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LSTM_LOOKBACK, len(lstm_features))),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_5m[lstm_features])
X_lstm, y_lstm = [], []
for i in range(LSTM_LOOKBACK, len(scaled)):
    X_lstm.append(scaled[i-LSTM_LOOKBACK:i])
    y_lstm.append(df_5m['target'].iloc[i])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
if len(X_lstm) > 10:
    early_stop = EarlyStopping(monitor='loss', patience=3, verbose=0)
    lstm_model.fit(X_lstm, y_lstm, epochs=20, batch_size=32, verbose=0, callbacks=[early_stop])
    lstm_model.save(MODEL_LSTM)

# RandomForest (قوي)
import joblib
if os.path.exists(MODEL_RF):
    rf_model = joblib.load(MODEL_RF)
    rf_model.fit(df_5m[features], df_5m['target'])
else:
    rf_model = RandomForestClassifier(n_estimators=500, max_depth=8)
    rf_model.fit(df_5m[features], df_5m['target'])
joblib.dump(rf_model, MODEL_RF)

# CatBoost (قوي)
if os.path.exists(MODEL_CAT):
    cat_model = CatBoostClassifier()
    cat_model.load_model(MODEL_CAT)
    cat_model.fit(df_5m[features], df_5m['target'], init_model=cat_model)
else:
    cat_model = CatBoostClassifier(iterations=500, depth=8, learning_rate=0.03, verbose=0)
    cat_model.fit(df_5m[features], df_5m['target'])
cat_model.save_model(MODEL_CAT)

capital = INITIAL_CAPITAL
position = 0
entry = 0
sl = 0
tp = 0
max_loss = 0
if os.path.exists(STATE_FILE):
    with open(STATE_FILE, 'r') as f:
        capital, position, entry, sl, tp = map(float, f.read().split(','))

i_5m = len(df_5m) - 1
latest_5m = df_5m.iloc[i_5m]
prob_xgb = xgb_model.predict_proba(latest_5m[features].values.reshape(1, -1))[0, 1]
prob_rf = rf_model.predict_proba(latest_5m[features].values.reshape(1, -1))[0, 1]
prob_cat = cat_model.predict_proba(latest_5m[features].values.reshape(1, -1))[0, 1]

if i_5m >= LSTM_LOOKBACK:
    lstm_input = df_5m[lstm_features].iloc[i_5m-LSTM_LOOKBACK:i_5m].values
    lstm_scaled = scaler.transform(lstm_input)
    prob_lstm = lstm_model.predict(lstm_scaled.reshape(1, LSTM_LOOKBACK, len(lstm_features)), verbose=0)[0,0]
else:
    prob_lstm = 0.5

prob = (prob_xgb + prob_lstm + prob_rf + prob_cat) / 4

price = latest_5m['close']
atr = max(latest_5m['atr_14'], 0.01*price)
trend_4h = 1 if price > df_4h['ema_200'].iloc[-1] else -1
adx_ok = df_1h['adx'].iloc[-1] > 22
volume_ok = df_1h['volume_ratio'].iloc[-1] > 0.8
macd_cross_up = latest_5m['macd'] > 0 and (df_5m['macd'].iloc[i_5m-1] if i_5m>0 else 0) <= 0
buy_signal = (trend_4h==1 and latest_5m['ema_9'] > latest_5m['ema_21'] and macd_cross_up and adx_ok and volume_ok and prob >= MIN_CONFIDENCE)
sell_signal = (trend_4h==-1 and latest_5m['ema_9'] < latest_5m['ema_21'] and latest_5m['macd'] < 0)

ob_bull = detect_order_block(df_5m, i_5m, 'bull')
if buy_signal and ob_bull and price <= ob_bull * 1.005:
    buy_signal = True
else:
    buy_signal = False

if position == 0 and buy_signal:
    stop_distance = STOP_ATR_MULT * atr
    max_loss = capital * RISK_PER_TRADE
    base_pos = max_loss / stop_distance if stop_distance > 0 else 0
    position = base_pos * LEVERAGE
    entry = price
    sl = price - stop_distance
    tp = price + TP_ATR_MULT * atr
    send_telegram(f"🥇 شراء ذهب (القوي)\nالسعر: {price:.2f}\nالوقف: {sl:.2f}\nالهدف: {tp:.2f}\nالرصيد: {capital:.2f}")
elif position > 0 and (price <= sl or price >= tp or sell_signal):
    pnl = position * (price - entry)
    if pnl < -max_loss: pnl = -max_loss
    capital += pnl
    send_telegram(f"🥇 إغلاق ذهب (القوي)\nالربح/الخسارة: {pnl:.2f}\nالرصيد: {capital:.2f}")
    position = 0

with open(STATE_FILE, 'w') as f:
    f.write(f"{capital},{position},{entry},{sl},{tp}")
with open(CAPITAL_FILE, 'w') as f:
    f.write(str(capital))

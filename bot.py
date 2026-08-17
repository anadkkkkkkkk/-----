import numpy as np, pandas as pd, datetime, time, os, requests, warnings, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from catboost import CatBoostClassifier
import yfinance as yf
warnings.filterwarnings('ignore')

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
CHAT_ID = os.environ.get('CHAT_ID', '7644255708')
if not TELEGRAM_TOKEN:
    print("TELEGRAM_TOKEN missing from Secrets"); raise SystemExit

def send_telegram(msg):
    try: requests.post(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',
                       data={'chat_id': CHAT_ID, 'text': msg}, timeout=10)
    except Exception as e:
        print(f"TG failed: {e}")

print("🧠 بوت الذهب - نظام التعلم المستمر")
send_telegram("🟢 بوت الذهب النهائي بدأ (خفيف وسريع)")

SYMBOL_YAHOO = "GC=F"
SYMBOLS_BINANCE = ["XAUUSDT", "PAXGUSDT"]
INITIAL_CAPITAL = 10000.0; RISK_PER_TRADE = 0.01; LEVERAGE = 5
STOP_ATR_MULT = 1.5; TP_ATR_MULT = 3.0; MIN_CONFIDENCE = 0.50
MODEL_XGB = 'gold_xgb.json'; MODEL_RF = 'gold_rf.pkl'; MODEL_CAT = 'gold_cat.cbm'
CAPITAL_FILE = 'capital_mtf.txt'; STATE_FILE = 'state.txt'
ACCURACY_FILE = 'accuracy_log.txt'

def yahoo(interval, days):
    try:
        end = datetime.datetime.now(); start = end - datetime.timedelta(days=days)
        df = yf.download(SYMBOL_YAHOO, start=start, end=end, interval=interval, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            df = df[['Open','High','Low','Close','Volume']].copy()
            df.columns = ['open','high','low','close','volume']
            df.dropna(inplace=True)
            if len(df) > 100: return df
    except Exception as e:
        print("Yahoo", interval, "failed:", e)
    return None

def binance(interval, limit):
    for sym in SYMBOLS_BINANCE:
        try:
            resp = requests.get("https://api.binance.com/api/v3/klines",
                                params={"symbol": sym, "interval": interval, "limit": limit},
                                headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data and len(data) > 50:
                    df = pd.DataFrame(data, columns=['time','open','high','low','close','volume','_','_','_','_','_','_'])
                    df = df[['time','open','high','low','close','volume']].astype(float)
                    df['time'] = pd.to_datetime(df['time'], unit='ms')
                    df.set_index('time', inplace=True)
                    return df
        except Exception as e:
            print("Binance", sym, interval, "failed:", e)
    return None

def get_df(interval, days, limit):
    df = yahoo(interval, days)
    if df is None:
        df = binance(interval, limit)
    return df

AGG = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}

df_5m = get_df('5m', 60, 1000)
LIVE_DATA = df_5m is not None
if df_5m is None:
    send_telegram("⚠️ استخدام بيانات تركيبية")
    np.random.seed(42)
    dates = pd.date_range(end=datetime.datetime.now(), periods=1000, freq='5min')
    close = 2600 + np.cumsum(np.random.randn(1000)*2)
    df_5m = pd.DataFrame({'open': close-1, 'high': close+2, 'low': close-2, 'close': close, 'volume': 1000}, index=dates)

df_1h = get_df('1h', 60, 500)
if df_1h is None: df_1h = df_5m.resample('1h').agg(AGG)
df_4h = df_1h.resample('4h').agg(AGG)

def compute_features(df):
    if df.empty or len(df) < 60: return df.iloc[0:0]
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
    df['volume_ratio'] = df['volume'] / (df['volume'].rolling(50).mean() + 1e-9)
    df['trend'] = np.where(df['close'] > df['ema_200'], 1, -1)
    # هدف متكيف مع ATR
    atr_pct = df['atr_14'] / df['close']
    target_threshold = atr_pct * 0.5
    df['target'] = (df['close'].shift(-3)/df['close'] - 1 > target_threshold).astype(int)
    df.dropna(inplace=True)
    return df

df_5m = compute_features(df_5m)
df_1h = compute_features(df_1h)
df_4h = compute_features(df_4h)

if len(df_5m) < 100:
    send_telegram("❌ فشل تحميل البيانات"); raise SystemExit

features = ['ema_9','ema_21','macd','macd_signal','atr_14','adx','volume_ratio','trend','close']

# تقسيم البيانات: تدريب (80%) + اختبار (20%) للتقييم
train_size = int(len(df_5m) * 0.8)
df_train = df_5m.iloc[:train_size]
df_test = df_5m.iloc[train_size:]

# تدريب XGBoost
if os.path.exists(MODEL_XGB):
    xgb_model = xgb.XGBClassifier(); xgb_model.load_model(MODEL_XGB)
    xgb_model.fit(df_5m[features], df_5m['target'], xgb_model=xgb_model.get_booster())
else:
    xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05)
    xgb_model.fit(df_train[features], df_train['target'])
xgb_pred = xgb_model.predict(df_test[features])
xgb_acc = accuracy_score(df_test['target'], xgb_pred)
xgb_model.save_model(MODEL_XGB)

# تدريب RandomForest
if os.path.exists(MODEL_RF):
    rf_model = joblib.load(MODEL_RF); rf_model.fit(df_5m[features], df_5m['target'])
else:
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=6)
    rf_model.fit(df_train[features], df_train['target'])
rf_pred = rf_model.predict(df_test[features])
rf_acc = accuracy_score(df_test['target'], rf_pred)
joblib.dump(rf_model, MODEL_RF)

# تدريب CatBoost
if os.path.exists(MODEL_CAT):
    cat_model = CatBoostClassifier(); cat_model.load_model(MODEL_CAT)
    cat_model.fit(df_5m[features], df_5m['target'], init_model=cat_model)
else:
    cat_model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0)
    cat_model.fit(df_train[features], df_train['target'])
cat_pred = cat_model.predict(df_test[features])
cat_acc = accuracy_score(df_test['target'], cat_pred)
cat_model.save_model(MODEL_CAT)

print(f"📊 Training Accuracy - XGB: {xgb_acc:.2%}, RF: {rf_acc:.2%}, CatBoost: {cat_acc:.2%}")

# حفظ سجل الدقة
timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
with open(ACCURACY_FILE, 'a') as f:
    f.write(f"{timestamp},{xgb_acc:.4f},{rf_acc:.4f},{cat_acc:.4f}\n")

# تقرير يومي (كل 24 ساعة)
hour = datetime.datetime.now().hour
if hour == 0 or not os.path.exists('last_report.txt'):
    try:
        with open(ACCURACY_FILE, 'r') as f:
            lines = f.readlines()
        if len(lines) >= 10:
            recent = lines[-24:] if len(lines) >= 24 else lines
            xgb_avg = np.mean([float(l.split(',')[1]) for l in recent])
            rf_avg = np.mean([float(l.split(',')[2]) for l in recent])
            cat_avg = np.mean([float(l.split(',')[3]) for l in recent])
            best = max([('XGBoost', xgb_avg), ('RandomForest', rf_avg), ('CatBoost', cat_avg)], key=lambda x: x[1])
            report = f"📈 تقرير التدريب اليومي\n\n"
            report += f"🏆 الأفضل: {best[0]} ({best[1]:.1%})\n\n"
            report += f"📊 متوسط الدقة:\n"
            report += f"  • XGBoost: {xgb_avg:.1%}\n"
            report += f"  • RandomForest: {rf_avg:.1%}\n"
            report += f"  • CatBoost: {cat_avg:.1%}\n\n"
            report += f"🔄 عدد نقاط التدريب: {len(recent)}"
            send_telegram(report)
            with open('last_report.txt', 'w') as f:
                f.write(timestamp)
    except Exception as e:
        print(f"Report failed: {e}")

# تحميل الحالة
capital = INITIAL_CAPITAL; position = 0; entry = 0; sl = 0; tp = 0; max_loss = 0
if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE, 'r') as f:
            capital, position, entry, sl, tp = map(float, f.read().split(','))
    except: pass

# الترجيح حسب الأداء
total_acc = xgb_acc + rf_acc + cat_acc + 1e-9
w_xgb = xgb_acc / total_acc
w_rf = rf_acc / total_acc
w_cat = cat_acc / total_acc

i_5m = len(df_5m) - 1
latest_5m = df_5m.iloc[i_5m]
prob_xgb = xgb_model.predict_proba(latest_5m[features].values.reshape(1, -1))[0, 1]
prob_rf = rf_model.predict_proba(latest_5m[features].values.reshape(1, -1))[0, 1]
prob_cat = cat_model.predict_proba(latest_5m[features].values.reshape(1, -1))[0, 1]

# احتمال مرجح حسب الأداء
prob = w_xgb * prob_xgb + w_rf * prob_rf + w_cat * prob_cat

price = latest_5m['close']
atr = max(latest_5m['atr_14'], 0.01*price)

adx_last = float(df_1h['adx'].iloc[-1]) if len(df_1h) > 0 else 0.0
vol_last = float(df_1h['volume_ratio'].iloc[-1]) if len(df_1h) > 0 else 0.0
ema50_4h = float(df_4h['ema_50'].iloc[-1]) if len(df_4h) > 0 else price
trend_4h = 1 if price > ema50_4h else -1

adx_ok = adx_last > 18
volume_ok = vol_last > 0.6

macd_cross_up = any(df_5m['macd'].iloc[k] > 0 and df_5m['macd'].iloc[k-1] <= 0
                    for k in range(max(1, i_5m-12), i_5m+1))
macd_cross_down = any(df_5m['macd'].iloc[k] < 0 and df_5m['macd'].iloc[k-1] >= 0
                      for k in range(max(1, i_5m-12), i_5m+1))

ema_trend_up = latest_5m['ema_9'] > latest_5m['ema_21']
ema_trend_down = latest_5m['ema_9'] < latest_5m['ema_21']

buy_signal = bool(LIVE_DATA and trend_4h == 1 and ema_trend_up and macd_cross_up and adx_ok and volume_ok and prob >= MIN_CONFIDENCE)
sell_signal = (trend_4h == -1 and ema_trend_down and macd_cross_down)

print(f"prob={prob:.2f} | trend4h={trend_4h} | ema_up={bool(ema_trend_up)} | cross_up={macd_cross_up} "
      f"| adx={adx_last:.1f} | vol={vol_last:.2f} | live={LIVE_DATA} | BUY={buy_signal}")

if position == 0 and buy_signal:
    stop_distance = STOP_ATR_MULT * atr
    max_loss = capital * RISK_PER_TRADE
    base_pos = max_loss / stop_distance if stop_distance > 0 else 0
    position = base_pos * LEVERAGE; entry = price
    sl = price - stop_distance; tp = price + TP_ATR_MULT * atr
    send_telegram(f"🥇 شراء ذهب\nالسعر: {price:.2f}\nالوقف: {sl:.2f}\nالهدف: {tp:.2f}\nالرصيد: {capital:.2f}")
elif position > 0 and (price <= sl or price >= tp or sell_signal):
    pnl = position * (price - entry)
    if pnl < -max_loss: pnl = -max_loss
    capital += pnl
    send_telegram(f"🥇 إغلاق ذهب\nPnL: {pnl:.2f}\nالرصيد: {capital:.2f}")
    position = 0

with open(STATE_FILE, 'w') as f:
    f.write(f"{capital},{position},{entry},{sl},{tp}")
with open(CAPITAL_FILE, 'w') as f:
    f.write(str(capital))

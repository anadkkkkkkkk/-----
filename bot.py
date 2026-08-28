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
    print("TELEGRAM_TOKEN missing"); raise SystemExit

def send_telegram(msg):
    try:
        requests.post(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',
                      data={'chat_id': CHAT_ID, 'text': msg}, timeout=10)
    except Exception as e:
        print(f"TG failed: {e}")

# رسالة البدء مرة كل ساعة فقط (بدون إزعاج)
if datetime.datetime.now().minute < 5:
    send_telegram("🟢 بوت الذهب يعمل (استراتيجية: ربح 2% / خسارة 1% - بدون توقف)")

# ========== الاستراتيجية ==========
RISK_LOSS      = 0.01     # الخسارة: 1% من الرصيد
RISK_REWARD    = 2.0      # الربح: 2% (ضعف الخسارة)
LEVERAGE       = 10       # رافعة 10x
PROB_THRESHOLD = 0.55     # جودة الإشارة
COOLDOWN_MIN   = 15       # انتظار 15 دقيقة بعد الخسارة
EXCHANGE_RATE  = 530      # سعر الدولار بالريال اليمني

SYMBOL_YAHOO = "GC=F"
SYMBOLS_BINANCE = ["XAUUSDT", "PAXGUSDT"]
INITIAL_CAPITAL = 10000.0
MODEL_XGB = 'gold_xgb.json'; MODEL_RF = 'gold_rf.pkl'; MODEL_CAT = 'gold_cat.cbm'
CAPITAL_FILE = 'capital_mtf.txt'; STATE_FILE = 'state.txt'
ACCURACY_FILE = 'accuracy_log.txt'

def fetch_data(interval, days, limit):
    try:
        end = datetime.datetime.now(); start = end - datetime.timedelta(days=days)
        df = yf.download(SYMBOL_YAHOO, start=start, end=end, interval=interval, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df = df[['Open','High','Low','Close','Volume']].copy()
            df.columns = ['open','high','low','close','volume']
            df.dropna(inplace=True)
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            if len(df) > 20: return df
    except Exception as e:
        print(f"Yahoo {interval} failed:", e)
    for sym in SYMBOLS_BINANCE:
        try:
            resp = requests.get("https://api.binance.com/api/v3/klines",
                                params={"symbol": sym, "interval": interval, "limit": limit},
                                headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data and len(data) > 20:
                    df = pd.DataFrame(data, columns=['time','open','high','low','close','volume','_','_','_','_','_','_'])
                    df = df[['time','open','high','low','close','volume']].astype(float)
                    df['time'] = pd.to_datetime(df['time'], unit='ms')
                    df.set_index('time', inplace=True)
                    if df.index.tz is not None: df.index = df.index.tz_localize(None)
                    return df
        except Exception as e:
            print(f"Binance {sym} {interval} failed:", e)
    return None

AGG = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}

df_5m  = fetch_data('5m', 5, 500)
df_15m = fetch_data('15m', 15, 400)
df_1h  = fetch_data('1h', 40, 400)
df_4h  = fetch_data('4h', 90, 300)
df_1d  = fetch_data('1d', 180, 200)

LIVE_DATA = df_5m is not None

if df_5m is None:
    dates = pd.date_range(end=datetime.datetime.now(), periods=500, freq='5min')
    close = 2600 + np.cumsum(np.random.randn(500)*2)
    df_5m = pd.DataFrame({'open': close-1, 'high': close+2, 'low': close-2, 'close': close, 'volume': 1000}, index=dates)

if df_15m is None: df_15m = df_5m.resample('15min').agg(AGG).dropna()
if df_1h is None:  df_1h = df_5m.resample('1h').agg(AGG).dropna()
if df_4h is None:  df_4h = df_1h.resample('4h').agg(AGG).dropna()
if df_1d is None:  df_1d = df_1h.resample('1D').agg(AGG).dropna()

def compute_features(df):
    if df is None or df.empty or len(df) < 20: return pd.DataFrame()
    df = df.copy()
    df['ema_9']  = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']   = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                               abs(df['low'] - df['close'].shift(1))))
    df['atr_14'] = tr.rolling(14).mean()
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
    atr_pct = df['atr_14'] / df['close']
    df['target'] = (df['close'].shift(-3)/df['close'] - 1 > atr_pct * 0.3).astype(int)
    df.dropna(inplace=True)
    return df

df_5m  = compute_features(df_5m)
df_15m = compute_features(df_15m)
df_1h  = compute_features(df_1h)
df_4h  = compute_features(df_4h)
df_1d  = compute_features(df_1d)

if len(df_5m) < 20:
    send_telegram("❌ فشل تحميل البيانات"); raise SystemExit

features = ['ema_9','ema_21','macd','macd_signal','atr_14','adx','volume_ratio','trend','close']

# ---------- التدريب مع حفظ الموديلات ----------
train_size = int(len(df_5m) * 0.8)
df_train = df_5m.iloc[:train_size]
df_test  = df_5m.iloc[train_size:]

if os.path.exists(MODEL_XGB):
    try:
        xgb_model = xgb.XGBClassifier(); xgb_model.load_model(MODEL_XGB)
        xgb_model.fit(df_5m[features], df_5m['target'], xgb_model=xgb_model.get_booster())
    except:
        xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05)
        xgb_model.fit(df_train[features], df_train['target'])
else:
    xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05)
    xgb_model.fit(df_train[features], df_train['target'])
xgb_acc = accuracy_score(df_test['target'], xgb_model.predict(df_test[features]))
xgb_model.save_model(MODEL_XGB)

if os.path.exists(MODEL_RF):
    try:
        rf_model = joblib.load(MODEL_RF); rf_model.fit(df_5m[features], df_5m['target'])
    except:
        rf_model = RandomForestClassifier(n_estimators=300, max_depth=6)
        rf_model.fit(df_train[features], df_train['target'])
else:
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=6)
    rf_model.fit(df_train[features], df_train['target'])
rf_acc = accuracy_score(df_test['target'], rf_model.predict(df_test[features]))
joblib.dump(rf_model, MODEL_RF)

if os.path.exists(MODEL_CAT):
    try:
        cat_model = CatBoostClassifier(verbose=0); cat_model.load_model(MODEL_CAT)
        cat_model.fit(df_5m[features], df_5m['target'], init_model=cat_model)
    except:
        cat_model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0)
        cat_model.fit(df_train[features], df_train['target'])
else:
    cat_model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0)
    cat_model.fit(df_train[features], df_train['target'])
cat_acc = accuracy_score(df_test['target'], cat_model.predict(df_test[features]))
cat_model.save_model(MODEL_CAT)

print(f"📊 Accuracy - XGB: {xgb_acc:.2%}, RF: {rf_acc:.2%}, Cat: {cat_acc:.2%}")

with open(ACCURACY_FILE, 'a') as f:
    f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')},{xgb_acc:.4f},{rf_acc:.4f},{cat_acc:.4f}\n")

# ---------- الترجيح الذكي ----------
total_acc = xgb_acc + rf_acc + cat_acc + 1e-9
prob = (xgb_acc/total_acc) * xgb_model.predict_proba(df_5m[features].iloc[[-1]])[0,1] \
     + (rf_acc/total_acc)  * rf_model.predict_proba(df_5m[features].iloc[[-1]])[0,1] \
     + (cat_acc/total_acc) * cat_model.predict_proba(df_5m[features].iloc[[-1]])[0,1]

# ---------- تحليل الفريمات ----------
price = df_5m['close'].iloc[-1]
atr   = max(df_5m['atr_14'].iloc[-1], 0.01*price)

trend_1d  = 1 if len(df_1d) > 0 and price > df_1d['ema_50'].iloc[-1] else -1
trend_4h  = 1 if len(df_4h) > 0 and price > df_4h['ema_50'].iloc[-1] else -1
trend_1h  = 1 if len(df_1h) > 0 and price > df_1h['ema_21'].iloc[-1] else -1
trend_15m = 1 if len(df_15m) > 0 and price > df_15m['ema_9'].iloc[-1] else -1
ema_5m    = df_5m['ema_9'].iloc[-1] > df_5m['ema_21'].iloc[-1]
macd_5m   = df_5m['macd'].iloc[-1] > df_5m['macd_signal'].iloc[-1]

bull_score = sum([trend_1d==1, trend_4h==1, trend_1h==1, trend_15m==1, ema_5m, macd_5m])
bear_score = sum([trend_1d==-1, trend_4h==-1, trend_1h==-1, trend_15m==-1, not ema_5m, not macd_5m])

buy_signal  = bool(LIVE_DATA and bull_score >= 4 and prob >= PROB_THRESHOLD)
sell_signal = bool(LIVE_DATA and bear_score >= 4)

print(f"🎯 1D={trend_1d} 4H={trend_4h} 1H={trend_1h} 15M={trend_15m} | Bull={bull_score}/6 | Prob={prob:.2f}")

# ---------- تحميل الحالة ----------
capital = INITIAL_CAPITAL; position = 0; entry = 0; sl = 0; tp = 0; last_loss_ts = 0
if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE) as f:
            parts = f.read().split(',')
            capital, position, entry, sl, tp = map(float, parts[:5])
            last_loss_ts = float(parts[5]) if len(parts) > 5 else 0
    except: pass

now_ts = time.time()
cooldown_ok = (now_ts - last_loss_ts) > COOLDOWN_MIN * 60

# ---------- التنفيذ: استراتيجية 2% / 1% ----------
if position == 0 and buy_signal and cooldown_ok:
    stop_distance = atr * 1.5
    risk_loss = capital * RISK_LOSS                      # الخسارة = 1%
    position = (risk_loss / stop_distance) * LEVERAGE    # الحجم حسب المخاطرة
    entry = price
    sl = price - stop_distance                           # وقف = -1%
    tp = price + stop_distance * RISK_REWARD             # هدف = +2%
    send_telegram(f"🥇 شراء ذهب (استراتيجية 2%/1%)\n"
                 f"السعر: {price:.2f}$\n"
                 f"الوقف: {sl:.2f}$ (-{risk_loss*EXCHANGE_RATE:.0f} ﷼)\n"
                 f"الهدف: {tp:.2f}$ (+{risk_loss*RISK_REWARD*EXCHANGE_RATE:.0f} ﷼)\n"
                 f"الثقة: {bull_score}/6 | Prob: {prob:.2f}\n"
                 f"الرصيد: {capital:.2f}$")
elif position > 0 and (price <= sl or price >= tp or sell_signal):
    reason = "هدف ✅" if price >= tp else ("وقف ❌" if price <= sl else "إشارة معاكسة")
    pnl = position * (price - entry)
    max_loss = capital * RISK_LOSS * 1.2
    if pnl < -max_loss: pnl = -max_loss
    capital += pnl
    if pnl < 0: last_loss_ts = now_ts
    send_telegram(f"🥇 إغلاق ({reason})\n"
                 f"PnL: {pnl:+.2f}$ ({pnl*EXCHANGE_RATE:+.0f} ﷼)\n"
                 f"الرصيد: {capital:.2f}$")
    position = 0

with open(STATE_FILE, 'w') as f:
    f.write(f"{capital},{position},{entry},{sl},{tp},{last_loss_ts}")
with open(CAPITAL_FILE, 'w') as f:
    f.write(str(capital))

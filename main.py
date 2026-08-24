import ccxt
import pandas as pd
import numpy as np

# ========== 1. إعدادات حساب OKX ومفاتيحه ==========
API_KEY = "22e91e96-99bc-4d0c-9f9c-2679bd7c6df5"
SECRET_KEY = "F30B08A12BD7DA258B3B706C57B939F0"
PASSPHRASE = "F30B08A12BD7DA258B3B706C57B939F0"  # استخدام كلمة المرور المعطاة كـ Passphrase أيضاً احتياطياً

# الاتصال بمنصة OKX (عقود السواب Futures)
exchange = ccxt.okx({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'password': PASSPHRASE,
    'options': {'defaultType': 'swap'},
    'enableRateLimit': True,
})

SYMBOL = 'BTC/USDT:USDT'  # زوج التداول على OKX
LEVERAGE = 10             # الرافعة المالية
RISK_REWARD = 2.0         # نسبة الهدف إلى الوقف (1:2)
PROB_THRESHOLD = 0.60     # عتبة فلتر النموذج للتنفيذ (60%)

# ضبط الرافعة المالية في OKX
try:
    exchange.set_leverage(LEVERAGE, 'BTC/USDT')
    print(f"✅ تم ضبط الرافعة المالية في OKX إلى {LEVERAGE}x")
except Exception as e:
    print(f"⚠️ ملاحظة الرافعة: {e}")

# ========== 2. جلب البيانات ==========
def fetch_candles(symbol, timeframe='15m', limit=250):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# ========== 3. استراتيجية BRK + EMA200 ==========
def analyze_market(df):
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    left, right = 3, 3
    df['PH'] = np.nan
    df['PL'] = np.nan
    
    for i in range(left, len(df) - right):
        if df['high'].iloc[i] == df['high'].iloc[i-left:i+right+1].max():
            df.iloc[i, df.columns.get_loc('PH')] = df['high'].iloc[i]
        if df['low'].iloc[i] == df['low'].iloc[i-left:i+right+1].min():
            df.iloc[i, df.columns.get_loc('PL')] = df['low'].iloc[i]
            
    i = len(df) - 1
    past_ph = df['PH'].iloc[:i].dropna()
    past_pl = df['PL'].iloc[:i].dropna()
    
    if len(past_ph) < 2 or len(past_pl) < 2:
        return None, 0, 0
        
    last_ph = past_ph.iloc[-1]
    last_pl = past_pl.iloc[-1]
    close = df['close'].iloc[i]
    ema = df['EMA200'].iloc[i]
    
    # إشارة شراء (Long)
    if close > ema and close > last_ph and df['close'].iloc[i-1] <= last_ph:
        sl = df['low'].iloc[i-2:i+1].min()
        risk = close - sl
        if 1.5 <= risk <= 12:
            return 'BUY', sl, close + (risk * RISK_REWARD)

    # إشارة بيع (Short)
    elif close < ema and close < last_pl and df['close'].iloc[i-1] >= last_pl:
        sl = df['high'].iloc[i-2:i+1].max()
        risk = sl - close
        if 1.5 <= risk <= 12:
            return 'SELL', sl, close - (risk * RISK_REWARD)

    return None, 0, 0

# ========== 4. نموذج فلتر الاحتمالية ==========
def get_model_probability(df, signal):
    if not signal:
        return 0.0
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Vol_Ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    last = df.iloc[-1]
    prob = 0.50
    
    if signal == 'BUY':
        if 50 <= last['RSI'] <= 68: prob += 0.20
        if last['Vol_Ratio'] > 1.2: prob += 0.15
    elif signal == 'SELL':
        if 32 <= last['RSI'] <= 50: prob += 0.20
        if last['Vol_Ratio'] > 1.2: prob += 0.15
        
    return min(round(prob, 2), 0.95)

# ========== 5. تنفيذ الصفقات على OKX ==========
def execute_trade(signal, sl, tp, amount=0.01):
    try:
        side = 'buy' if signal == 'BUY' else 'sell'
        sl_side = 'sell' if signal == 'BUY' else 'buy'
        
        print(f"🚀 تنفيذ صفقة {signal} على OKX...")
        exchange.create_order(SYMBOL, 'market', side, amount)
        exchange.create_order(SYMBOL, 'stop_market', sl_side, amount, params={'stopPrice': sl})
        exchange.create_order(SYMBOL, 'take_profit_market', sl_side, amount, params={'stopPrice': tp})
        print("✅ تم تنفيذ أوامر OKX بنجاح.")
    except Exception as e:
        print(f"❌ خطأ التنفيذ في OKX: {e}")

# ========== 6. تشغيل البوت ==========
def run_bot():
    print("🤖 جاري فحص السوق على منصة OKX...")
    try:
        df = fetch_candles(SYMBOL)
        current_hour = pd.Timestamp.now('UTC').hour
        
        if 12 <= current_hour <= 19:
            signal, sl, tp = analyze_market(df)
            prob = get_model_probability(df, signal)
            
            print(f"📊 الإشارة: {signal if signal else 'NONE'} | الاحتمالية: {prob*100}%")
            
            if signal and prob >= PROB_THRESHOLD:
                print("🚀 الاحتمالية تتجاوز الحد الأدنى، جاري الإرسال لمنصة OKX...")
                execute_trade(signal, sl, tp)
            else:
                print("⏳ لم تتطابق الشروط أو الاحتمالية منخفضة.")
        else:
            print("⏳ خارج أوقات الجلسة المحددة (12:00 - 19:00 UTC).")
    except Exception as e:
        print(f"⚠️ خطأ في البوت: {e}")

if __name__ == "__main__":
    run_bot()

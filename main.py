import os
import ccxt
import pandas as pd
import numpy as np

# 1. الإعدادات والمفاتيح
API_KEY = os.getenv("BINGX_API_KEY", "")
SECRET_KEY = os.getenv("BINGX_SECRET_KEY", "")

exchange = ccxt.bingx({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'options': {'defaultType': 'swap'},
    'enableRateLimit': True,
})

SYMBOL = 'GOLD/USDT'
LEVERAGE = 10
RISK_REWARD = 2.0
PROB_THRESHOLD = 0.60  # عتبة فلتر النموذج للتنفيذ (60%)

try:
    exchange.set_leverage(LEVERAGE, SYMBOL)
except Exception as e:
    pass

def fetch_candles(symbol, timeframe='15m', limit=250):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# 2. استراتيجية BRK + EMA200
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
    
    if close > ema and close > last_ph and df['close'].iloc[i-1] <= last_ph:
        sl = df['low'].iloc[i-2:i+1].min()
        risk = close - sl
        if 1.5 <= risk <= 12:
            return 'BUY', sl, close + (risk * RISK_REWARD)

    elif close < ema and close < last_pl and df['close'].iloc[i-1] >= last_pl:
        sl = df['high'].iloc[i-2:i+1].max()
        risk = sl - close
        if 1.5 <= risk <= 12:
            return 'SELL', sl, close - (risk * RISK_REWARD)

    return None, 0, 0

# 3. نموذج فلتر الاحتمالية (Probability Model)
def get_model_probability(df, signal):
    if not signal:
        return 0.0
    
    # حساب المؤشرات الداعمة للنموذج
    df['RSI'] = 100 - (100 / (1 + (df['close'].diff().clip(lower=0).rolling(14).mean() / (-df['close'].diff().clip(upper=0).rolling(14).mean()))))
    df['Vol_Ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    last = df.iloc[-1]
    prob = 0.50
    
    # فلتر الزخم وحجم التداول
    if signal == 'BUY':
        if 50 <= last['RSI'] <= 68: prob += 0.20
        if last['Vol_Ratio'] > 1.2: prob += 0.15
    elif signal == 'SELL':
        if 32 <= last['RSI'] <= 50: prob += 0.20
        if last['Vol_Ratio'] > 1.2: prob += 0.15
        
    return min(round(prob, 2), 0.95)

def execute_trade(signal, sl, tp, amount=0.01):
    try:
        side = 'buy' if signal == 'BUY' else 'sell'
        sl_side = 'sell' if signal == 'BUY' else 'buy'
        
        exchange.create_order(SYMBOL, 'market', side, amount)
        exchange.create_order(SYMBOL, 'STOP_MARKET', sl_side, amount, params={'stopPrice': sl})
        exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', sl_side, amount, params={'stopPrice': tp})
        print(f"✅ تم تنفيذ صفقة {signal} بنجاح.")
    except Exception as e:
        print(f"❌ خطأ التنفيذ: {e}")

def run_bot():
    print("🤖 جاري فحص استراتيجية BRK + فلتر النموذج...")
    try:
        df = fetch_candles(SYMBOL)
        current_hour = pd.Timestamp.now('UTC').hour
        
        if 12 <= current_hour <= 19:
            signal, sl, tp = analyze_market(df)
            prob = get_model_probability(df, signal)
            
            print(f"📊 الإشارة: {signal if signal else 'NONE'} | احتمالية النموذج: {prob*100}%")
            
            if signal and prob >= PROB_THRESHOLD:
                print("🚀 الاحتمالية أعلى من العتبة، جاري فتح الصفقة...")
                execute_trade(signal, sl, tp)
            else:
                print("⏳ لم تتجاوز الشروط أو الاحتمالية حد العتبة المطلوب.")
        else:
            print("⏳ خارج أوقات الجلسة (12:00 - 19:00 UTC).")
    except Exception as e:
        print(f"⚠️ خطأ: {e}")

if __name__ == "__main__":
    run_bot()

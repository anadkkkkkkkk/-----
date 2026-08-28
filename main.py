import ccxt
import pandas as pd
import numpy as np
import os
import time
import requests

# ---------- إعدادات من Secrets ----------
API_KEY = os.environ.get('OKX_API_KEY', '')
SECRET_KEY = os.environ.get('OKX_SECRET_KEY', '')
PASSPHRASE = os.environ.get('OKX_PASSPHRASE', '')
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

if not all([API_KEY, SECRET_KEY, PASSPHRASE]):
    print("❌ OKX keys missing"); raise SystemExit

def send_telegram(msg):
    if not TELEGRAM_TOKEN: return
    try:
        requests.post(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',
                      data={'chat_id': CHAT_ID, 'text': msg}, timeout=10)
    except Exception as e:
        print(f"TG failed: {e}")

# ---------- الاتصال بـ OKX Demo ----------
exchange = ccxt.okx({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'password': PASSPHRASE,
    'options': {'defaultType': 'swap'},
    'enableRateLimit': True,
})
exchange.set_sandbox_mode(True)

# ---------- إعدادات التداول ----------
SYMBOL = 'XAU/USDT:USDT'  # الذهب - إذا فشل نستخدم PAXG
ALT_SYMBOL = 'PAXG/USDT:USDT'  # بديل ذهبي
LEVERAGE = 5
RISK_REWARD = 2.5
PROB_THRESHOLD = 0.55
RISK_PER_TRADE = 0.02  # 2% من الرصيد لكل صفقة

# محاولة الذهب أولاً
try:
    exchange.load_markets()
    if SYMBOL in exchange.markets:
        TRADING_SYMBOL = SYMBOL
    elif ALT_SYMBOL in exchange.markets:
        TRADING_SYMBOL = ALT_SYMBOL
        print(f"⚠️ استخدام {ALT_SYMBOL} بدل {SYMBOL}")
    else:
        TRADING_SYMBOL = 'BTC/USDT:USDT'
        print(f"⚠️ استخدام BTC كبديل")
except:
    TRADING_SYMBOL = 'BTC/USDT:USDT'

try:
    balance = exchange.fetch_balance()
    usdt_balance = float(balance['USDT']['total']) if 'USDT' in balance else 0
    print(f"✅ اتصال ناجح | الرصيد: {usdt_balance:.2f} USDT | الرمز: {TRADING_SYMBOL}")
    exchange.set_leverage(LEVERAGE, TRADING_SYMBOL)
    send_telegram(f"🟢 بوت OKX التجريبي متصل\nالرمز: {TRADING_SYMBOL}\nالرصيد: {usdt_balance:.2f} USDT\nالرافعة: {LEVERAGE}x")
except Exception as e:
    print(f"❌ خطأ الاتصال: {e}")
    raise SystemExit

def fetch_candles(symbol, timeframe='15m', limit=250):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def analyze_market(df):
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    
    # حساب RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # حساب ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                               abs(df['low'] - df['close'].shift(1))))
    df['ATR'] = tr.rolling(14).mean()
    
    # Volume ratio
    df['Vol_Ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # PH/PL
    left, right = 3, 3
    df['PH'] = np.nan
    df['PL'] = np.nan
    for i in range(left, len(df) - right):
        if df['high'].iloc[i] == df['high'].iloc[i-left:i+right+1].max():
            df.iloc[i, df.columns.get_loc('PH')] = df['high'].iloc[i]
        if df['low'].iloc[i] == df['low'].iloc[i-left:i+right+1].min():
            df.iloc[i, df.columns.get_loc('PL')] = df['low'].iloc[i]
    
    i = len(df) - 1
    last = df.iloc[i]
    prev = df.iloc[i-1]
    
    close = last['close']
    ema200 = last['EMA200']
    ema50 = last['EMA50']
    rsi = last['RSI']
    atr = last['ATR']
    
    # تحديد الإشارة
    signal = None
    sl = 0
    tp = 0
    
    # شراء: فوق EMA200 + كسر قمة حديثة + RSI مناسب
    if close > ema200 and last['EMA9'] > last['EMA21']:
        past_ph = df['PH'].iloc[:i].dropna()
        if len(past_ph) > 0:
            last_ph = past_ph.iloc[-1]
            if close > last_ph and prev['close'] <= last_ph:
                sl = last['low'] - atr * 0.5
                risk = close - sl
                if risk > 0:
                    signal = 'BUY'
                    tp = close + (risk * RISK_REWARD)
    
    # بيع: تحت EMA200 + كسر قاع حديث + RSI مناسب
    elif close < ema200 and last['EMA9'] < last['EMA21']:
        past_pl = df['PL'].iloc[:i].dropna()
        if len(past_pl) > 0:
            last_pl = past_pl.iloc[-1]
            if close < last_pl and prev['close'] >= last_pl:
                sl = last['high'] + atr * 0.5
                risk = sl - close
                if risk > 0:
                    signal = 'SELL'
                    tp = close - (risk * RISK_REWARD)
    
    return signal, sl, tp, close, atr, rsi, last['Vol_Ratio']

def get_probability(signal, rsi, vol_ratio, atr, close):
    if not signal:
        return 0.0
    prob = 0.50
    if signal == 'BUY':
        if 45 <= rsi <= 65: prob += 0.15
        if rsi < 40: prob -= 0.10
        if vol_ratio > 1.3: prob += 0.15
        if vol_ratio > 1.8: prob += 0.10
    elif signal == 'SELL':
        if 35 <= rsi <= 55: prob += 0.15
        if rsi > 60: prob -= 0.10
        if vol_ratio > 1.3: prob += 0.15
        if vol_ratio > 1.8: prob += 0.10
    return min(round(prob, 2), 0.95)

def get_open_position():
    """جلب الصفقة المفتوحة"""
    try:
        positions = exchange.fetch_positions([TRADING_SYMBOL])
        for pos in positions:
            contracts = float(pos.get('contracts', 0) or 0)
            if contracts > 0:
                return {
                    'side': pos.get('side'),
                    'size': contracts,
                    'entry': float(pos.get('entryPrice', 0)),
                    'unrealizedPnl': float(pos.get('unrealizedPnl', 0))
                }
    except Exception as e:
        print(f"Error fetching positions: {e}")
    return None

def close_position(position, reason):
    """إغلاق الصفقة"""
    try:
        side = 'sell' if position['side'] == 'long' else 'buy'
        order = exchange.create_order(
            TRADING_SYMBOL,
            'market',
            side,
            position['size'],
            params={'reduceOnly': True}
        )
        pnl = position['unrealizedPnl']
        msg = f"🥇 إغلاق صفقة {TRADING_SYMBOL}\nالسبب: {reason}\nPnL: {pnl:.2f} USDT"
        print(msg)
        send_telegram(msg)
        return True
    except Exception as e:
        print(f"Error closing: {e}")
        return False

def open_position(signal, sl, tp, close, atr):
    """فتح صفقة جديدة"""
    try:
        balance = exchange.fetch_balance()
        usdt = float(balance['USDT']['total']) if 'USDT' in balance else 1000
        
        # حساب الحجم بناءً على المخاطرة
        risk_amount = usdt * RISK_PER_TRADE
        risk_per_unit = abs(close - sl)
        if risk_per_unit == 0:
            return
        
        # حجم الصفقة
        size = (risk_amount / risk_per_unit) * LEVERAGE
        min_size = exchange.market(TRADING_SYMBOL).get('limits', {}).get('amount', {}).get('min', 0.01)
        size = max(size, min_size * 2)
        
        side = 'buy' if signal == 'BUY' else 'sell'
        
        order = exchange.create_order(TRADING_SYMBOL, 'market', side, size)
        
        msg = f"🥇 صفقة {signal} جديدة\nالرمز: {TRADING_SYMBOL}\nالسعر: {close:.2f}\nالوقف: {sl:.2f}\nالهدف: {tp:.2f}\nالحجم: {size:.4f}\nالرافعة: {LEVERAGE}x"
        print(msg)
        send_telegram(msg)
        
        # وضع أوامر SL/TP
        try:
            if signal == 'BUY':
                exchange.create_order(TRADING_SYMBOL, 'stop', 'sell', size, sl, params={'stopPrice': sl, 'reduceOnly': True})
                exchange.create_order(TRADING_SYMBOL, 'take_profit', 'sell', size, tp, params={'takeProfitPrice': tp, 'reduceOnly': True})
            else:
                exchange.create_order(TRADING_SYMBOL, 'stop', 'buy', size, sl, params={'stopPrice': sl, 'reduceOnly': True})
                exchange.create_order(TRADING_SYMBOL, 'take_profit', 'buy', size, tp, params={'takeProfitPrice': tp, 'reduceOnly': True})
        except Exception as e:
            print(f"SL/TP order failed: {e}")
        
    except Exception as e:
        print(f"Error opening: {e}")
        send_telegram(f"⚠️ فشل فتح صفقة: {e}")

def run_bot():
    print("🤖 فحص السوق...")
    try:
        df = fetch_candles(TRADING_SYMBOL)
        signal, sl, tp, close, atr, rsi, vol_ratio = analyze_market(df)
        prob = get_probability(signal, rsi, vol_ratio, atr, close)
        
        position = get_open_position()
        
        print(f"📊 {TRADING_SYMBOL} | Price: {close:.2f} | RSI: {rsi:.1f} | Vol: {vol_ratio:.2f}")
        print(f"🎯 Signal: {signal or 'NONE'} | Prob: {prob*100:.0f}% | SL: {sl:.2f} | TP: {tp:.2f}")
        
        # إدارة الصفقة المفتوحة
        if position:
            if signal and signal != ('BUY' if position['side'] == 'long' else 'SELL'):
                # الإشارة عكس الصفقة → إغلاق
                close_position(position, f"إشارة معاكسة ({signal})")
        else:
            # لا توجد صفقة → افتح جديدة إذا الشروط مناسبة
            if signal and prob >= PROB_THRESHOLD:
                open_position(signal, sl, tp, close, atr)
            else:
                print(f"⏳ لا توجد صفقة: signal={signal}, prob={prob}, threshold={PROB_THRESHOLD}")
        
    except Exception as e:
        print(f"⚠️ خطأ: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_bot()

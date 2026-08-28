import ccxt, pandas as pd, numpy as np, os, requests, json, datetime

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

# ========== استراتيجية 5 دولار ==========
RISK_PER_TRADE = 0.02     # 2% = 0.10 دولار لكل صفقة
RISK_REWARD    = 2.5      # العائد:المخاطرة
LEVERAGE       = 5        # نبدأ بـ 5x (أكثر أمانًا من 10x)
FEE_RATE       = 0.0005   # 0.05% لكل جانب
MAX_TRADES_DAY = 2
PROB_THRESHOLD = 0.55     # خففنا قليلًا للحسابات الصغيرة
STATE_FILE     = 'okx_state.json'

exchange = ccxt.okx({
    'apiKey': API_KEY, 'secret': SECRET_KEY, 'password': PASSPHRASE,
    'options': {'defaultType': 'swap'}, 'enableRateLimit': True,
})
exchange.set_sandbox_mode(True)  # Demo Mode

# اختيار الرمز المناسب لحساب صغير
# PAXG (ذهب رقمي) أو BTC - كلاهما يسمح بأحجام صغيرة جدًا
try:
    exchange.load_markets()
    # الأولوية: ذهب رقمي → بيتكوين → إثيريوم
    candidates = ['PAXG/USDT:USDT', 'BTC/USDT:USDT', 'ETH/USDT:USDT']
    SYMBOL = None
    for sym in candidates:
        if sym in exchange.markets:
            SYMBOL = sym
            break
    if not SYMBOL:
        SYMBOL = 'BTC/USDT:USDT'
except:
    SYMBOL = 'BTC/USDT:USDT'

try:
    bal = exchange.fetch_balance()
    usdt = float(bal['USDT']['total']) if 'USDT' in bal else 0
    exchange.set_leverage(LEVERAGE, SYMBOL)
    send_telegram(f"🟢 بوت الـ 5$ متصل\nالرمز: {SYMBOL}\nالرصيد: {usdt:.2f} USDT (≈ {usdt*530:.0f} ﷼)\nالرافعة: {LEVERAGE}x | مخاطرة {RISK_PER_TRADE*100}%")
except Exception as e:
    print(f"❌ فشل: {e}"); raise SystemExit

today = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
state = {'date': today, 'trades': 0, 'wins': 0, 'losses': 0, 'balance': 0}
if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE) as f: st = json.load(f)
        if st.get('date') == today: state = st
    except: pass

def fetch_candles(tf='15m', limit=250):
    o = exchange.fetch_ohlcv(SYMBOL, timeframe=tf, limit=limit)
    df = pd.DataFrame(o, columns=['t','open','high','low','close','volume'])
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    return df

def analyze(df):
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['EMA50']  = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA21']  = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA9']   = df['close'].ewm(span=9, adjust=False).mean()
    d = df['close'].diff()
    g = d.where(d>0,0).rolling(14).mean()
    l = (-d.where(d<0,0)).rolling(14).mean()
    df['RSI'] = 100 - (100/(1+(g/l)))
    tr = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1))))
    df['ATR'] = tr.rolling(14).mean()
    df['VolR'] = df['volume']/df['volume'].rolling(20).mean()
    L=R=3; df['PH']=np.nan; df['PL']=np.nan
    for i in range(L, len(df)-R):
        if df['high'].iloc[i]==df['high'].iloc[i-L:i+R+1].max(): df.iloc[i, df.columns.get_loc('PH')]=df['high'].iloc[i]
        if df['low'].iloc[i]==df['low'].iloc[i-L:i+R+1].min(): df.iloc[i, df.columns.get_loc('PL')]=df['low'].iloc[i]
    return df

def signal_and_prob(df):
    i = len(df)-1
    last, prev = df.iloc[i], df.iloc[i-1]
    c, atr, rsi, vol = last['close'], last['ATR'], last['RSI'], last['VolR']
    sig = sl = tp = None

    # شراء
    if c > last['EMA200'] and last['EMA9'] > last['EMA21']:
        ph = df['PH'].iloc[:i].dropna()
        if len(ph) and c > ph.iloc[-1] and prev['close'] <= ph.iloc[-1]:
            sl = last['low'] - atr*0.5
            risk = c - sl
            if risk > 0:
                sig = 'BUY'
                tp = c + risk*RISK_REWARD + c*FEE_RATE*2*LEVERAGE
    # بيع
    elif c < last['EMA200'] and last['EMA9'] < last['EMA21']:
        pl = df['PL'].iloc[:i].dropna()
        if len(pl) and c < pl.iloc[-1] and prev['close'] >= pl.iloc[-1]:
            sl = last['high'] + atr*0.5
            risk = sl - c
            if risk > 0:
                sig = 'SELL'
                tp = c - risk*RISK_REWARD - c*FEE_RATE*2*LEVERAGE

    if not sig: return None, 0, 0, 0, c, atr

    # حساب الاحتمال
    prob = 0.50
    if sig=='BUY':
        if 45<=rsi<=65: prob+=0.15
        if vol>1.3: prob+=0.15
        if vol>1.8: prob+=0.10
    else:
        if 35<=rsi<=55: prob+=0.15
        if vol>1.3: prob+=0.15
        if vol>1.8: prob+=0.10
    return sig, min(prob,0.95), sl, tp, c, atr

def open_position(sig, sl, tp, c, atr):
    bal = exchange.fetch_balance()
    usdt = float(bal['USDT']['total']) if 'USDT' in bal else 5
    risk_amt = usdt * RISK_PER_TRADE  # 2% من 5$ = 0.10$
    risk_unit = abs(c - sl)
    if risk_unit <= 0: return

    # حجم الصفقة (مع الرافعة)
    size = (risk_amt / risk_unit) * LEVERAGE
    mn = exchange.market(SYMBOL).get('limits',{}).get('amount',{}).get('min',0.0001)
    size = max(size, mn)

    # فحص القيمة الاسمية (Notional Value)
    notional = size * c
    if notional < 10:  # OKX يحتاج حد أدنى 10 USDT للصفقة
        size = 10 / c  # نرفع الحجم ليصل الحد الأدنى
        notional = size * c
        send_telegram(f"⚠️ تم رفع الحجم لتجاوز الحد الأدنى ({notional:.2f} USDT)")

    side = 'buy' if sig=='BUY' else 'sell'
    try:
        order = exchange.create_order(SYMBOL, 'market', side, size)
        fees_est = size * c * FEE_RATE * 2
        real_risk = (risk_amt / LEVERAGE) * (size * c / notional)
        send_telegram(f"🥇 صفقة {sig} | حساب 5$\n"
                     f"الرمز: {SYMBOL}\n"
                     f"السعر: {c:.2f}\n"
                     f"الوقف: {sl:.2f} (-{risk_unit:.2f})\n"
                     f"الهدف: {tp:.2f} (+{risk_unit*RISK_REWARD:.2f})\n"
                     f"الحجم: {size:.6f} | قيمة {notional:.2f} USDT\n"
                     f"رافعة {LEVERAGE}x | رسوم ~{fees_est:.4f}$")
        
        # أوامر SL/TP
        try:
            if sig=='BUY':
                exchange.create_order(SYMBOL,'stop','sell',size,sl,params={'stopPrice':sl,'reduceOnly':True})
                exchange.create_order(SYMBOL,'take_profit','sell',size,tp,params={'takeProfitPrice':tp,'reduceOnly':True})
            else:
                exchange.create_order(SYMBOL,'stop','buy',size,sl,params={'stopPrice':sl,'reduceOnly':True})
                exchange.create_order(SYMBOL,'take_profit','buy',size,tp,params={'takeProfitPrice':tp,'reduceOnly':True})
        except Exception as e:
            print(f"SL/TP failed: {e}")
    except Exception as e:
        send_telegram(f"❌ فشل فتح صفقة: {e}")

def manage_position(sig):
    try:
        pos = exchange.fetch_positions([SYMBOL])
    except: return
    for p in pos:
        contracts = float(p.get('contracts',0) or 0)
        if contracts <= 0: continue
        side = p.get('side')
        pnl = float(p.get('unrealizedPnl',0))
        
        # إغلاق إذا الإشارة معاكسة أو الربح وصل الهدف
        if sig and ((side=='long' and sig=='SELL') or (side=='short' and sig=='BUY')):
            s = 'sell' if side=='long' else 'buy'
            exchange.create_order(SYMBOL,'market',s,contracts,params={'reduceOnly':True})
            state['wins' if pnl>0 else 'losses'] += 1
            send_telegram(f"🥇 إغلاق صفقة\n"
                         f"PnL: {pnl:.4f}$ ({pnl*530:.2f} ﷼)\n"
                         f"سجل اليوم: {state['wins']}✅ / {state['losses']}❌\n"
                         f"الرصيد الجديد: {state['balance']:.4f}$")

def run():
    print("🤖 فحص استراتيجية الـ 5$...")
    df = analyze(fetch_candles())
    sig, prob, sl, tp, c, atr = signal_and_prob(df)
    
    bal = exchange.fetch_balance()
    state['balance'] = float(bal['USDT']['total']) if 'USDT' in bal else 0
    
    print(f"📊 {SYMBOL} | {c:.2f}$ | Sig={sig} | Prob={prob:.2f} | Balance={state['balance']:.2f}$")

    manage_position(sig)

    if sig and prob >= PROB_THRESHOLD and state['trades'] < MAX_TRADES_DAY:
        open_position(sig, sl, tp, c, atr)
        state['trades'] += 1
    elif sig and prob < PROB_THRESHOLD:
        print(f"⏳ إشارة ضعيفة ({prob:.2f} < {PROB_THRESHOLD})")
    elif state['trades'] >= MAX_TRADES_DAY:
        print("⏳ اكتمل الحد اليومي")

    with open(STATE_FILE,'w') as f: json.dump(state, f)

if __name__ == "__main__":
    try: run()
    except Exception as e:
        print(f"⚠️ خطأ: {e}")
        import traceback; traceback.print_exc()

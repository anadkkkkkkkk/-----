import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from datetime import datetime

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(msg):
    if TELEGRAM_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        try:
            requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
        except Exception as e:
            print(f"Telegram error: {e}")

def get_multi_tf_data():
    symbol = "GC=F"
    tfs = {'1m': '7d', '5m': '1mo', '15m': '1mo', '1h': '2y', '1d': '5y'}
    df_main = None
    
    for tf, period in tfs.items():
        try:
            df = yf.download(tickers=symbol, period=period, interval=tf, progress=False)
            if df.empty:
                continue
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
            
            # حساب المؤشرات للفريم
            df[f'rsi_{tf}'] = 100 - (100 / (1 + df['close'].diff().clip(lower=0).rolling(14).mean() / (-df['close'].diff().clip(upper=0).rolling(14).mean())))
            df[f'sma_20_{tf}'] = df['close'].rolling(20).mean()
            df[f'sma_50_{tf}'] = df['close'].rolling(50).mean()
            df[f'return_{tf}'] = df['close'].pct_change()
            
            if tf == '15m':
                df_main = df[['close', 'high', 'low', f'rsi_{tf}', f'sma_20_{tf}', f'sma_50_{tf}', f'return_{tf}']].copy()
            else:
                cols = [f'rsi_{tf}', f'sma_20_{tf}', f'sma_50_{tf}', f'return_{tf}']
                df_main = pd.merge_asof(df_main.sort_index(), df[cols].sort_index(), left_index=True, right_index=True, direction='backward')
        except Exception as e:
            print(f"Error fetching {tf}: {e}")
            
    if df_main is not None:
        df_main['target'] = np.where(df_main['close'].shift(-1) > df_main['close'], 1, 0)
        df_main.dropna(inplace=True)
    return df_main

def main():
    df = get_multi_tf_data()
    if df is None or len(df) < 100:
        print("Data insufficient.")
        return

    features = [c for c in df.columns if c not in ['target', 'close', 'high', 'low']]
    X = df[features]
    y = df['target']
    
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    cat = CatBoostClassifier(iterations=100, depth=5, verbose=0, random_seed=42)

    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    cat.fit(X_train, y_train)

    last_row = X.iloc[[-1]]
    p_xgb = xgb.predict_proba(last_row)[0][1]
    p_rf = rf.predict_proba(last_row)[0][1]
    p_cat = cat.predict_proba(last_row)[0][1]

    # ترجيح الأوزان الذكي
    final_prob = (p_xgb * 0.45) + (p_rf * 0.30) + (p_cat * 0.25)
    current_price = df['close'].iloc[-1]
    
    # تسجيل الدقة
    acc_xgb = xgb.score(X_test, y_test)
    acc_rf = rf.score(X_test, y_test)
    acc_cat = cat.score(X_test, y_test)
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open("accuracy_log.txt", "a") as f:
        f.write(f"{now},{acc_xgb:.4f},{acc_rf:.4f},{acc_cat:.4f}\n")

    # الشروط والموافقة بين الفريمات
    if final_prob > 0.65:
        msg = f"🥇 إشارة شراء ذهب (كل الفريمات)\n💰 السعر الحالي: {current_price:.2f}\n🎯 نسبة الثقة: {final_prob*100:.1f}%\n📊 تحليل: 1M, 5M, 15M, 1H, 1D"
        send_telegram(msg)
    elif final_prob < 0.35:
        msg = f"🔻 إشارة بيع ذهب (كل الفريمات)\n💰 السعر الحالي: {current_price:.2f}\n🎯 نسبة الثقة: {(1-final_prob)*100:.1f}%\n📊 تحليل: 1M, 5M, 15M, 1H, 1D"
        send_telegram(msg)

if __name__ == "__main__":
    main()

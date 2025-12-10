# Quick Start Guide (For Non-Coders)

## Welcome!

This guide will help you get started with the MES Trading System without needing to write any code. You'll learn how to use the visual interface to test trading strategies.

---

## What You'll Need

1. **A computer** with Windows, Mac, or Linux
2. **Internet connection** (for installation only)
3. **30 minutes** to get everything set up

**No coding experience required!**

---

## Step 1: Installation (One-Time Setup)

### Install Python

1. Go to [python.org/downloads](https://python.org/downloads)
2. Download Python 3.11 or newer
3. Run the installer
4. **Important**: Check the box that says "Add Python to PATH"
5. Click "Install Now"

### Install Node.js

1. Go to [nodejs.org](https://nodejs.org)
2. Download the LTS version (left button)
3. Run the installer
4. Use all default settings

### Download the Trading System

1. Go to the GitHub repository (link provided by your team)
2. Click the green "Code" button
3. Click "Download ZIP"
4. Extract the ZIP file to your Desktop

---

## Step 2: Start the System

### Start the Backend (Server)

1. **Open Command Prompt** (Windows) or **Terminal** (Mac/Linux)
2. **Navigate to the folder**:
   ```
   cd Desktop/mlang
   ```
   (Replace "mlang" with the actual folder name)

3. **Install dependencies** (first time only):
   ```
   pip install -r requirements.txt
   ```
   Wait for it to finish (may take 5-10 minutes)

4. **Start the server**:
   ```
   uvicorn src.api:app --reload --port 8000
   ```

5. **Look for this message**: `Application startup complete`
6. **Keep this window open!** The server is now running.

### Start the Frontend (User Interface)

1. **Open another Command Prompt/Terminal window**
2. **Navigate to the frontend folder**:
   ```
   cd Desktop/mlang/frontend
   ```

3. **Install dependencies** (first time only):
   ```
   npm install
   ```
   Wait for it to finish (may take 5-10 minutes)

4. **Start the interface**:
   ```
   npm run dev
   ```

5. **Look for**: `Local: http://localhost:5173`
6. **Open your web browser** and go to: `http://localhost:5173`

---

## Step 3: Using the Trading Interface

### Option A: Visual Playback (Recommended First)

This lets you watch AI detect setups and execute trades on real market data.

#### Load Some Data

1. Click the **"YFinance Playback"** tab at the top
2. Make sure **"Use Mock Data"** is checked ✓
3. Click **"Load Data"** button
4. Wait 2-3 seconds for data to load
5. You should see "✓ Loaded 500 bars" below the button

#### Start the Playback

1. Click the big green **"Play"** button
2. Watch the chart move forward tick by tick
3. Look for:
   - **Green light** glowing = AI detected a setup
   - **Dashed lines** = Orders placed
   - **Solid lines** = Trade entered
   - **PnL numbers** updating on the left

#### What You're Seeing

- **Model Confidence %**: How sure the AI is about the setup
- **ATR**: Measures market volatility (higher = bigger moves)
- **Realized PnL**: Profit/Loss from closed trades
- **Floating PnL**: Profit/Loss from open positions

#### Experiment!

Try adjusting these controls on the left:

- **Sensitivity Slider**: 
  - Left (lower %) = More trades, less selective
  - Right (higher %) = Fewer trades, more selective

- **Speed Slider**: 
  - Left = Slower playback (good for learning)
  - Right = Faster playback (good for testing)

- **Limit/Stop Factors**:
  - Higher = Wider orders, more room
  - Lower = Tighter orders, closer entries

### Option B: Pattern Generator

This shows you how the AI generates synthetic price data.

1. Click **"Pattern Generator"** tab at the top
2. Select a date from the dropdown
3. Choose a timeframe (start with "5 Minutes")
4. Click **"Generate Session"**
5. Watch synthetic data overlay on real data

---

## Step 4: Understanding Your Results

### Reading the Stats (Left Panel)

**Realized PnL**:
- Green number = Making money
- Red number = Losing money
- This is from trades that finished

**Floating PnL**:
- Current value of open positions
- Changes as price moves
- Not "real" until trade closes

**Win Rate**:
- Percentage of winning trades
- Appears after some trades close
- 55-60% is good for this strategy

**Open/Pending/Closed**:
- Open = Active positions
- Pending = Orders waiting to fill
- Closed = Finished trades

### The Green Light

This is the **AI Signal Indicator**:

- **Bright Green** = High confidence, trade likely
- **Dim Green** = Low confidence, probably no trade
- **Dark** = No signal, AI is just watching

### Trade Markers on Chart

- **Yellow dashed line** = Sell limit order (short)
- **Blue dashed line** = Buy limit order (long)
- **Solid line with markers** = Active position with stop & target
- **Green dot** = Trade closed for profit
- **Red dot** = Trade closed for loss

---

## Step 5: Tips for Success

### Start Simple
1. Use Mock Data first (no rate limits, unlimited testing)
2. Set speed to 500-1000ms (easy to follow)
3. Keep sensitivity at 15-20% initially
4. Watch at least 50 bars before judging results

### What to Look For

**Good Signs**:
- Signals appear at logical times (after big moves)
- Orders fill and reach targets reasonably
- Win rate stays above 50%
- PnL trends upward over time

**Warning Signs**:
- Signals firing constantly (sensitivity too low)
- No trades happening (sensitivity too high)
- Orders never filling (limit factor too high)
- All trades hitting stop loss (setup not working)

### Common Mistakes

❌ Judging after only 5-10 trades (need 50+ for reliability)  
❌ Changing settings mid-test (restart instead)  
❌ Expecting 100% win rate (60% is excellent)  
❌ Not using mock data for practice first  

---

## Step 6: Next Steps

### Once You're Comfortable

1. **Try Real Data**:
   - Uncheck "Use Mock Data"
   - Enter a symbol like "MES=F" or "ES=F"
   - Load data and test

2. **Try Different Models**:
   - Click the "Model" dropdown
   - Select different CNN models
   - Compare which works best

3. **Adjust Parameters**:
   - Experiment with Limit Factor (1.0-2.0)
   - Try different Stop Factors (0.5-1.5)
   - Find what works for your style

### Understanding Different Setups

The system currently implements **Mean Reversion**:
- Price moves too far, likely to snap back
- Works best in ranging markets
- Targets return to the middle

**Future setups to implement**:
- Breakouts (momentum)
- Rejections (reversal)
- Trend following

See `docs/SETUP_LIBRARY.md` for details.

---

## Troubleshooting

### "Can't connect" or white screen
**Fix**: Make sure both backend AND frontend are running  
Check both Command Prompt/Terminal windows are still open

### "No models available"
**Fix**: The system needs trained models  
Ask your team about pre-trained model files

### Playback is frozen
**Fix**: 
- Check if data is loaded (should say "✓ Loaded X bars")
- Try clicking Reset then Play again
- Close browser and reopen

### Data won't load
**Fix**:
- If using real data, try mock data instead
- Check internet connection
- Wait 5 minutes and try again (rate limits)

### Everything is slow
**Fix**:
- Use fewer days (try 3 instead of 7)
- Use mock data instead of real
- Restart the backend server
- Close other programs

---

## Learning More

### Documentation
- **For Concepts**: Read `docs/USER_GUIDE.md`
- **For Architecture**: Read `docs/ARCHITECTURE.md`
- **For Trading Setups**: Read `docs/SETUP_LIBRARY.md`

### Key Concepts to Learn

1. **Setup** = Market condition (e.g., "price too high")
2. **Signal** = AI decision to trade (e.g., "65% confidence")
3. **Entry** = How we enter (limit order vs market)
4. **Exit** = How we close (stop loss or take profit)

### Practice Exercises

**Exercise 1**: Watch 100 bars at normal speed
- Count how many signals fired
- Note win rate at the end
- Document what you noticed

**Exercise 2**: Compare sensitivities
- Run at 10% sensitivity, note results
- Reset, run at 50% sensitivity
- Which had better win rate?

**Exercise 3**: Test different markets
- Try MES=F (micro S&P)
- Try ES=F (full S&P)
- Try AAPL (Apple stock)
- Which works best?

---

## Getting Help

### If You're Stuck

1. **Read the error message** carefully
2. **Check both windows** (backend and frontend) for errors
3. **Try restarting** everything
4. **Use mock data** to isolate the issue

### Asking for Help

When asking your team, include:
- What you were trying to do
- What button you clicked
- Any error messages (screenshot is helpful)
- Whether you're using real or mock data

---

## Safety Notes

### This is a TESTING System

⚠️ **Important**:
- This does NOT connect to any brokerage
- No real money is at risk
- This is for LEARNING and VALIDATION only
- To trade real money, you need separate software

### Data Privacy

- Your data stays on your computer
- Nothing is sent to external servers
- Mock data is generated locally

---

## Appendix: Quick Reference

### Starting the System

```
# Terminal/Command Prompt 1 (Backend)
cd Desktop/mlang
uvicorn src.api:app --reload --port 8000

# Terminal/Command Prompt 2 (Frontend)
cd Desktop/mlang/frontend
npm run dev

# Browser
http://localhost:5173
```

### Stopping the System

1. Close the browser tab
2. Go to each Terminal/Command Prompt
3. Press `Ctrl+C` (Windows/Linux) or `Cmd+C` (Mac)
4. Wait for it to stop
5. Close the windows

### Default Settings

- Sensitivity: 15%
- Limit Factor: 1.5
- Stop Factor: 1.0
- Playback Speed: 200ms
- Model: CNN_Predictive_5m

---

## Congratulations!

You've learned how to:
✅ Install and start the trading system  
✅ Load data and run playback  
✅ Read signals and understand results  
✅ Adjust settings and experiment  
✅ Troubleshoot common issues  

**Next Steps**: Practice with different settings, read the user guide, and explore other features!

---

**Questions?** Check `docs/USER_GUIDE.md` or ask your team.

**Last Updated**: 2025-12-10

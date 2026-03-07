# Windows Task Scheduler Setup Guide

## Complete Step-by-Step Instructions

Follow these exact steps to set up automated daily portfolio updates.

---

## Step 1: Open Task Scheduler

**Method 1 (Fastest):**
1. Press `Windows Key + R`
2. Type: `taskschd.msc`
3. Press `Enter`

**Method 2 (Via Search):**
1. Press `Windows Key`
2. Type: "Task Scheduler"
3. Click "Task Scheduler" app

---

## Step 2: Create Morning Task (8:30 AM - Pre-Market)

### 2.1 Start Creating Task

1. In Task Scheduler window, click **"Create Task..."** in right panel
   - ⚠️ NOT "Create Basic Task" - use "Create Task"

### 2.2 General Tab

Fill in these fields:

**Name:** `Portfolio Update - Pre-Market`

**Description:** `Update portfolio data before market opens (8:30 AM ET)`

**Security options:**
- ✅ Check: "Run whether user is logged on or not"
- ✅ Check: "Run with highest privileges"
- ⚠️ DON'T check "Hidden" (you want to see if it runs)

**Configure for:** Windows 10 (or your Windows version)

### 2.3 Triggers Tab

1. Click **"New..."** button at bottom

2. Fill in:
   - **Begin the task:** `On a schedule`
   - **Settings:** `Daily`
   - **Recur every:** `1` days
   - **Start:** Set to `8:30:00 AM`
   - **Start date:** Today's date

3. **Advanced settings:**
   - ❌ Uncheck: "Delay task for up to (random delay)"
   - ✅ Check: "Stop task if it runs longer than:" → `30 minutes`
   - ✅ Check: "Enabled"

4. Click **OK**

### 2.4 Actions Tab

1. Click **"New..."** button

2. Fill in:
   - **Action:** `Start a program`

3. **Program/script:** `python`
   - Just type: `python` (no path needed if Python is in PATH)

4. **Add arguments:** `run_daily_update.py`

5. **Start in:** `C:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main`
   - ⚠️ Use YOUR actual path (copy from File Explorer address bar)
   - ⚠️ No quotes needed

6. Click **OK**

### 2.5 Conditions Tab

1. **Power:**
   - ❌ Uncheck: "Start the task only if the computer is on AC power"
   - ❌ Uncheck: "Stop if the computer switches to battery power"
   - ✅ Check: "Wake the computer to run this task"

2. **Network:**
   - ✅ Check: "Start only if the following network connection is available"
   - Select: "Any connection"

### 2.6 Settings Tab

1. **General:**
   - ✅ Check: "Allow task to be run on demand"
   - ✅ Check: "Run task as soon as possible after a scheduled start is missed"
   - ❌ Uncheck: "Stop the task if it runs longer than:" (we set this in Triggers)

2. **If the task fails, restart every:** `5 minutes`
3. **Attempt to restart up to:** `3` times

4. **If the running task does not end when requested, force it to stop:** ✅ Check

5. Click **OK**

### 2.7 Enter Password

1. Windows will ask for your password
2. Enter your Windows password
3. Click **OK**

✅ **Morning task created!**

---

## Step 3: Create Evening Task (4:30 PM - Post-Market)

### Quick Method: Duplicate & Edit

1. In Task Scheduler, find your "Portfolio Update - Pre-Market" task
2. Right-click → **Copy**
3. Right-click empty space → **Paste**

4. Right-click the copy → **Properties**

5. **General Tab:**
   - Change **Name:** to `Portfolio Update - Post-Market`
   - Change **Description:** to `Update portfolio data after market closes (4:30 PM ET)`

6. **Triggers Tab:**
   - Double-click the trigger
   - Change **Start:** to `4:30:00 PM`
   - Click **OK**

7. Click **OK** to save

8. Enter your password again

✅ **Evening task created!**

---

## Step 4: Test Your Tasks

### Test Morning Task

1. Find "Portfolio Update - Pre-Market" in Task Scheduler
2. Right-click → **Run**
3. Watch the **Status** column change to "Running"
4. Wait ~20 seconds
5. Status should change to "Ready"
6. **Last Run Result:** Should show "The operation completed successfully (0x0)"

### Test Evening Task

1. Find "Portfolio Update - Post-Market" in Task Scheduler
2. Right-click → **Run**
3. Verify it completes successfully

### Check Logs

```powershell
# View today's log
type "C:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main\logs\daily_update_20260307.log"
```

You should see:
```
[SUCCESS] All steps completed successfully in 15.7s
[SUCCESS] Database updated: outputs/allocations.db
```

---

## Step 5: Verify Automation

### Check Task Scheduler History

1. In Task Scheduler, click your task
2. Click **History** tab at bottom
3. You should see events:
   - Event ID 100: Task Started
   - Event ID 200: Action Started
   - Event ID 201: Action Completed
   - Event ID 102: Task Completed

### Wait for Scheduled Run

1. Leave computer on at 8:30 AM tomorrow
2. Check logs folder for new log file
3. Verify task ran automatically

---

## Troubleshooting

### "Task Scheduler is not running"

**Fix:**
```powershell
# Check if Task Scheduler service is running
Get-Service -Name "Task Scheduler"

# If not running, start it
Start-Service -Name "Task Scheduler"
```

### "Access is denied" when creating task

**Fix:**
1. Right-click Task Scheduler app
2. Click "Run as administrator"
3. Try again

### Task shows "Could not start" in Last Run Result

**Fix 1: Check Python path**
```powershell
# Verify Python is in PATH
python --version

# If not found, use full path in Action:
# C:\Users\dns81\AppData\Local\Programs\Python\Python311\python.exe
```

**Fix 2: Check working directory**
- Make sure "Start in" path is correct
- Copy from File Explorer address bar

### Task runs but nothing happens

**Check logs:**
```powershell
# Check if log was created
dir "C:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main\logs\"

# View latest log
Get-Content "C:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main\logs\*.log" | Select-Object -Last 50
```

### Task says "The task is currently running (0x41301)"

**Fix:**
- Task is still running, wait for it to finish
- If stuck, right-click → End

---

## Alternative: PowerShell Script Method

If the above doesn't work, use PowerShell to create tasks:

```powershell
# Run PowerShell as Administrator

# Morning task
$action = New-ScheduledTaskAction `
    -Execute "python" `
    -Argument "run_daily_update.py" `
    -WorkingDirectory "C:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main"

$trigger = New-ScheduledTaskTrigger -Daily -At 8:30AM

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType ServiceAccount `
    -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask `
    -TaskName "Portfolio Update - Pre-Market" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings

# Evening task
$trigger2 = New-ScheduledTaskTrigger -Daily -At 4:30PM

Register-ScheduledTask `
    -TaskName "Portfolio Update - Post-Market" `
    -Action $action `
    -Trigger $trigger2 `
    -Principal $principal `
    -Settings $settings

Write-Host "Tasks created successfully!"
```

---

## Verification Checklist

- [ ] Task Scheduler shows both tasks
- [ ] Both tasks have status "Ready"
- [ ] Manual run completes successfully
- [ ] Logs are being created
- [ ] Database is being updated
- [ ] Task History shows no errors

---

## Quick Reference

**Open Task Scheduler:** `Win + R` → `taskschd.msc`

**Test task:** Right-click → Run

**View logs:** `logs\daily_update_YYYYMMDD.log`

**Check database:** `python -c "from src.database import Database; db = Database(); print(db.get_latest_backtest_results()); db.close()"`

---

## You're Done! 🎉

Your portfolio will now update automatically:
- **8:30 AM ET** every day (before market)
- **4:30 PM ET** every day (after market)

Check logs daily for first week to ensure stability.

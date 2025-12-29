# Wandb API Key Setup Guide

## 🔑 Where to Put Your Wandb API Key

You have **3 options** (recommended order):

---

## ✅ Option 1: Command Line Login (RECOMMENDED - Most Secure)

**Best practice**: Use interactive login, stores key securely.

### Steps:
```bash
# 1. Run this command
wandb login

# 2. You'll be prompted:
#    Paste an API key from your profile: https://wandb.ai/authorize
#    You can find your API key at: https://wandb.ai/settings

# 3. Paste your API key and press Enter
#    Output: Successfully logged in to Weights & Biases
```

**Where it's stored**: `~/.netrc` (secure file, not in code)

**Advantage**: 
- ✅ Most secure (key not in code)
- ✅ Works automatically after login
- ✅ No need to modify code

**How it works**: 
- Wandb automatically reads from `~/.netrc` when you run `wandb.init()`
- No code changes needed!

---

## ✅ Option 2: Environment Variable (Good for Scripts)

**Useful for**: Automated scripts, CI/CD

### Steps:
```bash
# Set environment variable (temporary for current session)
export WANDB_API_KEY="your-api-key-here"

# Or add to ~/.bashrc or ~/.zshrc (permanent)
echo 'export WANDB_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Advantage**: 
- ✅ Key not in code
- ✅ Works with scripts
- ✅ Can be set per project

**How to use**: Just run training normally, wandb will use the env variable.

---

## ⚠️ Option 3: Add to Code (Like Your Example - Less Secure)

**Similar to your `IDS_CICIDS2017.py` example**

### Steps:
1. Edit `train.py`
2. Add login before `wandb.init()`:

```python
# In train.py, around line 199, add:
if use_wandb:
    wandb.login(key="your-api-key-here")  # Add this line
    wandb.init(
        project=config['logging'].get('wandb_project', 'qr-phishing-detection'),
        ...
    )
```

**Disadvantage**: 
- ❌ API key visible in code
- ❌ Not secure if you share code
- ❌ Need to remember to remove before committing

**Better alternative**: Use config.yaml (see Option 4)

---

## ✅ Option 4: Add to config.yaml (Convenient but Less Secure)

**Useful for**: Quick setup, local development

### Steps:
1. Edit `config.yaml`:
```yaml
logging:
  use_wandb: true
  wandb_project: "qr-phishing-detection"
  wandb_entity: null
  wandb_api_key: "your-api-key-here"  # Add this line
```

2. Update `train.py` to read from config:
```python
if use_wandb:
    api_key = config['logging'].get('wandb_api_key')
    if api_key:
        wandb.login(key=api_key)
    wandb.init(...)
```

**Advantage**: 
- ✅ Easy to configure
- ✅ Can be different per project

**Disadvantage**: 
- ⚠️ Key in config file (add to .gitignore!)

---

## 🎯 RECOMMENDED: Option 1 (Command Line)

**Just run once:**
```bash
wandb login
# Paste your API key when prompted
```

**Then run training normally:**
```bash
uv run train.py
```

**That's it!** No code changes needed.

---

## 📝 How to Get Your API Key

1. Go to: https://wandb.ai/authorize
   OR
   Go to: https://wandb.ai/settings → API keys

2. Copy your API key (looks like: `b6fd84e6b68bef7524f6d74e5bc9842f45ac4304`)

3. Use it with one of the options above

---

## 🔒 Security Best Practices

1. **Never commit API keys to git**
   - Add to `.gitignore` if using config.yaml
   - Don't hardcode in code you'll share

2. **Use Option 1 (command line)** for best security

3. **If using config.yaml**, add to `.gitignore`:
   ```
   # In .gitignore
   config.yaml  # Or just the wandb_api_key line
   ```

---

## ✅ Quick Setup (Recommended)

```bash
# 1. Login once (stores key securely)
wandb login

# 2. Enable in config.yaml
# Set: use_wandb: true

# 3. Run training
uv run train.py
```

**No code changes needed!** Wandb will automatically use your logged-in credentials.

---

## 🐛 Troubleshooting

### "Not logged in" error?
```bash
# Check if logged in
wandb login --relogin

# Or verify
wandb status
```

### Want to use different key?
```bash
wandb login --relogin
# Enter new key
```

### Key in wrong place?
- Check `~/.netrc` file (should have wandb entry)
- Or check environment variable: `echo $WANDB_API_KEY`

